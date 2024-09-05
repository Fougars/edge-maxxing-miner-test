import torch
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from diffusers import DiffusionPipeline
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
MODEL_ID = "stablediffusionapi/newdream-sdxl-20"

class TensorRTEngine:
    def __init__(self, onnx_file_path, precision='fp16', max_batch_size=256):
        self.engine = self.build_engine(onnx_file_path, precision, max_batch_size)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(max_batch_size)

    def build_engine(self, onnx_file_path, precision='fp16', max_batch_size=256):
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config = builder.create_builder_config()
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        config.max_workspace_size = 24 * (1 << 30)  # 24 GB
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 4, 128, 128), (max_batch_size // 2, 4, 128, 128), (max_batch_size, 4, 128, 128))
        config.add_optimization_profile(profile)

        return builder.build_engine(network, config)

    def allocate_buffers(self, max_batch_size):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            shape = (max_batch_size,) + shape[1:]
            size = trt.volume(shape) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        batch_size = input_data.shape[0]
        self.context.set_binding_shape(0, (batch_size,) + input_data.shape[1:])
        cuda.memcpy_htod_async(self.inputs[0].device, input_data.ravel(), self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)
        self.stream.synchronize()
        return self.outputs[0].host.reshape((batch_size,) + self.engine.get_binding_shape(1)[1:])

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

def optimize_model(model, input_shape, precision='fp16', max_batch_size=256):
    model = model.eval().half().cuda()

    # Pruning
    from torch.nn.utils import prune
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)

    # Quantization
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    # Export to ONNX
    dummy_input = torch.randn(*input_shape).half().cuda() 
    onnx_file = f"{model.__class__.__name__}.onnx"
    torch.onnx.export(model, dummy_input, onnx_file, opset_version=13,
                      do_constant_folding=True, input_names=['input'],  
                      output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 
                                                             'output': {0: 'batch_size'}})

    # Create TensorRT engine
    trt_engine = TensorRTEngine(onnx_file, precision, max_batch_size)

    def inference_fn(input_data):
        return trt_engine.infer(input_data)

    return inference_fn

def optimize(pipeline: DiffusionPipeline) -> DiffusionPipeline:
    max_batch_size = 256  
    precision = 'fp16'  

    # Optimize for RTX 4090
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    pipeline.unet = optimize_model(pipeline.unet, (2, 4, 128, 128), precision, max_batch_size)
    pipeline.text_encoder = optimize_model(pipeline.text_encoder, (1, 77), precision, max_batch_size)
    pipeline.vae.decoder = optimize_model(pipeline.vae.decoder, (1, 4, 128, 128), precision, max_batch_size)
    pipeline.vae.encoder = optimize_model(pipeline.vae.encoder, (1, 3, 1024, 1024), precision, max_batch_size)

    return pipeline

def main():
    print("Starting optimization process...")
    
    start_time = time.time()
    pipeline = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    pipeline = optimize(pipeline)
    optimization_time = time.time() - start_time
    print(f"Optimization completed in {optimization_time:.2f} seconds")

    # Test the optimized pipeline
    prompt = "A beautiful sunset over a calm ocean"
    print(f"Generating image for prompt: '{prompt}'")
    image = pipeline(prompt).images[0]
    image.save("test_output.png")
    print("Image generated and saved as 'test_output.png'")

if __name__ == '__main__':
    main()
