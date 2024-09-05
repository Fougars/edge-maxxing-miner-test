import torch
import numpy as np
from diffusers import DiffusionPipeline
import time

MODEL_ID = "stablediffusionapi/newdream-sdxl-20"

def optimize_model(model, precision='fp16'):
    model = model.eval()
    if precision == 'fp16':
        model = model.half()
    model = model.cuda()

    # Pruning
    from torch.nn.utils import prune
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.2)

    # Quantization-aware training (simulated quantization)
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    return model

def optimize(pipeline: DiffusionPipeline) -> DiffusionPipeline:
    precision = 'fp16'  

    # Optimize for RTX 4090
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    pipeline.unet = optimize_model(pipeline.unet, precision)
    pipeline.text_encoder = optimize_model(pipeline.text_encoder, precision)
    pipeline.vae.decoder = optimize_model(pipeline.vae.decoder, precision)
    pipeline.vae.encoder = optimize_model(pipeline.vae.encoder, precision)

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
