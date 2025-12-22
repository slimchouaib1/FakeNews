use_gpu = torch.cuda.is_available()

kwargs = {
    "device_map": settings.DEVICE_MAP,
    "torch_dtype": torch.float16 if use_gpu else torch.float32,
}

# Quantization ONLY if GPU is available
if settings.QUANT == "4bit" and use_gpu:
    kwargs["load_in_4bit"] = True
