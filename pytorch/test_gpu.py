import torch

def check_torch_gpu():
    print("PyTorch version:", torch.__version__)
    gpu_available = torch.cuda.is_available()
    print("CUDA available:", gpu_available)

    if gpu_available:
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected. Ensure your NVIDIA drivers and CUDA are installed correctly.")

if __name__ == "__main__":
    check_torch_gpu()
