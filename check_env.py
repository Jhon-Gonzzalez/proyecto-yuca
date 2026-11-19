import torch, platform
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("MPS disponible:", torch.backends.mps.is_available())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(1000, 1000, device=device)
print("Prueba en", device, "->", (x @ x).sum().item())