import torch

# Check if MPS is available
print("MPS available:", torch.backends.mps.is_available())

# Create a tensor and move to device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.randn(2, 2).to(device)
print("Tensor device:", x.device)

# Check model device
# model is your PyTorch model
# print("Model device:", next(model.parameters()).device)
