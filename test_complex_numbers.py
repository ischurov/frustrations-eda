import torch
assert torch.cuda.is_available()
device = torch.device("cuda:0")
print(torch.exp(torch.tensor([1, 2, 3], device=device, dtype=torch.complex128)))
print("OK")
