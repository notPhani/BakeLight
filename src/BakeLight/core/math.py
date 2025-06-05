import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normailze(v: torch.Tensor) -> torch.Tensor:
    norm = (v/ (torch.norm(v, dim=-1, keepdim=True)+ 1e-8))
    return norm.to(device)

