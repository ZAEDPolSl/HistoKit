import torch

class GrandQC(torch.nn.Module):
    def __init__(self, gqc_weights, device="cuda"):
        super().__init__()
        self.grandQC = torch.load(gqc_weights, map_location=device)

    def forward(self, x):
        y = self.grandQC(x)
        return y