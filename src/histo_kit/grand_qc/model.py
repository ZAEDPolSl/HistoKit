import torch

class GrandQC(torch.nn.Module):
    """
    Wrapper module for a pre-trained GrandQC model.

    Parameters
    ----------
    gqc_weights : str or pathlib.Path
        Path to the `.pt` or `.pth` file containing the serialized GrandQC model.
    device : str, optional
        Device on which to load the model "cuda" or "cpu" (default is ``"cuda"``).

    Attributes
    ----------
    grandQC : torch.nn.Module
        The loaded GrandQC model instance.

    Examples
    --------
    >>> model = GrandQC("grandqc.pth", device="cpu")
    >>> x = torch.randn(1, 3, 224, 224)
    >>> y = model(x)
    """
    def __init__(self, gqc_weights, device="cuda"):
        super().__init__()
        self.grandQC = torch.load(gqc_weights, map_location=device)

    def forward(self, x):
        """
        Run a forward pass through the GrandQC model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor shaped appropriately for the underlying model
            (e.g., ``(N, C, H, W)`` for images).

        Returns
        -------
        torch.Tensor
            The model output tensor.
        """
        y = self.grandQC(x)
        return y