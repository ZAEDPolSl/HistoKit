import numpy as np
from torch.utils.data import Dataset
from .artifacts import Artifact
import segmentation_models_pytorch as smp
from ..utils.image import to_tensor_x
from ..utils.patches import get_patch_grid


class GrandQCDataset(Dataset):
    """
    Pytorch Dataset for extracting fixed-size patches from a region while applying padding
    to boundary areas. Also returns a background mask patch and metadata describing
    the location of each patch.

    Parameters
    ----------
    region : np.ndarray
        Source RGB region image from which patches will be extracted.
        Expected shape is ``(H, W, 3)``.
    bg : np.ndarray
        Background mask associated with `region`, matching spatial dimensions
        ``(H, W)``.
    bbox_list : list of tuples
        List of bounding boxes defining areas of interest. Each bounding box should be
        represented as ``(x_start, y_start, x_end, y_end)``.
    patch_size : int, optional
        Target size (height and width) for the extracted patches (default is ``512`` which is valid for the GrandQC model).
    overlap : float, optional
        Fractional overlap between neighboring patches (default is ``0.7``).
    pad_value : int, optional
        Value used to pad pixels when patches extend beyond the region boundary.
        Typically background (default is ``Artifact.BG_THR.value``, which corresponds to 0).
    encoder : str, optional
        Name of the encoder used for preprocessing, passed to
        `segmentation_models_pytorch.encoders.get_preprocessing_fn`.
    weights : str, optional
        Pre-trained weights to use with the encoder (default is ``"imagenet"``).

    Attributes
    ----------
    coords : dict
        Dictionary of patch coordinates with keys ``"x_start"``, ``"y_start"``,
        ``"x_end"``, ``"y_end"``.
    prep_fn : callable
        Preprocessing function for encoder normalization.
    patch_size : int
        Final patch spatial size.
    pad_value : int
        Background padding value.

    Notes
    -----
    Returned items are dictionaries rather than `(image, label)` pairs to allow
    downstream inference pipelines to use bounding box metadata.

    """

    def __init__(self, region, bg, bbox_list, patch_size=512, overlap=0.7, pad_value=Artifact.BG_THR.value, encoder='timm-efficientnet-b0', weights="imagenet"):

        self.bg = bg
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.region = region
        self.bg = bg
        self.prep_fn = smp.encoders.get_preprocessing_fn(encoder, weights)
        self.coords = get_patch_grid(bbox_list, patch_size=patch_size, overlap=overlap)

    def __len__(self):
        return len(self.coords["x_start"])

    def preprocess(self, img):
        """
        Apply encoder-specific preprocessing and convert the image to a tensor.

        Parameters
        ----------
        img : np.ndarray
            Input patch of shape ``(patch_size, patch_size, 3)``.

        Returns
        -------
        torch.Tensor
            Preprocessed tensor suitable for the GrandQC model input.
        """
        x = self.prep_fn(img)
        x = to_tensor_x(x)
        return x

    def __getitem__(self, idx):
        """
        Retrieve the patch and its associated metadata for a given index.

        Parameters
        ----------
        idx : int
            Index of the patch to retrieve.

        Returns
        -------
        dict
            A dictionary containing:
            - ``"patch"`` : torch.Tensor, preprocessed patch image
            - ``"patch_bg"`` : np.ndarray, background mask patch
            - ``"x_start"``, ``"y_start"``, ``"x_end"``, ``"y_end"`` : int coordinates
            - ``"all_bg"`` : bool, whether the patch is entirely background
        """

        x_start = int(self.coords["x_start"][idx])
        y_start = int(self.coords["y_start"][idx])
        x_end = int(self.coords["x_end"][idx])
        y_end = int(self.coords["y_end"][idx])

        sx0 = max(0, x_start)
        sy0 = max(0, y_start)
        sy1 = min(self.region.shape[0], y_end)
        sx1 = min(self.region.shape[1], x_end)

        patch = self.region[sy0:sy1, sx0:sx1]
        bg_patch = self.bg[sy0:sy1, sx0:sx1]

        if patch.shape[0] != self.patch_size or patch.shape[1] != self.patch_size:

            padded = np.full((self.patch_size, self.patch_size, 3), self.pad_value, dtype=np.uint8)
            padded_bg = np.full((self.patch_size, self.patch_size), self.pad_value, dtype=np.uint8)

            paste_x = max(0, -x_start)
            paste_y = max(0, -y_start)

            h_copy = min(patch.shape[0], self.patch_size - paste_y)
            w_copy = min(patch.shape[1], self.patch_size - paste_x)

            if h_copy > 0 and w_copy > 0:
                padded[paste_y:paste_y + h_copy, paste_x:paste_x + w_copy] = patch[:h_copy, :w_copy]
                padded_bg[paste_y:paste_y + h_copy, paste_x:paste_x + w_copy] = bg_patch[:h_copy, :w_copy]
            patch = padded
            bg_patch = padded_bg

        res_dict = {
            "patch": self.preprocess(patch),
            "patch_bg": bg_patch,
            "x_start": x_start,
            "y_start": y_start,
            "x_end": x_end,
            "y_end": y_end,
            "all_bg": np.all(bg_patch == self.pad_value)
        }

        return res_dict






