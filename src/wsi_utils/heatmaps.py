from PIL import Image
from openslide import OpenSlide
from src.run_tissue_seg_single import mag_layers


def patch_wsi(wsi_path, patch_size, overlap, mirroring_type, mag, save = None):
    """
    Split WSI into patches.
    :param wsi_path:
    :param patch_size:
    :param overlap:
    :param mirroring_type:
    :param mag: Desired slide magnification.
    :param save: Where to save the patches. If None, patches won't be saved.'
    :return: numpy array of patches.
    """
    wsi = OpenSlide(wsi_path)

def rescale_wsi(wsi, desired_mag, rescale_method = Image.LANCZOS, verbose = False, allow_upscaling = True):
    # TODO: ? is it ok to do upscaling? when we want 40x and we have only 20x? probably not a good idea. ?
    """"
    Rescale the WSI.
    :param wsi: OpenSlide wsi object
    :param desired_mag: Desired slide magnification.
    :param rescale_method: Method to downscale WSI if the desired magnification is not available. You can choose from 
    PIL.Image.Resampling (Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.HAMMING, Image.LANCZOS, Image.NEAREST)
    :param allow_upscaling: Allow for image upscaling when desired magnification is higher than the highest magnification available.
    :param verbose: print info messages or not
    :return: rescaled wsi region.
    """
    
    ratio = wsi.level_downsamples
    mag_l0 = float(wsi.properties["openslide.objective-power"])
    mag_layers = [round(mag_l0/r, 2) for r in ratio]
    
    if desired_mag in mag_layers:
        info = "Desired magnification is available"
        mag_idx = mag_layers.index(desired_mag)
        w, h = wsi.level_dimensions[mag_idx]
        region = wsi.read_region((0, 0), mag_idx, (w, h))
        scale_val = ratio[mag_idx]
    else:
        if desired_mag < mag_l0:
            info = "Desired slide magnification is larger than available, image will be magnified."
            if not allow_upscaling:
                raise ValueError("The desired magnification is smaller than the highest magnification available. "
                                 "The parameter allow_upscaling is set to False, so the image will not be upscaled. "
                                 "If you want to upscale the image, set the parameter allow_upscaling to True. ")
        else:
            info = "desired resolution is not available, image will be rescaled from the highest resolution"
        mag_idx = 0  # get the highest magnification and rescale
        w0, h0 = wsi.level_dimensions[mag_idx]
        region = wsi.read_region((0, 0), mag_idx, (w0, h0))
        scale_val = desired_mag/ mag_l0
        region = region.resize((int(w0 * scale_val), int(h0 * scale_val)), rescale_method)
    
    if verbose:
        print(info)
        
    return region, scale_val, info
    
    

    





