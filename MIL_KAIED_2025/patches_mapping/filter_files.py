import os

#root = "/mnt/data/Datasets/HE_data/CPTAC_UCEC/"
#dir = "/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/CPTAC UCEC/svs/"
#files = os.listdir(dir)

#for f in files:
#    print(f)
#    with open("CPTAC_UCEC.txt", "a") as file:
#        file.write(root+f + "\n")



from src.histo_kit.utils.wsi import load_wsi_mag
import scipy.io as sio
import openslide
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

scale_factor = 5
patch_size = 256
mag_extr = 10

patches_dir = "/mnt/warehouse/Projects/HE/HE_UCEC/Data/10x/Original/TCGA_UCEC"

res = {"slide_name": [], "patch_name_org": [], "patch_name_mod":[], "x0": [], "y0": [], "x1": [], "y1": [], "region": []}

wsi_dir = f"/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/TCGA UCEC/svs/"
wsis = os.listdir(wsi_dir)

for w in wsis[165:-1]:
    slide_name = w.split(".")[0]+"."+w.split(".")[1]
    mask_path = f"/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/TCGA UCEC/tissue_region_detection_old/{slide_name}_mask_all.mat"
    slide_path = f"/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/TCGA UCEC/svs/{slide_name}.svs"
    path_res_org = f"/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/TCGA UCEC/patches/10x/original/{slide_name}/"
    path_res_check = f"/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/TCGA UCEC/patches/10x/original_check/{slide_name}/"

    os.makedirs(path_res_check, exist_ok=True)
    os.makedirs(path_res_org, exist_ok=True)

    mask = sio.loadmat(mask_path)
    svs = openslide.OpenSlide(slide_path)
    slide_mag = float(svs.properties["openslide.objective-power"])
    mag_load = slide_mag / mask["scale_val"][0][0]

    for i in range(len(mask['tiss_stats'])):

        try:
            region, scale_val, info, mpp_slide, ratio = load_wsi_mag(svs, 10, rescale_method = Image.BICUBIC)
            region_np = np.array(region)
            slide_height, slide_width = region_np.shape[:2]

            x, y, w, h = mask['tiss_stats'][i]

            scale = mag_extr/mag_load
            x0 = int(x*scale)-1
            y0 = int(y*scale)-1

            patches = f"{patches_dir}/{slide_name}_R{str(i+1)}/"
            patches_names = os.listdir(patches)

            region_img = Image.fromarray(region_np)
            region_img_draw = Image.fromarray(region_np.copy())
            draw = ImageDraw.Draw(region_img_draw)

            for p_name in patches_names:

                p_name = p_name.split(".")[0]
                _, a, b = p_name.split("_")
                row_idx = int(a) - 1
                col_idx = int(b) - 1

                y0_p = y0 + row_idx * patch_size
                x0_p = x0 + col_idx * patch_size
                x1_p = x0_p + patch_size
                y1_p = y0_p + patch_size


                p_name_mod = f"patch_{a}_{b}_{y0_p}_{x0_p}_{y1_p}_{x1_p}.png"
                res["slide_name"].append(slide_name)
                res["patch_name_org"].append(p_name+".png")
                res["patch_name_mod"].append(p_name_mod)
                res["x0"].append(x0_p)
                res["y0"].append(y0_p)
                res["x1"].append(x1_p)
                res["y1"].append(y1_p)
                res["region"].append(f"R{str(i+1)}")

                patch = region_img.crop((x0_p, y0_p, x1_p, y1_p))
                patch_org = Image.open(os.path.join(patches, p_name+".png"))
                patch_org.save(os.path.join(path_res_org, p_name_mod))
                patch.save(os.path.join(path_res_check, p_name_mod))

                draw.rectangle(
                    [x0_p, y0_p, x1_p, y1_p],
                    outline=(255, 0, 0),
                    width=15
                )

            new_size = (region_img_draw.width // scale_factor, region_img_draw.height // scale_factor)
            region_small = region_img_draw.resize(new_size, resample=Image.BICUBIC)
            region_small.save(os.path.join(path_res_check, f"{slide_name}_R{i+1}_downscaled.png"))
        except Exception as e:
            print(f"Error processing {slide_name} region {i+1}: {e}")

df = pd.DataFrame(res)
df.to_csv(f"/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/TCGA UCEC/patches/10x/mapping.csv", index=False)