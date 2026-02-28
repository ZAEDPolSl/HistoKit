
import torch
import os
import pandas as pd

models = ["CTranspath", "Hoptimus", "resnet18_SSL_histo", "UNI"]
types = ["original", "vahadane_filtered_13_02_2025"]
dataset = ["TCGA"]

for model in models:
    for t in types:
        for d in dataset:

            dir = f"/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_org/tensors/{model}/{t}/{d}/"
            dir_res = f"/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/{model}/{t}/{d}/"
            df = pd.read_csv(f"/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/{d} UCEC/patches/10x/mapping.csv")
            df_labels = pd.read_csv(f"/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/{d} UCEC/labels/{d}_UCEC_grade.csv")

            files = os.listdir(dir)
            os.makedirs(os.path.join(dir_res,"FFPE"), exist_ok=True)
            os.makedirs(os.path.join(dir_res, "OCT"), exist_ok=True)
            wsi_names = df_labels["wsi"].unique().tolist()

            for f in files:
                slide_name = f.split("_")[0]

                if slide_name not in wsi_names:
                    print(f"Skipping {slide_name}")
                    continue

                regions = [f for f in os.listdir(dir) if f.startswith(f"{slide_name}_R") and os.path.isfile(os.path.join(dir, f))]

                features = []
                patches_names = []

                for r in regions:
                    tensor = torch.load(os.path.join(dir, r))
                    features.append(tensor["features"])

                    p_names = tensor["patches_names"]
                    p_bases = [os.path.basename(t[0]) for t in p_names]

                    region = r.split("_")[-1].split(".")[0]
                    df_region = df[(df["slide_name"] == slide_name) & (df["region"] == region)]
                    mapping_dict = dict(zip(df_region['patch_name_org'], df_region['patch_name_mod']))
                    patch_bases = [mapping_dict.get(name) for name in p_bases]
                    for n in patch_bases:
                        if n is None:
                            print(f"Warning: Patch name {n} not found in mapping for slide {slide_name}, region {region}")
                    patches_names.append(patch_bases)

                features = torch.cat(features, dim=1)
                patches_names = [item for sublist in patches_names for item in sublist]
                wsi = slide_name
                grade = df_labels[df_labels["wsi"] == wsi]["grade"].values[0]

                sample = {
                    "features": features,
                    "patch_names": patches_names,
                    "wsi": wsi,
                    "grade": int(grade)
                }

                #medium = df_labels[df_labels["wsi"] == wsi]["embedding medium"].values[0]
                #if medium == "FFPE":
                #    continue

                res_name = f"{slide_name}.pth"
                #res_pt = os.path.join(dir_res,medium, res_name)
                res_pt = os.path.join(dir_res, res_name)
                torch.save(sample, res_pt)

