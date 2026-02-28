import os

import pandas as pd

cptac_org_ffpe = os.listdir("/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/CTranspath/original/CPTAC/FFPE/")
cptac_norm_ffpe = os.listdir("/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/CTranspath/vahadane_filtered_13_02_2025/CPTAC/FFPE/")

cptac_org_otc = os.listdir("/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/CTranspath/original/CPTAC/OCT/")
cptac_norm_otc = os.listdir("/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/CTranspath/vahadane_filtered_13_02_2025/CPTAC/OCT/")

tcga_org = os.listdir("/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/CTranspath/original/TCGA/")
tcga_norm = os.listdir("/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/CTranspath/vahadane_filtered_13_02_2025/TCGA/")

cracow_org = os.listdir("/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/CTranspath/original/Cracow/")
cracow_norm = os.listdir("/mnt/warehouse/jmerta/HistoKit/MIL_KAIED_2025/features_updated/tensors/CTranspath/vahadane_filtered_13_02_2025/Cracow/")


cracow_labels = pd.read_csv("/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/Cracow UCEC/labels/Cracow_UCEC_grade.csv")
tcga_labels = pd.read_csv("/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/TCGA UCEC/labels/TCGA_UCEC_grade.csv")
cptac_labels = pd.read_csv("/mnt/warehouse/Projects/HE/MIL UCEC/Datasets/CPTAC UCEC/labels/CPTAC_UCEC_grade.csv")


cracow_labels = cracow_labels[cracow_labels["wsi"].isin([f.split(".")[0] for f in cracow_norm])]
tcga_labels = tcga_labels[tcga_labels["wsi"].isin([f.split(".p")[0] for f in tcga_norm])]

cptac_norm = cptac_norm_otc+cptac_norm_ffpe
cptac_labels = cptac_labels[cptac_labels["wsi"].isin([f.split(".p")[0] for f in cptac_norm])]

summary = (
    cracow_labels
    .agg(
        n_patients=("patient", "nunique"),
        n_wsi=("wsi", "nunique")
    )
    .reset_index()
)
print("Cracow dataset summary:")
print(summary)

summary = (
    tcga_labels
    .agg(
        n_patients=("patient", "nunique"),
        n_wsi=("wsi", "nunique")
    )
    .reset_index()
)
print("TCGA dataset summary:")
print(summary)

summary = (
    cptac_labels
    .groupby(["grade", "embedding medium"])
    .agg(
        n_patients=("patient", "nunique"),
        n_wsi=("wsi", "nunique")
    )
    .reset_index()
)

print("CPTAC dataset summary:")
print(summary)

