getwd()

# =========================
# GrandQC - Breast
# =========================
df_breast_binary <- read.csv("Data/HistoKit (postprocessed)/GrandQC_Breast_binary_metrics.csv")
df_breast_binary$Organ <- "Breast"
df_breast_binary$Dataset <- "GrandQC"

df_breast_multiclass <- read.csv("Data/HistoKit (postprocessed)/GrandQC_Breast_multiclass_metrics.csv")
df_breast_multiclass$Organ <- "Breast"
df_breast_multiclass$Dataset <- "GrandQC"

# =========================
# GrandQC - Colon
# =========================
df_colon_binary <- read.csv("Data/HistoKit (postprocessed)/GrandQC_Colon_binary_metrics.csv")
df_colon_binary$Organ <- "Colon"
df_colon_binary$Dataset <- "GrandQC"

df_colon_multiclass <- read.csv("Data/HistoKit (postprocessed)/GrandQC_Colon_multiclass_metrics.csv")
df_colon_multiclass$Organ <- "Colon"
df_colon_multiclass$Dataset <- "GrandQC"

# =========================
# GrandQC - Kidney
# =========================
df_kidney_binary <- read.csv("Data/HistoKit (postprocessed)/GrandQC_Kidney_binary_metrics.csv")
df_kidney_binary$Organ <- "Kidney"
df_kidney_binary$Dataset <- "GrandQC"

df_kidney_multiclass <- read.csv("Data/HistoKit (postprocessed)/GrandQC_Kidney_multiclass_metrics.csv")
df_kidney_multiclass$Organ <- "Kidney"
df_kidney_multiclass$Dataset <- "GrandQC"

# =========================
# GrandQC - Prostate
# =========================
df_prostate_binary <- read.csv("Data/HistoKit (postprocessed)/GrandQC_Prostate_binary_metrics.csv")
df_prostate_binary$Organ <- "Prostate"
df_prostate_binary$Dataset <- "GrandQC"

df_prostate_multiclass <- read.csv("Data/HistoKit (postprocessed)/GrandQC_Prostate_multiclass_metrics.csv")
df_prostate_multiclass$Organ <- "Prostate"
df_prostate_multiclass$Dataset <- "GrandQC"

# =========================
# SliDL
# =========================
df_slidl_binary <- read.csv("Data/HistoKit (postprocessed)/SliDL_binary_metrics.csv")
df_slidl_binary$Organ <- "Multiple"
df_slidl_binary$Dataset <- "SliDL"

df_slidl_multiclass <- read.csv("Data/HistoKit (postprocessed)/SliDL_multiclass_metrics.csv")
df_slidl_multiclass$Organ <- "Multiple"
df_slidl_multiclass$Dataset <- "SliDL"

# =========================
# TCGA CompassNMD
# =========================
df_tcga_compass_binary <- read.csv("Data/HistoKit (postprocessed)/TCGA_CompassNMD_binary_metrics.csv")
df_tcga_compass_binary$Organ <- "Multiple"
df_tcga_compass_binary$Dataset <- "TCGA_CompassNMD"

df_tcga_compass_multiclass <- read.csv("Data/HistoKit (postprocessed)/TCGA_CompassNMD_multiclass_metrics.csv")
df_tcga_compass_multiclass$Organ <- "Multiple"
df_tcga_compass_multiclass$Dataset <- "TCGA_CompassNMD"


df_binary <- rbind(
  df_breast_binary,
  df_colon_binary,
  df_kidney_binary,
  df_prostate_binary,
  df_slidl_binary,
  df_tcga_compass_binary
)

write.csv(df_binary, "Data/HistoKit (postprocessed)/binary_metrics_all.csv", row.names = FALSE)

df_multiclass <- rbind(
  df_breast_multiclass,
  df_colon_multiclass,
  df_kidney_multiclass,
  df_prostate_multiclass,
  df_tcga_compass_multiclass
)
write.csv(df_binary, "Data/HistoKit (postprocessed)/multiclass_metrics_all.csv", row.names = FALSE)
