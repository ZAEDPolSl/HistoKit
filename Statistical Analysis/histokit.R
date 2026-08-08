library(dplyr)
library(tidyr)
library(tibble)
library(purrr)
library(rstatix)
library(PMCMRplus)
library(effectsize)
library(ComplexHeatmap)
library(circlize)
library(stringr)
library(grid)

df <- read.csv("Data/HistoKit (postprocessed)/binary_metrics.csv")

metrics <- c("DICE",
             "FDR",
             "RECALL",
             "JACCARD",
             "PRECISION",
             "NPV",
             "F1",
             "SPECIFICITY")

df <- df %>%
  select(Mode, Overlap, Sigma, all_of(metrics), Image) %>%
  mutate(Group = if_else(
    Mode == "constant",
    paste0("Constant_Overlap", sprintf("%.2f", Overlap)),
    paste0("Gaussian_Overlap", sprintf("%.2f", Overlap), "_Std", Sigma)
  ))

group_order <- c(
  "Constant_Overlap0.25",
  "Gaussian_Overlap0.25_Std0.75",
  "Gaussian_Overlap0.25_Std1.25",
  "Gaussian_Overlap0.25_Std1.75",
  "Gaussian_Overlap0.25_Std2.25",
  
  "Constant_Overlap0.50",
  "Gaussian_Overlap0.50_Std0.75",
  "Gaussian_Overlap0.50_Std1.25",
  "Gaussian_Overlap0.50_Std1.75",
  "Gaussian_Overlap0.50_Std2.25",
  
  "Constant_Overlap0.75",
  "Gaussian_Overlap0.75_Std0.75",
  "Gaussian_Overlap0.75_Std1.25",
  "Gaussian_Overlap0.75_Std1.75",
  "Gaussian_Overlap0.75_Std2.25",
  
  "Constant_Overlap0.90",
  "Gaussian_Overlap0.90_Std0.75",
  "Gaussian_Overlap0.90_Std1.25",
  "Gaussian_Overlap0.90_Std1.75",
  "Gaussian_Overlap0.90_Std2.25"
)

df <- df %>% mutate(Group = factor(Group, levels = group_order))

prepare_metric_data <- function(data, metric, group_order) {
  metric_data <- data %>%
    transmute(
      Image = as.character(Image),
      Group = factor(Group, levels = group_order),
      Value = .data[[metric]]
    )
  
  # when recall is na, f1 is also na, that happens for one image without TP
  incomplete_images <- metric_data %>%
    group_by(Image) %>%
    summarise(
      incomplete = n_distinct(Group) != length(group_order) ||
        any(!is.finite(Value)),
      .groups = "drop"
    ) %>%
    filter(incomplete) %>%
    pull(Image)
  
  metric_data <- metric_data %>%
    filter(!Image %in% incomplete_images) %>%
    mutate(Group = droplevels(Group))
  
  list(data = metric_data, removed_images = incomplete_images)
}

# Friedman test
# Effect size: Kendall's W

friedman_results <- list()
prepared_data <- list()
removed_images <- list()

for (metric in metrics) {
  prepared <- prepare_metric_data(data = df,
                                  metric = metric,
                                  group_order = group_order)
  
  metric_data <- prepared$data
  
  prepared_data[[metric]] <- metric_data
  removed_images[[metric]] <- prepared$removed_images
  
  # friedman test
  friedman_result <- metric_data %>% friedman_test(Value ~ Group |
                                                     Image)
  
  # effect size
  effect_result <- metric_data %>% friedman_effsize(Value ~ Group |
                                                      Image)
  
  friedman_results[[metric]] <- friedman_result %>%
    mutate(
      Metric = metric,
      RemovedImages = length(prepared$removed_images),
      .before = 1
    ) %>%
    left_join(effect_result %>%
                select(effsize, magnitude), by = character())
}

friedman_results <- bind_rows(friedman_results)
friedman_results

# Print metrics with significant differences between groups

significant_metrics <- friedman_results %>% filter(p < 0.05) %>% pull(Metric)
significant_metrics

# Post-hoc conover-friedman
conover_results <- list()
metric_matrices <- list()

for (metric in significant_metrics) {
  metric_data <- prepared_data[[metric]]
  
  metric_matrix <- metric_data %>%
    mutate(Group = as.character(Group)) %>%
    pivot_wider(id_cols = Image,
                names_from = Group,
                values_from = Value) %>%
    arrange(Image) %>%
    column_to_rownames("Image") %>%
    as.matrix()
  
  storage.mode(metric_matrix) <- "double"
  
  metric_matrix <- metric_matrix[, group_order, drop = FALSE]
  metric_matrices[[metric]] <- metric_matrix
  
  # with Holm correction for multiple testing
  conover_results[[metric]] <-
    PMCMRplus::frdAllPairsConoverTest(y = metric_matrix, p.adjust.method = "holm")
}

## Rank - biserial correlation

calculate_effect_matrix <- function(metric_data, group_order) {
  effect_matrix <- matrix(
    NA_real_,
    nrow = length(group_order),
    ncol = length(group_order),
    dimnames = list(group_order, group_order)
  )
  
  for (i in seq_len(length(group_order) - 1)) {
    for (j in seq.int(i + 1, length(group_order))) {
      group1 <- group_order[i]
      group2 <- group_order[j]
      
      pair_data <- metric_data %>%
        filter(Group %in% c(group1, group2)) %>%
        mutate(Group = factor(Group, levels = c(group1, group2))) %>%
        pivot_wider(
          id_cols = Image,
          names_from = Group,
          values_from = Value
        ) %>%
        filter(is.finite(.data[[group1]]), is.finite(.data[[group2]]))
      
      effect_result <- effectsize::rank_biserial(
        x = pair_data[[group1]],
        y = pair_data[[group2]],
        paired = TRUE,
        ci = 0.95
      )
      
      effect_value <- as.data.frame(effect_result)$r_rank_biserial[1]
      
      effect_matrix[group1, group2] <- effect_value
      effect_matrix[group2, group1] <- -effect_value
    }
  }
  
  diag(effect_matrix) <- NA_real_
  effect_matrix
}

# filter for significant metrics
effect_matrices <- list()

for (metric in significant_metrics) {
  effect_matrices[[metric]] <- calculate_effect_matrix(metric_data = prepared_data[[metric]], group_order = group_order)
}


# heatmaps

overlap_colors <- c(
  "0.25" = "#AAF0A8",
  "0.50" = "#22BF1D",
  "0.75" = "#11660E",
  "0.90" = "#053604"
)

sigma_colors <- c(
  "Constant" = "gray40",
  "Gaussian std=0.75"     = "#C6DBEF",
  "Gaussian std=1.25"     = "#9ECAE1",
  "Gaussian std=1.75"     = "#4292C6",
  "Gaussian std=2.25"     = "#084594"
)


make_combined_heatmap <- function(conover_result,
                                  effect_matrix,
                                  metric,
                                  group_order,
                                  overlap_colors,
                                  sigma_colors)
{
  p_matrix_raw <- conover_result$p.value
  
  p_matrix_full <- matrix(
    NA_real_,
    nrow = length(group_order),
    ncol = length(group_order),
    dimnames = list(group_order, group_order)
  )
  
  for (row_name in rownames(p_matrix_raw)) {
    for (col_name in colnames(p_matrix_raw)) {
      p_value <- p_matrix_raw[row_name, col_name]
      
      if (!is.na(p_value) &&
          row_name %in% group_order && col_name %in% group_order) {
        p_matrix_full[row_name, col_name] <- p_value
        p_matrix_full[col_name, row_name] <- p_value
      }
    }
  }
  
  diag(p_matrix_full) <- NA_real_
  effect_matrix <- effect_matrix[group_order, group_order, drop = FALSE]
  diag(effect_matrix) <- NA_real_
  
  overlap_annotation <- str_extract(group_order, "(?<=Overlap)[0-9.]+")
  
  overlap_annotation <- factor(overlap_annotation, levels = c("0.25", "0.50", "0.75", "0.90"))
  
  names(overlap_annotation) <- group_order
  
  sigma_annotation <- str_extract(group_order, "(?<=Std)[0-9.]+")
  sigma_annotation[is.na(sigma_annotation)] <- "Constant"
  
  sigma_annotation <- factor(
    sigma_annotation,
    levels = c("Constant", "0.75", "1.25", "1.75", "2.25"),
    labels = c(
      "Constant",
      "Gaussian std=0.75",
      "Gaussian std=1.25",
      "Gaussian std=1.75",
      "Gaussian std=2.25"
    )
  )
  names(sigma_annotation) <- group_order
  
  # row and col annotation panels
  top_ha <- HeatmapAnnotation(
    Overlap = overlap_annotation,
    Sigma = sigma_annotation,
    col = list(Overlap = overlap_colors, Sigma = sigma_colors),
    
    simple_anno_size = unit(5, "mm"),
    show_annotation_name = FALSE,
    
    annotation_legend_param = list(
      Overlap = list(title = "Overlap", nrow = 1),
      Sigma = list(title = "Mode")
    )
  )
  
  left_ha <- rowAnnotation(
    Overlap = overlap_annotation,
    Sigma = sigma_annotation,
    
    col = list(Overlap = overlap_colors, Sigma = sigma_colors),
    
    simple_anno_size = unit(5, "mm"),
    show_annotation_name = FALSE,
    
    annotation_legend_param = list(
      Overlap = list(title = "Overlap", ncol = 1),
      Sigma = list(title = "Mode")
    )
  )
  
  
  # adjusted p-value
  p_col_fun <- colorRamp2(
    c(0, 0.001, 0.01, 0.05, 0.50, 1),
    c(
      "#9A133DFF",
      "#B93961FF",
      "#D8527CFF",
      "#F28AAAFF",
      "#F9B4C9FF",
      "#F9E0E8FF"
    )
  )
  
  
  # rank-biserial correlation
  effect_col_fun <- colorRamp2(
    c(-1, -0.5, 0, 0.5, 1),
    c(
      "#0055FFFF",
      "#66CCFFFF",
      "#ffffff",
      "#FFCC66FF",
      "#FF5500FF"
    )
  )
  
  
  
  base_matrix <- matrix(
    0,
    nrow = length(group_order),
    ncol = length(group_order),
    dimnames = list(group_order, group_order)
  )
  
  
  heatmap_object <- Heatmap(
    base_matrix,
    
    name = "Base",
    col = c("0" = "white"),
    
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    show_row_dend = FALSE,
    show_column_dend = FALSE,
    
    row_order = group_order,
    column_order = group_order,
    
    top_annotation = top_ha,
    left_annotation = left_ha,
    
    column_title = metric,
    column_title_gp = gpar(fontsize = 16, fontface = "bold"),
    
    show_row_names = FALSE,
    show_column_names = FALSE,
    
    rect_gp = gpar(type = "none"),
    
    show_heatmap_legend = FALSE,
    
    cell_fun = function(j, i, x, y, width, height, fill) {
      if (i == j) {
        grid.rect(
          x = x,
          y = y,
          width = width,
          height = height,
          gp = gpar(
            fill = "black",
            col = "white",
            lwd = 0.5
          )
        )
        
        return()
      }
      
      # adjusted p-value
      if (i > j) {
        p_value <- p_matrix_full[i, j]
        
        if (!is.na(p_value)) {
          grid.rect(
            x = x,
            y = y,
            width = width,
            height = height,
            gp = gpar(
              fill = p_col_fun(p_value),
              col = "white",
              lwd = 0.5
            )
          )
          
          significance <- case_when(p_value < 0.001 ~ "***",
                                    p_value < 0.01  ~ "**",
                                    p_value < 0.05  ~ "*",
                                    TRUE            ~ "")
          
          if (significance != "") {
            grid.text(
              significance,
              x = x,
              y = y,
              gp = gpar(fontsize = 12, fontface = "bold")
            )
          }
        }
      }
      
      
      # effect size
      if (i < j) {
        effect_value <- effect_matrix[i, j]
        
        if (!is.na(effect_value)) {
          grid.rect(
            x = x,
            y = y,
            width = width,
            height = height,
            gp = gpar(
              fill = effect_col_fun(effect_value),
              col = "white",
              lwd = 0.5
            )
          )
          
          text_color <- "black"
          
          
          grid.text(
            sprintf("%.2f", effect_value),
            x = x,
            y = y,
            gp = gpar(fontsize = 9, col = text_color)
          )
        }
      }
    }
  )
  
  
  p_legend <- Legend(
    title = "Adjusted p-value",
    col_fun = p_col_fun,
    
    at = c(0, 0.001, 0.01, 0.05, 0.50, 1),
    
    labels = c("0", "0.001", "0.01", "0.05", "0.50", "1"),
    
    legend_height = unit(7, "cm")
  )
  
  effect_legend <- Legend(
    title = "Rank-biserial\ncorrelation",
    
    col_fun = effect_col_fun,
    
    at = c(-1, -0.5, 0, 0.5, 1),
    
    labels = c("-1", "-0.5", "0", "0.5", "1.0"),
    
    legend_height = unit(7, "cm")
  )
  
  
  list(
    heatmap = heatmap_object,
    p_legend = p_legend,
    effect_legend = effect_legend,
    p_matrix = p_matrix_full,
    effect_matrix = effect_matrix
  )
}



combined_heatmaps <- list()

for (metric in significant_metrics) {
  combined_heatmaps[[metric]] <- make_combined_heatmap(
    conover_result = conover_results[[metric]],
    effect_matrix = effect_matrices[[metric]],
    metric = metric,
    group_order = group_order,
    overlap_colors = overlap_colors,
    sigma_colors = sigma_colors
  )
}


dir.create("Results/Heatmaps",
           recursive = TRUE,
           showWarnings = FALSE)

for (metric in names(combined_heatmaps)) {
  heatmap_result <- combined_heatmaps[[metric]]
  
  output_path <- file.path(
    "Results/Heatmaps",
    paste0(
      tolower(metric),
      "_pvalue_effectsize_heatmap_postprocessed.pdf"
    )
  )
  
  pdf(output_path, width = 11, height = 10)
  
  draw(
    heatmap_result$heatmap,
    
    merge_legends = TRUE,
    
    heatmap_legend_side = "right",
    annotation_legend_side = "right",
    
    heatmap_legend_list = list(heatmap_result$p_legend, heatmap_result$effect_legend)
  )
  
  dev.off()
  
  cat("Saved: ", output_path, "\n", sep = "")
}