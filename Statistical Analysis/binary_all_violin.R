library(ggplot2)
library(tidyverse)
library(patchwork)
library(cowplot)

df <- read.csv("Data/HistoKit (postprocessed)/binary_metrics_all.csv")


metrics <- c(
  "DICE", "JACCARD", "PRECISION", "RECALL", "ACCURACY",
  "F1", "FDR", "NPV", "SPECIFICITY"
)

df_long <- df %>%
  pivot_longer(
    cols = all_of(metrics),
    names_to = "Metric",
    values_to = "Value"
  ) %>%
  mutate(
    Overlap = factor(
      sprintf("%.2f", Overlap),
      levels = sprintf(
        "%.2f",
        sort(unique(df$Overlap))
      )
    ),
    
    Metric = recode(
      Metric,
      RECALL = "Sensitivity",
      DICE   = "Dice",
      FDR    = "FDR"
    ),
    
    Mode = recode(
      Mode,
      gaussian = "Gaussian",
      constant = "Constant"
    ),
    
    Sigma_label = case_when(
      Mode == "Constant" ~ "",
      Mode == "Gaussian" ~ paste0("\u03C3 = ", Sigma)
    )
  )

reference_line <- df_long %>%
  group_by(
    Metric,
    Mode,
    Sigma_label,
    Overlap
  ) %>%
  summarise(
    MeanValue = mean(Value, na.rm = TRUE),
    MedianValue = median(Value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(
    Metric,
    Mode,
    Sigma_label
  ) %>%
  summarise(
    BestMean = case_when(
      first(Metric) %in% c("Dice", "Sensitivity") ~
        max(MeanValue, na.rm = TRUE),
      
      first(Metric) == "FDR" ~
        min(MeanValue, na.rm = TRUE)
    ),
    
    BestMedian = case_when(
      first(Metric) %in% c("Dice", "Sensitivity") ~
        max(MedianValue, na.rm = TRUE),
      
      first(Metric) == "FDR" ~
        min(MedianValue, na.rm = TRUE)
    ),
    
    .groups = "drop"
  )

plots <- df_long %>%
  split(.$Metric) %>%
  imap(
    ~ {
      current_reference <- reference_line %>%
        filter(Metric == .y)
      
      ggplot(
        .x,
        aes(
          x = Overlap,
          y = Value,
          fill = Overlap
        )
      ) +
        geom_violin(
          trim = TRUE,
          scale = "width",
          na.rm = TRUE
        ) +
        geom_boxplot(
          width = 0.3,
          fill = "white",
          color = "black",
          linewidth = 0.5,
          outlier.shape = NA,
          na.rm = TRUE
        ) +
        facet_grid(
          cols = vars(Mode, Sigma_label),
          scales = "free_x",
          space = "free_x"
        ) +
        scale_linetype_manual(
          name = NULL,
          values = c(
            Mean = "solid",
            Median = "dashed"
          )
        ) +
        labs(
          title = .y,
          x = "Overlap",
          y = NULL
        ) +
        guides(fill = "none") +
        theme_bw()
    }
  )

combined_plot <- wrap_plots(
  plots,
  ncol = 3
) +
  plot_annotation(
    theme = theme(
      plot.margin = margin(
        t = 5,
        r = 5,
        b = 10,
        l = 5
      )
    )
  )

print(combined_plot)