library(ggplot2)
library(tidyverse)
library(patchwork)
library(cowplot)

df_no_postprocessing <- read.csv("Data/HistoKit (no postprocessing)/binary_metrics_all.csv")
df_no_postprocessing$Method <- "No Postprocessing"
df_postprocessed <- read.csv("Data/HistoKit (postprocessed)/binary_metrics_all.csv")
df_postprocessed$Method <- "Postprocessed"
df <- rbind(df_no_postprocessing, df_postprocessed)
df<-df[df$Dataset=="SliDL",]
df<-df[df$TP != 0, ]


dodge <- position_dodge(width = 0.9)

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

## Dice
df_Dice <- df_long %>%
  filter(Metric == "Dice") %>%
  mutate(
    Method = factor(
      Method,
      levels = c(
        "No Postprocessing",
        "Postprocessed"
      )
    ),
    
    Mode_sigma = case_when(
      Mode == "Constant" ~ "Constant",
      Mode == "Gaussian" ~ paste0(
        "Gaussian\n\u03C3 = ",
        Sigma
      )
    )
  )

mode_sigma_levels <- df_Dice %>%
  distinct(Mode, Sigma, Mode_sigma) %>%
  arrange(
    factor(
      Mode,
      levels = c("Constant", "Gaussian")
    ),
    Sigma
  ) %>%
  pull(Mode_sigma)

df_Dice <- df_Dice %>%
  mutate(
    Mode_sigma = factor(
      Mode_sigma,
      levels = unique(mode_sigma_levels)
    )
  )

Dice_plot <- ggplot(
  df_Dice,
  aes(
    x = Mode_sigma,
    y = Value,
    fill = Method,
    group = interaction(Mode_sigma, Method)
  )
) +
  geom_violin(
    position = dodge,
    trim = TRUE,
    scale = "width",
    width = 0.85,
    alpha = 0.75,
    linewidth = 0.6,
    na.rm = TRUE
  ) +
  geom_boxplot(
    position = dodge,
    width = 0.2,
    fill = "white",
    color = "black",
    linewidth = 0.45,
    outlier.shape = NA,
    na.rm = TRUE
  ) +
  facet_wrap(
    vars(Overlap),
    nrow = 1
  ) +
  labs(
    title = "Dice",
    subtitle = "Overlap",
    x = "Mode",
    y = NULL,
    fill = "Method"
  ) +
  theme_bw() +
  theme(
    axis.ticks = element_blank(),
    strip.background = element_rect(
      fill = "grey95",
      color = "grey60"
    ),
    strip.text = element_text(
      face = "bold"
    ),
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    ),
    plot.subtitle = element_text(
      hjust = 0.5
    ),
    legend.position = "bottom"
  )

print(Dice_plot)

## FDR
df_FDR <- df_long %>%
  filter(Metric == "FDR") %>%
  mutate(
    Method = factor(
      Method,
      levels = c(
        "No Postprocessing",
        "Postprocessed"
      )
    ),
    
    Mode_sigma = case_when(
      Mode == "Constant" ~ "Constant",
      Mode == "Gaussian" ~ paste0(
        "Gaussian\n\u03C3 = ",
        Sigma
      )
    )
  )

mode_sigma_levels <- df_FDR %>%
  distinct(Mode, Sigma, Mode_sigma) %>%
  arrange(
    factor(
      Mode,
      levels = c("Constant", "Gaussian")
    ),
    Sigma
  ) %>%
  pull(Mode_sigma)

df_FDR <- df_FDR %>%
  mutate(
    Mode_sigma = factor(
      Mode_sigma,
      levels = unique(mode_sigma_levels)
    )
  )

FDR_plot <- ggplot(
  df_FDR,
  aes(
    x = Mode_sigma,
    y = Value,
    fill = Method,
    group = interaction(Mode_sigma, Method)
  )
) +
  geom_violin(
    position = dodge,
    trim = TRUE,
    scale = "width",
    width = 0.85,
    alpha = 0.75,
    linewidth = 0.6,
    na.rm = TRUE
  ) +
  geom_boxplot(
    position = dodge,
    width = 0.2,
    fill = "white",
    color = "black",
    linewidth = 0.45,
    outlier.shape = NA,
    na.rm = TRUE
  ) +
  facet_wrap(
    vars(Overlap),
    nrow = 1
  ) +
  labs(
    title = "FDR",
    subtitle = "Overlap",
    x = "Mode",
    y = NULL,
    fill = "Method"
  ) +
  theme_bw() +
  theme(
    axis.ticks = element_blank(),
    strip.background = element_rect(
      fill = "grey95",
      color = "grey60"
    ),
    strip.text = element_text(
      face = "bold"
    ),
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    ),
    plot.subtitle = element_text(
      hjust = 0.5
    ),
    legend.position = "bottom"
  )

print(FDR_plot)


## Sensitivity
df_Sensitivity <- df_long %>%
  filter(Metric == "Sensitivity") %>%
  mutate(
    Method = factor(
      Method,
      levels = c(
        "No Postprocessing",
        "Postprocessed"
      )
    ),
    
    Mode_sigma = case_when(
      Mode == "Constant" ~ "Constant",
      Mode == "Gaussian" ~ paste0(
        "Gaussian\n\u03C3 = ",
        Sigma
      )
    )
  )

mode_sigma_levels <- df_Sensitivity %>%
  distinct(Mode, Sigma, Mode_sigma) %>%
  arrange(
    factor(
      Mode,
      levels = c("Constant", "Gaussian")
    ),
    Sigma
  ) %>%
  pull(Mode_sigma)

df_Sensitivity <- df_Sensitivity %>%
  mutate(
    Mode_sigma = factor(
      Mode_sigma,
      levels = unique(mode_sigma_levels)
    )
  )

Sensitivity_plot <- ggplot(
  df_Sensitivity,
  aes(
    x = Mode_sigma,
    y = Value,
    fill = Method,
    group = interaction(Mode_sigma, Method)
  )
) +
  geom_violin(
    position = dodge,
    trim = TRUE,
    scale = "width",
    width = 0.85,
    alpha = 0.75,
    linewidth = 0.6,
    na.rm = TRUE
  ) +
  geom_boxplot(
    position = dodge,
    width = 0.2,
    fill = "white",
    color = "black",
    linewidth = 0.45,
    outlier.shape = NA,
    na.rm = TRUE
  ) +
  facet_wrap(
    vars(Overlap),
    nrow = 1
  ) +
  labs(
    title = "Sensitivity",
    subtitle = "Overlap",
    x = "Mode",
    y = NULL,
    fill = "Method"
  ) +
  theme_bw() +
  theme(
    axis.ticks = element_blank(),
    strip.background = element_rect(
      fill = "grey95",
      color = "grey60"
    ),
    strip.text = element_text(
      face = "bold"
    ),
    plot.title = element_text(
      hjust = 0.5,
      face = "bold"
    ),
    plot.subtitle = element_text(
      hjust = 0.5
    ),
    legend.position = "bottom"
  )

print(Sensitivity_plot)
