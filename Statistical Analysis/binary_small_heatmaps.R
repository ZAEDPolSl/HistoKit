library(ggplot2)
library(tidyverse)
library(patchwork)
library(cowplot)

## binary all

postprocessing <- c("no postprocessing", "postprocessed")
stats <- c("mean", "median")
datasets <- c("all", "SliDL", "GrandQC", "TCGA_CompassNMD")

for (postproc in postprocessing){
  for (stat in stats){
    for (ds in datasets){
      
      df <- read.csv(
        paste0(
          "Data/HistoKit (",
          postproc,
          ")/binary_metrics_all.csv"
        )
      )
      
      if (ds != "all")
      {
        df <- df[df$Dataset == ds, ]
      }
      
      df$Overlap <- factor(
        sprintf("%.2f", df$Overlap),
        levels = sprintf("%.2f", sort(unique(df$Overlap)))
      )
      
      metrics <- c("DICE", "RECALL", "FDR")
      
      df_long <- df %>%
        pivot_longer(
          cols = all_of(metrics),
          names_to = "Metric",
          values_to = "Value"
        )
      
      if (stat == "median"){
        
        heatmap_data <- df_long %>%
          group_by(Overlap, Sigma, Mode, Metric) %>%
          summarise(
            MeanValue = median(Value, na.rm = TRUE),
            .groups = "drop"
          ) %>%
          mutate(
            Sigma_label = case_when(
              Mode == "constant" ~ "",
              TRUE ~ as.character(Sigma)
            )
          )
      } else {
        heatmap_data <- df_long %>%
          group_by(Overlap, Sigma, Mode, Metric) %>%
          summarise(
            MeanValue = mean(Value, na.rm = TRUE),
            .groups = "drop"
          ) %>%
          mutate(
            Sigma_label = case_when(
              Mode == "constant" ~ "",
              TRUE ~ as.character(Sigma)
            )
          )
      }
      
      heatmap_data <- heatmap_data %>%
        mutate(
          Metric = recode(
            Metric,
            ACCURACY = "Accuracy",
            JACCARD = "Jaccard",
            RECALL = "Sensitivity",
            PRECISION = "Precision",
            SPECIFICITY = "Specificity",
            DICE = "Dice",
            F1 = "F1",
            FDR = "FDR",
            NPV = "NPV"
          )
        )
      
      heatmap_data[heatmap_data$Mode == "gaussian", ]$Mode <- "Gaussian"
      heatmap_data[heatmap_data$Mode == "constant", ]$Mode <- "Constant"
      
      plots <- heatmap_data %>%
        group_by(Metric) %>%
        mutate(
          is_max = MeanValue == max(MeanValue, na.rm = TRUE),
          is_min = MeanValue == min(MeanValue, na.rm = TRUE)
        ) %>%
        ungroup() %>%
        split(.$Metric) %>%
        map(
          ~ ggplot(
            .x,
            aes(
              x = factor(Overlap),
              y = factor(Sigma_label),
              fill = MeanValue
            )
          ) +
            geom_tile(color = "white") +
            
            geom_text(
              aes(
                label = sprintf("%.4f", MeanValue),
                color = MeanValue > (
                  min(MeanValue, na.rm = TRUE) +
                    max(MeanValue, na.rm = TRUE)
                ) / 2
              ),
              size = 3.5,
              fontface = "bold"
            ) +
            scale_color_manual(
              values = c(
                "TRUE" = "black",
                "FALSE" = "white"
              ),
              guide = "none"
            ) +
            
            geom_tile(
              data = filter(.x, is_max),
              fill = NA,
              color = "red",
              linewidth = 1.5
            ) +
            geom_tile(
              data = filter(.x, is_min),
              fill = NA,
              color = "#149ed9",
              linewidth = 1.5
            ) +
            
            scale_fill_continuous(
              labels = function(x) sprintf("%.4f", x),
              low = "#064003",
              high = "#d2f7e2"
            ) +
            facet_grid(
              cols = vars(Mode),
              scales = "free_x",
              space = "free_x"
            ) +
            theme_bw() +
            coord_flip() +
            labs(
              y = "                 std",
              fill = stat,
              title = unique(.x$Metric),
              x = NULL
            ) +
            theme(
              axis.ticks = element_blank()
            )
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
              b = 20,
              l = 35
            )
          )
        )
      
      main_plot <- ggdraw(combined_plot) +
        draw_label(
          "Mode",
          x = 0.5,
          y = 0.005,
          hjust = 0.5,
          vjust = 0,
          size = 15
        ) +
        draw_label(
          "Overlap",
          x = 0.02,
          y = 0.5,
          hjust = 0.5,
          vjust = 0,
          size = 15,
          angle = 90
        )
      
      legend_plot <- ggplot(
        data.frame(
          x = 1,
          y = 1,
          type = factor(
            c("MIN", "MAX"),
            levels = c("MIN", "MAX")
          )
        ),
        aes(x = x, y = y, color = type)
      ) +
        geom_tile(
          fill = NA,
          linewidth = 1.5
        ) +
        scale_color_manual(
          values = c(
            "MIN" = "#149ed9",
            "MAX" = "red"
          ),
          labels = c(
            "MIN" = "Min",
            "MAX" = "Max"
          ),
          name = NULL
        ) +
        guides(
          color = guide_legend(
            override.aes = list(
              fill = NA,
              linewidth = 1.5
            ),
            ncol = 1
          )
        ) +
        theme_void() +
        theme(
          legend.position = "right",
          legend.direction = "vertical",
          legend.text = element_text(size = 10),
          legend.key.size = grid::unit(0.5, "cm")
        )
      
      outline_legend <- cowplot::get_legend(legend_plot)
      
      final_plot <- cowplot::plot_grid(
        main_plot,
        outline_legend,
        ncol = 2,
        rel_widths = c(1, 0.07),
        align = "h"
      )
      
      print(final_plot)
      
      ggsave(
        filename = paste0(
          "Data/HistoKit (",
          postproc,
          ")/heatmaps_metrics/",
          stat,
          "_",
          ds,
          "small.pdf"
        ),
        plot = final_plot,
        width = 14,
        height = 3.5,
        units = "in",
        device = cairo_pdf
      )
      
    }
  }
}