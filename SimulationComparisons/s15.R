library(tidyverse)
library(scales)
library(glue)
library(patchwork)
library(ggstatsplot)
library(ggpubr)

options(scipen=10000)

theme_set(theme_classic())


spacing_data <- read_csv('results_gridspacing.csv') %>%
  mutate(is_grid_cell = grid_score >= 0.4)


size_data <- read_csv('results_gridsize.csv') %>%
  mutate(is_grid_cell = grid_score >= 0.4)
         
size_data_mt <- size_data %>% 
  mutate(frac_sig_bins = sig_bins / total_bins) %>%
  filter(is_grid_cell == TRUE, !is.na(frac_sig_bins))
  
size_plot <- ggscatter(size_data_mt,
          x = 'field_size',
          y = 'frac_sig_bins',
          add = 'reg.line',
          conf.int = TRUE,
          cor.coef = TRUE,
          cor.method = 'pearson',
          xlab = expression('Field size cm'^{'2'}),
          ylab = 'Proportion of directional bins',
          caption = ''
          )

size_plot



spacing_data_mt <- spacing_data %>% 
  mutate(frac_sig_bins = sig_bins / total_bins) %>%
  filter(is_grid_cell == TRUE, !is.na(frac_sig_bins))
  
spacing_plot <- ggscatter(spacing_data_mt,
          x = 'calculated_grid_spacing',
          y = 'frac_sig_bins',
          add = 'reg.line',
          conf.int = TRUE,
          cor.coef = TRUE,
          cor.method = 'pearson',
          xlab = 'Field spacing (cm)',
          ylab = 'Proportion of directional bins',
          caption = ''
          )
spacing_plot


panel_plot <- size_plot + spacing_plot + plot_layout(ncol = 1) + plot_annotation(tag_levels = 'a') & theme(plot.tag= element_text(size = 16))

ggsave('s15.png', panel_plot)
ggsave('s15.pdf', panel_plot)




