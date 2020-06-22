library(knitr)
library(tidyverse)
library(scales)
library(glue)
library(ggstatsplot)

options(scipen=10000)

theme_set(theme_classic())

simulation_data_long <- read_csv('results_simulations_long.csv')

simulation_data_long_mt <- simulation_data_long %>% 
  group_by(name) %>% 
  mutate(frac_directional = sum(directional_correction) / n()) %>%
  ungroup()

simulation_data_long_mt$type[simulation_data_long_mt$name %in% c('Burgess', 'Giocomo', 'Guanella', 'Pastoll')] = 'Previous models'
simulation_data_long_mt$type[simulation_data_long_mt$name %in% c('Non-uniform conjunctive', '0.25 Non-uniform conjunctive', 'Uniform conjunctive', '0.25 Uniform conjunctive')] = 'New conjunctive cell input models'
simulation_data_long_mt$type[simulation_data_long_mt$name %in% c('Experimental data (mouse)', 'Experimental data (rat)')] = 'Experimental data'

simulation_data_long$type = factor(simulation_data_long_mt$type, levels = c('Previous models', 'New conjunctive cell input models', 'Experimental data'))

simulation_data_long_mt$name[simulation_data_long_mt$name == '0.25 Non-uniform conjunctive'] = 'Non-uniform (high gₑₓ)'
simulation_data_long_mt$name[simulation_data_long_mt$name == '0.25 Uniform conjunctive'] = 'Uniform (high gₑₓ)'
simulation_data_long_mt$name[simulation_data_long_mt$name == 'Uniform conjunctive'] = 'Uniform (low gₑₓ)'
simulation_data_long_mt$name[simulation_data_long_mt$name == 'Non-uniform conjunctive'] = 'Non-uniform (low gₑₓ)'
simulation_data_long_mt$name[simulation_data_long_mt$name == 'Experimental data (rat)'] = 'Rat'
simulation_data_long_mt$name[simulation_data_long_mt$name == 'Experimental data (mouse)'] = 'Mouse'


simulation_data_long_mt$name = factor(simulation_data_long_mt$name, levels=c(
  'Burgess',
  'Giocomo',
  'Guanella',
  'Pastoll',
  'Uniform (low gₑₓ)',
  'Non-uniform (low gₑₓ)',
  'Uniform (high gₑₓ)',
  'Non-uniform (high gₑₓ)',
  'Rat',
  'Mouse'
))



p1 <- simulation_data_long_mt %>% 
  filter(type %in% c('Previous models', 'Experimental data')) %>%
  ggplot(aes(name, number_of_different_bins_bh)) +
  geom_boxplot(aes(), outlier.shape = NA) +
  geom_jitter(alpha = 0.3, height = 0) + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  scale_fill_distiller(palette = 'Reds', direction = 0) +
  labs(x = '', y = 'Significant bins per field', fill = '') +
  scale_y_continuous(limits = c(0,20)) +
  facet_grid(~ type, scale = 'free_x', space = 'free') +
  theme(strip.background = element_blank(),
   strip.text.y = element_blank(), text = element_text(size = 24), strip.text.x = element_text(size = 13), plot.margin = margin(10, 10, 10, 15)) 

ggsave("6c.png", p1)

p2 <- simulation_data_long_mt %>% 
  filter(type %in% c('Previous models', 'Experimental data')) %>%
  group_by(name) %>%
  filter(row_number() == 1) %>%
  ggplot(aes(name, frac_directional)) +
  geom_bar(stat = 'identity') +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  scale_fill_distiller(palette = 'Reds', direction = 0) +
  labs(x = '', y = 'Proportion of directional fields', fill = '') +
  scale_y_continuous(limits = c(0,1)) +
  facet_grid(~ type, scale = 'free_x', space = 'free') +
  theme(strip.background = element_blank(),
   strip.text.y = element_blank(), text = element_text(size = 24), strip.text.x = element_text(size = 13), plot.margin = margin(11, 11, 11, 15)) 

ggsave("6d.png", p2)


p <- ggbetweenstats(
  data = simulation_data_long_mt  %>% filter(type %in% c('Previous models', 'Experimental data')),
  x = name,
  y = number_of_different_bins_bh,
  type = "parametric",
  k = 9, 
  outlier.tagging = FALSE,
  outlier.label.color = "darkgreen",
  pairwise.comparisons = TRUE,
) + theme(axis.text.x = element_text(angle = 45, hjust = 1))
pb <- ggplot_build(p)
pb$plot$plot_env$df_pairwise

write_csv(pb$plot$plot_env$df_pairwise, '6cd_comparisons.csv')

