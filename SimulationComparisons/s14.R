library(tidyverse)
library(scales)
library(glue)
library(patchwork)

options(scipen=10000)

theme_set(theme_classic())


sampling_data <- read_csv('results_sampling.csv') %>%
  mutate(is_grid_cell = grid_score >= 0.4)
         
field_sampling_data <- read_csv('field_results_sampling.csv')
field_sampling_data["id"] = field_sampling_data$grid_id
field_sampling_data <- left_join(field_sampling_data, sampling_data, by = 'id')
  
sampling_data_shuffle_cell <- read_csv('results_sampling_shuffled_cell.csv')
sampling_data_correlation_cell <- read_csv('results_sampling_correlation_cell.csv')


field_sampling_data_mt <- field_sampling_data %>%
  filter(is_grid_cell == TRUE) %>%
  mutate(is_significant_corr = p_value < 0.01) 

p1 <- field_sampling_data_mt %>% 
  ggplot(aes(length_minutes, corr_within_R)) +
  annotate("rect", xmin=21.3, xmax=29.0, ymin=-1.00, ymax=Inf, fill=rgb(0, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  geom_hline(yintercept = 0, linetype = 'dotted', alpha = 0.4) +
  geom_boxplot(aes(group = length_minutes), outlier.shape = NA) +
  geom_jitter(aes(alpha = 0.25)) +
  labs(y = 'Within-field correlation', x = 'Recording length (minutes)') +
  scale_x_log10(breaks = c(1, 10, 100), limits = c(1, 650)) + 
  scale_y_continuous(limits = c(-1, 1)) +
  scale_colour_brewer(palette = 'Set1') +
  theme(legend.position = 'none') 

p1



field_corr_rec <- field_sampling_data_mt%>% filter(length_minutes >= 21.3, length_minutes <= 29)
  
print(median(field_corr_rec$corr_within_R))
print(sd(field_corr_rec$corr_within_R))

p9 <- 
  field_corr_rec %>% 
  ggplot(aes(corr_within_R)) + 
  annotate("rect", xmin=-1, xmax=1.0, ymin=0.00, ymax=1, fill=rgb(0, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  stat_ecdf(pad = FALSE) +
  labs(y = 'Cumulative probability', x = 'r') +
  geom_vline(xintercept = 0, colour="red") + 
  scale_x_continuous(breaks = c(-1, -0.5, 0, 0.5, 1), limits = c(-1, 1))+
  scale_y_continuous(breaks = c(0, 1), limits = c(0, 1), labels = c(0, 1))

p9

field_sampling_data_mt <- field_sampling_data %>%
  filter(is_grid_cell == TRUE) %>%
  mutate(is_significant_corr = factor(p_value < 0.01, labels = c("NS", "p < 0.01")))


summ <- field_sampling_data_mt %>% filter(field_number_of_spikes >= 183, field_number_of_spikes <= 1579, !is.na(corr_within_R))

median(summ$corr_within_R)
sd(summ$corr_within_R)


p2 <- field_sampling_data_mt %>%
  ggplot(aes(field_number_of_spikes, corr_within_R)) +
  labs(y = 'Within-field correlation', x = 'Number of spikes in field', colour = '') +
  annotate("rect", xmin=183, xmax=1579, ymin=-1.00, ymax=Inf, fill=rgb(0, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  geom_hline(yintercept = 0, linetype = 'dotted', alpha = 0.4) +
  #geom_point(aes(colour = is_significant_corr), alpha = 0.50) +
  geom_point(alpha = 0.50) +
  #theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  scale_x_log10(breaks = c(100, 1000, 10000)) + 
  scale_y_continuous(limits = c(-1, 1)) +
  scale_colour_brewer(palette = "Set1") +
  theme(legend.title = element_text(size=8, color = "salmon", face="bold"),
           legend.justification=c(1,0), 
           legend.position=c(1.00, 0.10),  
           legend.background = element_blank(),
           legend.key = element_blank()) 

p2 

field_sampling_data_mt <- field_sampling_data %>%
  filter(is_grid_cell == TRUE) %>%
  mutate(is_significant_corr = p_value < 0.01) 

p3 <- field_sampling_data_mt %>%
  ggplot(aes(field_sampling_points, corr_within_R)) +
  labs(y = 'R (within field)', x = 'Number of sampling events in field') +
  scale_x_log10() + 
  scale_colour_brewer(palette = "Set1") +
  theme(legend.position = 'none') 

p3

p4_alt <- field_sampling_data %>%
  filter(is_grid_cell == TRUE) %>%
  ggplot(aes(length_minutes, field_sig_bins)) +
  annotate("rect", xmin=0, xmax=Inf, ymin=4.29-4.9, ymax=4.29+4.9, fill=rgb(0, 0, 150, 0.2, maxColorValue = 255), alpha=0.122) +
  annotate("rect", xmin=21.3, xmax=29.0, ymin=-1.00, ymax=Inf, fill=rgb(150, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  geom_boxplot(aes(group = length_minutes), outlier.shape = NA) +
  geom_jitter(alpha = 0.2, height = 0) + 
  scale_x_log10(breaks = c(1, 10, 100), limits = c(1, 650)) + 
  labs(y = 'Number of significant bins', x = 'Recording length (minutes)') +
  scale_colour_brewer(palette = "Set1") 

p4_alt


p4 <- field_sampling_data %>%
  filter(is_grid_cell == TRUE) %>%
  ggplot(aes(length_minutes, field_sig_bins)) +
  annotate("rect", xmin=21.3, xmax=29.0, ymin=-1.00, ymax=Inf, fill=rgb(0, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  geom_boxplot(aes(group = length_minutes), outlier.shape = NA) +
  geom_jitter(alpha = 0.2, height = 0) + 
  scale_x_log10(breaks = c(1, 10, 100), limits = c(1, 650)) + 
  labs(y = 'Number of significant bins', x = 'Recording length (minutes)') +
  scale_colour_brewer(palette = "Set1") 

p4


summ <- field_sampling_data %>% filter(length_minutes <= 29.0, length_minutes >= 21.3, !is.na(field_sig_bins), is_grid_cell==TRUE)

mean(summ$field_sig_bins)
sd(summ$field_sig_bins)


field_sampling_data_mt <- field_sampling_data %>%
  filter(is_grid_cell == TRUE) 

p5 <- field_sampling_data_mt %>%
  ggplot(aes(field_number_of_spikes, field_sig_bins)) +
  annotate("rect", xmin=183, xmax=1579, ymin=-1.00, ymax=Inf, fill=rgb(0, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  geom_point(aes(alpha = 0.85)) +
  labs(y = 'Number of significant bins', x = 'Number of spikes in field') +
  #theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) +
  scale_x_log10(breaks = c(100, 1000, 10000)) + 
  scale_colour_brewer(palette = "Set1") +
  theme(legend.position = 'none') 
  #theme(axis.title.x = element_text(size = 10),
  #axis.title.y = element_text(size = 10))


p5


summ <- field_sampling_data_mt %>% filter(field_number_of_spikes >= 183, field_number_of_spikes <= 1579, !is.na(field_sig_bins), is_grid_cell==TRUE)

mean(summ$field_sig_bins)
sd(summ$field_sig_bins)




field_sampling_data_mt <- field_sampling_data %>%
  filter(is_grid_cell == TRUE)

p6 <- field_sampling_data_mt %>%
  ggplot(aes(field_sampling_points, field_sig_bins)) +
  geom_point(alpha = 0.85) +
  labs(y = 'Number of significant bins', x = 'Sampling events in field') +
  scale_colour_brewer(palette = "Set1") 

p6

p6 <- sampling_data_shuffle_cell %>%
  ggplot(aes(length_minutes, sig_bins)) +
  annotate("rect", xmin=21.3, xmax=29.0, ymin=-1.00, ymax=Inf, fill=rgb(0, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  geom_boxplot(aes(group = length_minutes), outlier.shape = NA) +
  geom_jitter(alpha = 0.2, height = 0) + 
  scale_x_log10(breaks = c(1, 10, 100), limits = c(1, 650)) + 
  labs(y = 'Number of significant bins', x = 'Recording length (minutes)') +
  scale_colour_brewer(palette = "Set1") 

p6
mean(sampling_data_shuffle_cell$sig_bins, na.rm=TRUE)
sd(sampling_data_shuffle_cell$sig_bins, na.rm=TRUE)

p7 <- sampling_data_correlation_cell %>%
  ggplot(aes(length_minutes, cell_hd_corr)) +
  annotate("rect", xmin=21.3, xmax=29.0, ymin=-1.00, ymax=Inf, fill=rgb(0, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  geom_boxplot(aes(group = length_minutes), outlier.shape = NA) +
  geom_jitter(alpha = 0.2, height = 0) + 
  scale_x_log10(breaks = c(1, 10, 100), limits = c(1, 650)) + 
  labs(y = 'Cell correlation', x = 'Recording length (minutes)') +
  geom_hline(yintercept = 0, linetype = 'dotted', alpha = 0.4) +
  scale_colour_brewer(palette = "Set1") 

p7


cell_corr_rec <- sampling_data_correlation_cell %>% filter(length_minutes >= 21.3, length_minutes <= 29)

print(median(cell_corr_rec$cell_hd_corr))
print(sd(cell_corr_rec$cell_hd_corr))
p8 <- 
  cell_corr_rec %>%
  ggplot(aes(cell_hd_corr)) + 
  annotate("rect", xmin=-1, xmax=1.0, ymin=0.00, ymax=1, fill=rgb(0, 0, 0, 0.2, maxColorValue = 255), alpha=0.122) +
  stat_ecdf(pad = FALSE) +
  labs(y = 'Cumulative probability', x = 'r') +
  geom_vline(xintercept = 0, colour="red") + 
  scale_x_continuous(breaks = c(-1, -0.5, 0, 0.5, 1), limits = c(-1, 1)) +
  scale_y_continuous(breaks = c(0, 1), limits = c(0, 1), labels = c(0, 1)) 

p8



panel_plot <- (p6 + p7) / (p4 + p5) / (p1 + p2) / (p8 + p9) + plot_annotation(tag_levels = 'a') & theme(plot.tag= element_text(size = 16))
panel_plot


ggsave('s14.png', panel_plot, width=10, height=13)
ggsave('s14.pdf', panel_plot, width=10, height=13)




