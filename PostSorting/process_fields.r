#! /usr/bin/Rscript
args = commandArgs(trailingOnly=TRUE)
.libPaths("/home/nolanlab/R/x86_64-pc-linux-gnu-library/3.4")
library(tidyverse)
library(circular)


setwd(args[1])

# load data
read_plus <- function(flnm) {
  read_csv(flnm, col_names = FALSE) %>%
    mutate(filename = flnm)
}

c <-list.files(pattern = "*cluster.csv",
               full.names = T) %>%
  map_dfr(~read_plus(.)) %>%
  group_by(filename) %>%
  nest(filename, X1, .key = "c")


s <-list.files(pattern = "*session.csv",
               full.names = T) %>%
  map_dfr(~read_plus(.)) %>%
  group_by(filename) %>%
  nest(filename, X1, .key = "s")

cs <- bind_cols(c, s)

# Test for uniformity of data using Kuiper’s test. First performs test on all cluster data then on all session data. Makes new columns for test statistic and p value.

cs <- cs %>%
  mutate(c_circ = map(c,circular)) %>%
  mutate(c_kuiper = map(c_circ, kuiper.test)) %>%
  mutate(c_kuiper_ts = map_dbl(c_kuiper, ~.$statistic)) %>%
  mutate(s_circ = map(s,circular)) %>%
  mutate(s_kuiper = map(s_circ, kuiper.test)) %>%
  mutate(s_kuiper_ts = map_dbl(s_kuiper, ~.$statistic))

# Test for uniformity of data using Watson’s test

cs <- cs %>%
  mutate(w_c = map(c_circ, watson.test)) %>%
  mutate(w_c_ts = map_dbl(w_c, ~.$statistic)) %>%
  mutate(w_s = map(s_circ, watson.test)) %>%
  mutate(w_s_ts = map_dbl(w_s, ~.$statistic))

# Compare the two distributions using Watson’s two sample test.

cs <- cs %>%
  mutate(w2st = map2(c_circ, s_circ, watson.two.test)) %>%
  mutate(w2st_ts = map_dbl(w2st, ~.$statistic))

# Make table with test statistics

table <- tibble(cluster = cs$filename, session = cs$filename1, Kuiper_Cluster = cs$c_kuiper_ts, Kuiper_Session = cs$s_kuiper_ts, Watson_Cluster = cs$w_c_ts, Watson_Session = cs$w_s_ts, Watson_two_sample = cs$w2st_ts)

knitr::kable(table)

# save table as csv
write_csv(table, "circular_out.csv")

# close open files
closeAllConnections()
