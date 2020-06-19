import numpy as np
import pandas as pd

analysis_path = '/grid_fields/analysis/'
analysis_path = analysis_path + 'firing_rates_for_plots/'


first_path = analysis_path + 'first/'
second_path = analysis_path + 'second/'
all_path = analysis_path

spatial_first = pd.read_pickle(first_path + 'spatial_firing.pkl')
spatial_second = pd.read_pickle(second_path + 'spatial_firing.pkl')
spatial_all = pd.read_pickle(all_path + 'spatial_firing.pkl')

cell_first = spatial_first.iloc[0]
cell_second = spatial_second.iloc[0]
all = spatial_all.iloc[0]

print(' ')
print('fields in whole session')
for field in range(len(cell_first.firing_fields_hd_cluster)):
    hd_hist_cluster = all.firing_fields_hd_cluster[field]
    hd_hist_session = np.array(all.firing_fields_hd_session[field])
    hd_hist = hd_hist_cluster / hd_hist_session / 1000
    max_firing_rate = np.max(hd_hist[~np.isnan(hd_hist)].flatten())
    print(max_firing_rate)


print(' ')
print('fields in first half')
for field in range(len(cell_first.firing_fields_hd_cluster)):
    hd_hist_cluster = cell_first.firing_fields_hd_cluster[field]
    hd_hist_session = cell_first.firing_fields_hd_session[field] / 30000
    hd_hist = hd_hist_cluster / hd_hist_session / 1000
    max_firing_rate = np.max(hd_hist[~np.isnan(hd_hist)].flatten())
    print(max_firing_rate)


print('fields in second half')
for field in range(len(cell_second.firing_fields_hd_cluster)):
    hd_hist_cluster = cell_second.firing_fields_hd_cluster[field]
    hd_hist_session = cell_second.firing_fields_hd_session[field] / 30000
    hd_hist = hd_hist_cluster / hd_hist_session / 1000
    max_firing_rate = np.max(hd_hist[~np.isnan(hd_hist)].flatten())
    print(max_firing_rate)




