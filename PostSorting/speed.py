import array_utility
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import plot_utility
import scipy.ndimage
import scipy.stats

from typing import Tuple
import OverallAnalysis.analyze_speed


'''

The speed score is a measure of the correlation between the firing rate of the neuron and the running speed of the
animal. The firing times of the neuron are binned at the same sampling rate as the position data (speed). The resulting
temporal firing histogram is then smoothed with a Gaussian (standard deviation ~250ms). Speed and temporal firing rate
are correlated (Pearson correlation) to obtain the speed score.

Based on: Gois & Tort, 2018, Cell Reports 25, 1872–1884


position : data frame that contains the speed of the animal as a column ('speed').
spatial_firing : data frame that contains the firing times ('firing_times')
sigma : standard deviation for Gaussian filter (sigma = 250 / video_sampling)
sampling_rate_conversion : sampling rate of ephys data relative to seconds. If the firing times are in seconds then this
should be 1.

'''


def calculate_speed_score(position: pd.DataFrame, spatial_firing: pd.DataFrame, gauss_sd: float, sampling_rate_conversion: int) -> pd.DataFrame:
    avg_sampling_rate_video = float(1 / position['synced_time'].diff().mean())
    sigma = gauss_sd / avg_sampling_rate_video
    speed = scipy.ndimage.filters.gaussian_filter(position.speed, sigma)
    speed_scores = []
    speed_score_ps = []
    for index, cell in spatial_firing.iterrows():
        firing_times = cell.firing_times
        firing_hist, edges = np.histogram(firing_times, bins=len(speed), range=(0, max(position.synced_time) * sampling_rate_conversion))
        smooth_hist = scipy.ndimage.filters.gaussian_filter(firing_hist.astype(float), sigma)
        speed, smooth_hist = array_utility.remove_nans_from_both_arrays(speed, smooth_hist)
        speed_score, p = scipy.stats.pearsonr(speed, smooth_hist)
        speed_scores.append(speed_score)
        speed_score_ps.append(p)
    spatial_firing['speed_score'] = speed_scores
    spatial_firing['speed_score_p_values'] = speed_score_ps

    return spatial_firing


'''
Calculate median, 25th and 75th percentile of firing rate (y) at given speed (x) values. Speed is binned into 6 cm/s 
overlapping bins with a 2 cm/s step size.

Based on: Gois & Tort, 2018, Cell Reports 25, 1872–1884
'''


def calculate_median_for_scatter_binned(x: np.ndarray, y: np.ndarray) -> 'Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]':
    bin_size = 6
    step_size = 2
    number_of_bins = int((max(x) - min(x)) / 2)

    median_x = []
    median_y = []
    percentile_25 = []
    percentile_75 = []
    for x_bin in range(number_of_bins):
        median_x.append(x_bin * step_size + bin_size/2)
        data_in_bin = np.take(y, np.where((x_bin * step_size < x) & (x < x_bin * step_size + bin_size)))
        if len(data_in_bin[0]) > 0:
            med_y = np.median(data_in_bin)
            median_y.append(med_y)
            percentile_25.append(np.percentile(data_in_bin, 25))
            percentile_75.append(np.percentile(data_in_bin, 75))
        else:
            median_y.append(0)
            percentile_25.append(0)
            percentile_75.append(0)

    return np.array(median_x), np.array(median_y), np.array(percentile_25), np.array(percentile_75)


# plot grid cells only
def plot_speed_vs_firing_rate_grid(position: pd.DataFrame, spatial_firing: pd.DataFrame, sampling_rate_conversion: int, video_sampling_rate: int, save_path: str) -> None:
    sigma = 250 / video_sampling_rate
    speed = scipy.ndimage.filters.gaussian_filter(position.speed, sigma)
    spatial_firing = OverallAnalysis.analyze_speed.add_cell_types_to_data_frame(spatial_firing)
    for index, cell in spatial_firing.iterrows():
        if cell['cell type'] == 'grid':
            firing_times = cell.firing_times
            firing_hist, edges = np.histogram(firing_times, bins=len(speed), range=(0, max(position.synced_time) * sampling_rate_conversion))
            firing_hist *= video_sampling_rate
            smooth_hist = scipy.ndimage.filters.gaussian_filter(firing_hist.astype(float), sigma)
            speed, smooth_hist = array_utility.remove_nans_from_both_arrays(speed, smooth_hist)
            median_x, median_y, percentile_25, percentile_75 = calculate_median_for_scatter_binned(speed, smooth_hist)
            plt.cla()
            fig, ax = plt.subplots()
            ax = plot_utility.format_bar_chart(ax, 'Speed (cm/s)', 'Firing rate (Hz)')
            plt.scatter(speed[::10], smooth_hist[::10], color='gray', alpha=0.7)
            plt.plot(median_x, percentile_25, color='black', linewidth=5)
            plt.plot(median_x, percentile_75, color='black', linewidth=5)
            plt.scatter(median_x, median_y, color='black', s=100)
            plt.title('speed score: ' + str(np.round(cell.speed_score, 4)))
            plt.xlim(0, 50)
            plt.ylim(0, None)
            plt.savefig(save_path + cell.session_id + str(cell.cluster_id) + '_speed.png')
            plt.close()





