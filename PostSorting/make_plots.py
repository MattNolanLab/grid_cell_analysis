import array_utility
import os
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
import plot_utility
import scipy.ndimage

from typing import Tuple


def plot_spike_histogram(spatial_firing, prm):
    sampling_rate = prm.get_sampling_rate()
    print('I will plot spikes vs time for the whole session excluding opto tagging.')
    save_path = prm.get_output_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        number_of_bins = int((spatial_firing.firing_times[cluster][-1] - spatial_firing.firing_times[cluster][0]) / (5 * sampling_rate))
        firings_cluster = spatial_firing.firing_times[cluster] / sampling_rate / 60
        spike_hist = plt.figure()
        spike_hist.set_size_inches(5, 5, forward=True)
        ax = spike_hist.add_subplot(1, 1, 1)
        spike_hist, ax = plot_utility.style_plot(ax)
        if number_of_bins > 0:
            hist, bins = np.histogram(firings_cluster, bins=number_of_bins)
            width = bins[1] - bins[0]
            center = (bins[:-1] + bins[1:]) / 2
            plt.bar(center, hist, align='center', width=width, color='black')
        plt.title('Spike histogram \n total spikes = ' + str(spatial_firing.number_of_spikes[cluster]) + ', \n mean fr = ' + str(round(spatial_firing.mean_firing_rate[cluster], 0)) + ' Hz', y=1.08, fontsize=24)
        plt.xlabel('Time (min)', fontsize=25)
        plt.ylabel('Number of spikes', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_histogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_histogram.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_firing_rate_vs_speed(spatial_firing, spatial_data,  prm):
    sampling_rate = 30
    print('I will plot spikes vs speed for the whole session excluding opto tagging.')
    save_path = prm.get_output_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    speed = spatial_data.speed[~np.isnan(spatial_data.speed)]
    number_of_bins = math.ceil(max(speed)) - math.floor(min(speed))
    session_hist, bins_s = np.histogram(speed, bins=number_of_bins, range=(math.floor(min(speed)), math.ceil(max(speed))))
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        speed_cluster = spatial_firing.speed[cluster]
        speed_cluster = sorted(speed_cluster)
        spike_hist = plt.figure()
        spike_hist.set_size_inches(5, 5, forward=True)
        ax = spike_hist.add_subplot(1, 1, 1)
        speed_hist, ax = plot_utility.style_plot(ax)
        if number_of_bins > 0:
            hist, bins = np.histogram(speed_cluster[1:], bins=number_of_bins, range=(math.floor(min(speed)), math.ceil(max(speed))))
            width = bins[1] - bins[0]
            center_bin = (bins[:-1] + bins[1:]) / 2
            center = center_bin[tuple([np.where(session_hist > sum(session_hist)*0.005)])]
            hist = np.array(hist, dtype=float)
            session_hist = np.array(session_hist, dtype=float)
            rate = np.divide(hist, session_hist, out=np.zeros_like(hist), where=session_hist != 0)
            rate = rate[tuple([np.where(session_hist[~np.isnan(session_hist)] > sum(session_hist)*0.005)])]
            plt.bar(center[0], rate[0]*sampling_rate, align='center', width=width, color='black')
        plt.xlabel('speed [cm/s]')
        plt.ylabel('firing rate [Hz]')
        plt.xlim(0, 30)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_speed_histogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_speed_histogram.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()


def calculate_autocorrelogram_hist(spikes, bin_size, window):
    half_window = int(window/2)
    number_of_bins = int(math.ceil(spikes[-1]*1000))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)-1):
        bin = math.floor(spikes[spike]*1000)
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = int(bins[b])
        window_start = int(bin - half_window)
        window_end = int(bin + half_window + 1)
        if (window_start > 0) and (window_end < len(train)):
            counts = counts + train[window_start:window_end]
            counted = counted + sum(train[window_start:window_end]) - train[bin]

    counts[half_window] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-half_window, half_window + 1, bin_size)
    return corr, time


def get_10ms_autocorr(firing_times_cluster, prm):
    corr1, time1 = calculate_autocorrelogram_hist(np.array(firing_times_cluster) / prm.get_sampling_rate(), 1, 20)
    return corr1, time1


def get_250ms_autocorr(firing_times_cluster, prm):
    corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster) / prm.get_sampling_rate(), 1, 500)
    return corr, time


def make_combined_autocorr_plot(time_10, corr_10, time_250, corr_250, spike_data, save_path, cluster):
    grid = plt.GridSpec(2, 1, hspace=0.5)
    autocorr_plot = plt.subplot(grid[0, 0])
    plt.suptitle("Autocorrelograms", fontsize=24)
    plt.xlabel('Time lag (ms)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.xlim(-10, 10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks([-10, 0, 10], [-10, 0, 10])
    plt.bar(time_10, corr_10, align='center', width=1, color='black')

    autocorr_plot2 = plt.subplot(grid[1, 0])
    plt.xlim(-250, 250)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time lag (ms)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.xticks([-250, 0, 250], [-250, 0, 250])
    plt.bar(time_250, corr_250, align='center', width=1, color='black')
    plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelograms.png',
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_autocorrelograms(spike_data: pd.DataFrame, prm: object) -> None:
    plt.close()
    print('I will plot autocorrelograms for each cluster (10 ms and 250 ms windows).')
    save_path = prm.get_output_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster = spike_data.cluster_id.values[cluster] - 1
        firing_times_cluster = spike_data.firing_times[cluster]
        corr_10, time_10 = get_10ms_autocorr(firing_times_cluster, prm)
        corr_250, time_250 = get_250ms_autocorr(firing_times_cluster, prm)
        make_combined_autocorr_plot(time_10, corr_10, time_250, corr_250, spike_data, save_path, cluster)


def plot_spikes_for_channel(grid, highest_value, lowest_value, spike_data, cluster, channel, snippet_column_name):
    snippet_plot = plt.subplot(grid[int(channel/2), channel % 2])
    plt.ylim(lowest_value - 10, highest_value + 30)
    plot_utility.style_plot(snippet_plot)
    snippet_plot.plot(spike_data[snippet_column_name][cluster][channel, :, :] * -1, color='lightslategray')
    snippet_plot.plot(np.mean(spike_data[snippet_column_name][cluster][channel, :, :], 1) * -1, color='red')
    plt.xticks([0, 10, 30], [-10, 0, 20])


def plot_spikes_for_channel_centered(grid, spike_data, cluster, channel, snippet_column_name):
    max_channel = spike_data.primary_channel[cluster]
    sd = np.std(spike_data.random_snippets[cluster][max_channel - 1, :, :] * -1)
    highest_value = np.median(spike_data.random_snippets[cluster][max_channel - 1, :, :] * -1) + (sd * 4)
    lowest_value = np.median(spike_data.random_snippets[cluster][max_channel - 1, :, :] * -1) - (sd * 4)
    snippet_plot = plt.subplot(grid[int(channel/2), channel % 2])
    plt.ylim(lowest_value - 10, highest_value + 30)
    plot_utility.style_plot(snippet_plot)
    snippet_plot.plot(spike_data[snippet_column_name][cluster][channel, :, :] * -1, color='lightslategray')
    snippet_plot.plot(np.mean(spike_data[snippet_column_name][cluster][channel, :, :], 1) * -1, color='red')
    plt.xticks([0, 30], [0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Voltage (µV)', fontsize=14)


def plot_waveforms(spike_data, prm):
    print('I will plot the waveform shapes for each cluster.')
    save_path = prm.get_output_path() + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spike_data)):
        cluster = spike_data.cluster_id.values[cluster] - 1
        fig = plt.figure(figsize=(5, 5))
        plt.suptitle("Spike waveforms", fontsize=24)
        grid = plt.GridSpec(2, 2, wspace=1, hspace=0.5)
        for channel in range(4):
            plot_spikes_for_channel_centered(grid, spike_data, cluster, channel, 'random_snippets')

        plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_waveforms_opto(spike_data, prm):
    if 'random_snippets_opto' in spike_data:
        print('I will plot the waveform shapes for each cluster for opto_tagging data.')
        save_path = prm.get_output_path() + '/Figures/opto_tagging'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            max_channel = spike_data.primary_channel[cluster]
            highest_value = np.max(spike_data.random_snippets_opto[cluster][max_channel-1, :, :] * -1)
            lowest_value = np.min(spike_data.random_snippets_opto[cluster][max_channel-1, :, :] * -1)
            fig = plt.figure(figsize=(5, 5))
            grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)
            for channel in range(4):
                plot_spikes_for_channel(grid, highest_value, lowest_value, spike_data, cluster, channel, 'random_snippets_opto')

            plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms_opto.png', dpi=300, bbox_inches='tight', pad_inches=0)
            # plt.savefig(save_path + '/' + spike_data.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms_opto.pdf', bbox_inches='tight', pad_inches=0)
            plt.close()


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


'''
Make scatter plot of speed vs firing rate and mark the median and the 25th and 75th percentiles.

position : data frame that contains the speed of the animal as a column ('speed')
spatial_firing : data frame that contains the firing times ('firing_times') and speed scores ('speed_score')
sigma : standard deviation for Gaussian filter (sigma = 250 / video_sampling)
sampling_rate_conversion : sampling rate of ephys data relative to seconds. If the firing times are in seconds then this
should be 1.
save_path : path to folder where the plot gets saved

'''


def plot_speed_vs_firing_rate(position: pd.DataFrame, spatial_firing: pd.DataFrame, sampling_rate_conversion: int, gauss_sd: float, prm: object) -> None:
    sampling_rate_video = int(1 / position['synced_time'].diff().mean())
    sigma = gauss_sd / sampling_rate_video

    speed = scipy.ndimage.filters.gaussian_filter(position.speed, sigma)
    save_path = prm.get_output_path() + '/Figures/firing_properties'
    for index, cell in spatial_firing.iterrows():
        firing_times = cell.firing_times
        firing_hist, edges = np.histogram(firing_times, bins=len(speed), range=(0, max(position.synced_time) * sampling_rate_conversion))
        firing_hist *= sampling_rate_video
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
        plt.title('Speed score: ' + str(np.round(cell.speed_score, 4)), fontsize=24)
        plt.xlim(0, 50)
        plt.ylim(0, None)
        plt.savefig(save_path + '/' + cell.session_id + '_' + str(cell.cluster_id) + '_speed_vs_firing_rate.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

