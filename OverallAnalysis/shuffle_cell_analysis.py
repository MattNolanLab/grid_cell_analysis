import glob
import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.false_positives
import OverallAnalysis.folder_path_settings
import pandas as pd
import plot_utility
import PostSorting.open_field_grid_cells
import scipy
from scipy import stats
import shutil
from statsmodels.sandbox.stats.multicomp import multipletests
import PostSorting.open_field_firing_maps
import PostSorting.parameters
import array_utility

local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/shuffled_analysis_cell/'
local_path_mouse = local_path + 'all_mice_df.pkl'
local_path_mouse_down_sampled = local_path + 'all_mice_df_down_sampled.pkl'
local_path_rat = local_path + 'all_rats_df.pkl'

server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()

prm = PostSorting.parameters.Parameters()


def format_bar_chart(ax):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head direction (deg)', fontsize=30)
    ax.set_ylabel('Frequency (Hz)', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    return ax


def load_data_frame_spatial_firing(output_path, server_path, spike_sorter='/MountainSort', df_path='/DataFrames'):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        firing_data_frame_path = recording_folder + spike_sorter + df_path + '/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + df_path + '/position.pkl'
        if os.path.exists(firing_data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(firing_data_frame_path)
            position = pd.read_pickle(position_path)

            spatial_firing_to_combine = pd.DataFrame()
            if 'position_x' in spatial_firing:
                if 'number_of_spikes_in_fields' in spatial_firing:
                    spatial_firing_to_combine['number_of_spikes_in_fields'] = spatial_firing.number_of_spikes_in_fields
                else:
                    spatial_firing_to_combine['number_of_spikes_in_fields'] = 0

                spatial_firing_to_combine = spatial_firing[['session_id', 'cluster_id', 'hd_score', 'position_x', 'position_y', 'hd', 'firing_maps', 'number_of_spikes_in_fields', 'firing_times']].copy()
                spatial_firing_to_combine['trajectory_hd'] = [position.hd] * len(spatial_firing)
                spatial_firing_to_combine['trajectory_x'] = [position.position_x] * len(spatial_firing)
                spatial_firing_to_combine['trajectory_y'] = [position.position_y] * len(spatial_firing)
                spatial_firing_to_combine['trajectory_times'] = [position.synced_time] * len(spatial_firing)

                number_spikes = []
                for index, cell in spatial_firing_to_combine.iterrows():
                    num_spikes = len(cell.position_x)
                    number_spikes.append(num_spikes)
                spatial_firing_to_combine['number_of_spikes'] = number_spikes
                spatial_firing_to_combine = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing_to_combine)

                spatial_firing_data = spatial_firing_data.append(spatial_firing_to_combine)
                print(spatial_firing_data.head())

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def add_combined_id_to_df(spatial_firing):
    animal_ids = [session_id.split('_')[0] for session_id in spatial_firing.session_id.values]
    dates = [session_id.split('_')[1] for session_id in spatial_firing.session_id.values]
    if 'tetrode' in spatial_firing:
        tetrode = spatial_firing.tetrode.values
        cluster = spatial_firing.cluster_id.values

        combined_ids = []
        for cell in range(len(spatial_firing)):
            id = animal_ids[cell] + '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
            combined_ids.append(id)
        spatial_firing['false_positive_id'] = combined_ids
    else:
        cluster = spatial_firing.cluster_id.values
        combined_ids = []
        for cell in range(len(spatial_firing)):
            id = animal_ids[cell] + '-' + dates[cell] + '-Cluster-' + str(cluster[cell])
            combined_ids.append(id)
        spatial_firing['false_positive_id'] = combined_ids

    return spatial_firing


def tag_false_positives(spatial_firing):
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(local_path + 'false_positives_all.txt')
    spatial_firing = add_combined_id_to_df(spatial_firing)
    spatial_firing['false_positive'] = spatial_firing['false_positive_id'].isin(list_of_false_positives)
    return spatial_firing


def add_mean_and_std_to_df(spatial_firing, sampling_rate_video, number_of_bins=20):
    shuffled_means = []
    shuffled_stdevs = []
    real_data_hz_all = []
    time_spent_in_bins_all = []
    histograms_hz_all_shuffled = []
    for index, cell in spatial_firing.iterrows():
        shuffled_histograms = cell['shuffled_data']
        cell_spikes_hd = np.asanyarray(cell['hd'])
        cell_spikes_hd = cell_spikes_hd[~np.isnan(cell_spikes_hd)]  # real hd when the cell fired
        cell_session_hd = np.asanyarray(cell['trajectory_hd'])  # hd from the whole session in field
        cell_session_hd = cell_session_hd[~np.isnan(cell_session_hd)]
        time_spent_in_bins = np.histogram(cell_session_hd, bins=number_of_bins)[0]
        time_spent_in_bins_all.append(time_spent_in_bins)
        shuffled_histograms_hz = shuffled_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        histograms_hz_all_shuffled.append(shuffled_histograms_hz)
        mean_shuffled = np.mean(shuffled_histograms_hz, axis=0)
        shuffled_means.append(mean_shuffled)
        std_shuffled = np.std(shuffled_histograms_hz, axis=0)
        shuffled_stdevs.append(std_shuffled)
        real_data_hz = np.histogram(cell_spikes_hd, bins=number_of_bins)[0] * sampling_rate_video / time_spent_in_bins
        real_data_hz_all.append(real_data_hz)
    spatial_firing['shuffled_means'] = shuffled_means
    spatial_firing['shuffled_std'] = shuffled_stdevs
    spatial_firing['hd_histogram_real_data_hz'] = real_data_hz_all
    spatial_firing['time_spent_in_bins'] = time_spent_in_bins_all
    spatial_firing['shuffled_histograms_hz'] = histograms_hz_all_shuffled
    return spatial_firing


def add_percentile_values_to_df(spatial_firing, sampling_rate_video, number_of_bins=20):
    percentile_values_95_all = []
    percentile_values_5_all = []
    error_bar_up_all = []
    error_bar_down_all = []
    for index, cell in spatial_firing.iterrows():
        shuffled_cell_histograms = cell['shuffled_data']
        time_spent_in_bins = cell.time_spent_in_bins  # based on trajectory
        shuffled_histograms_hz = shuffled_cell_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        percentile_value_shuffled_95 = np.percentile(shuffled_histograms_hz, 95, axis=0)
        percentile_values_95_all.append(percentile_value_shuffled_95)
        percentile_value_shuffled_5 = np.percentile(shuffled_histograms_hz, 5, axis=0)
        percentile_values_5_all.append(percentile_value_shuffled_5)
        error_bar_up = percentile_value_shuffled_95 - cell.shuffled_means
        error_bar_down = cell.shuffled_means - percentile_value_shuffled_5
        error_bar_up_all.append(error_bar_up)
        error_bar_down_all.append(error_bar_down)
    spatial_firing['shuffled_percentile_threshold_95'] = percentile_values_95_all
    spatial_firing['shuffled_percentile_threshold_5'] = percentile_values_5_all
    spatial_firing['error_bar_95'] = error_bar_up_all
    spatial_firing['error_bar_5'] = error_bar_down_all
    return spatial_firing


# test whether real and shuffled data differ and add results (true/false for each bin) and number of diffs to data frame
def test_if_real_hd_differs_from_shuffled(spatial_firing):
    real_and_shuffled_data_differ_bin = []
    number_of_diff_bins = []
    for index, cell in spatial_firing.iterrows():
        # diff_field = np.abs(field.shuffled_means - field.hd_histogram_real_data) > field.shuffled_std * 2
        diff_cell = (cell.shuffled_percentile_threshold_95 < cell.hd_histogram_real_data_hz) + (cell.shuffled_percentile_threshold_5 > cell.hd_histogram_real_data_hz)  # this is a pairwise OR on the binary arrays
        number_of_diffs = diff_cell.sum()
        real_and_shuffled_data_differ_bin.append(diff_cell)
        number_of_diff_bins.append(number_of_diffs)
    spatial_firing['real_and_shuffled_data_differ_bin'] = real_and_shuffled_data_differ_bin
    spatial_firing['number_of_different_bins'] = number_of_diff_bins
    return spatial_firing


# this uses the p values that are based on the position of the real data relative to shuffled (corrected_
def count_number_of_significantly_different_bars_per_field(spatial_firing, significance_level=95, type='bh'):
    number_of_significant_p_values = []
    false_positive_ratio = (100 - significance_level) / 100
    for index, cell in spatial_firing.iterrows():
        # count significant p values
        if type == 'bh':
            number_of_significant_p_values_cell = (cell.p_values_corrected_bars_bh < false_positive_ratio).sum()
            number_of_significant_p_values.append(number_of_significant_p_values_cell)
        if type == 'holm':
            number_of_significant_p_values_cell = (cell.p_values_corrected_bars_holm < false_positive_ratio).sum()
            number_of_significant_p_values.append(number_of_significant_p_values_cell)
    field_name = 'number_of_different_bins_' + type
    spatial_firing[field_name] = number_of_significant_p_values
    return spatial_firing


# this is to find the null distribution of number of rejected null hypothesis based on the shuffled data
def test_if_shuffle_differs_from_other_shuffles(spatial_firing):
    number_of_shuffles = len(spatial_firing.shuffled_data.iloc[0])
    rejected_bins_all_shuffles = []
    for index, cell in spatial_firing.iterrows():
        rejects_cell = np.empty(number_of_shuffles)
        rejects_cell[:] = np.nan
        for shuffle in range(number_of_shuffles):
            diff_cell = (cell.shuffled_percentile_threshold_95 < cell.shuffled_histograms_hz[shuffle]) + (cell.shuffled_percentile_threshold_5 > cell.shuffled_histograms_hz[shuffle])  # this is a pairwise OR on the binary arrays
            number_of_diffs = diff_cell.sum()
            rejects_cell[shuffle] = number_of_diffs
        rejected_bins_all_shuffles.append(rejects_cell)
    spatial_firing['number_of_different_bins_shuffled'] = rejected_bins_all_shuffles
    return spatial_firing


# this is to find the null distribution of number of rejected null hypothesis based on the shuffled data
# perform B/H analysis on each shuffle and count rejects
def test_if_shuffle_differs_from_other_shuffles_corrected_p_values(spatial_firing, sampling_rate_video, number_of_bars=20):
    number_of_shuffles = len(spatial_firing.shuffled_data.iloc[0])
    rejected_bins_all_shuffles = []
    for index, cell in spatial_firing.iterrows():
        shuffled_histograms = cell['shuffled_data']
        time_spent_in_bins = cell.time_spent_in_bins
        shuffled_data_normalized = shuffled_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        rejects_cell = np.empty(number_of_shuffles)
        rejects_cell[:] = np.nan
        percentile_observed_data_bars = []
        for shuffle in range(number_of_shuffles):
            percentiles_of_observed_bars = np.empty(number_of_bars)
            percentiles_of_observed_bars[:] = np.nan
            for bar in range(number_of_bars):
                observed_data = shuffled_data_normalized[shuffle][bar]
                shuffled_data = shuffled_data_normalized[:, bar]
                percentile_of_observed_data = stats.percentileofscore(shuffled_data, observed_data)
                percentiles_of_observed_bars[bar] = percentile_of_observed_data
            percentile_observed_data_bars.append(percentiles_of_observed_bars)  # percentile of shuffle relative to all other shuffles
            # convert percentile to p value
            percentiles_of_observed_bars[percentiles_of_observed_bars > 50] = 100 - percentiles_of_observed_bars[percentiles_of_observed_bars > 50]
            # correct p values (B/H)
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(percentiles_of_observed_bars, alpha=0.05, method='fdr_bh')
            # count significant bars and put this number in df
            number_of_rejects = reject.sum()
            rejects_cell[shuffle] = number_of_rejects
        rejected_bins_all_shuffles.append(rejects_cell)
    spatial_firing['number_of_different_bins_shuffled_corrected_p'] = rejected_bins_all_shuffles
    return spatial_firing


# calculate percentile of real data relative to shuffled for each bar
def calculate_percentile_of_observed_data(spatial_firing, sampling_rate_video, number_of_bars=20):
    percentile_observed_data_bars = []
    for index, cell in spatial_firing.iterrows():
        shuffled_histograms = cell['shuffled_data']
        time_spent_in_bins = cell.time_spent_in_bins
        shuffled_data_normalized = shuffled_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        percentiles_of_observed_bars = np.empty(number_of_bars)
        percentiles_of_observed_bars[:] = np.nan
        for bar in range(number_of_bars):
            observed_data = cell.hd_histogram_real_data_hz[bar]
            shuffled_data = shuffled_data_normalized[:, bar]
            percentile_of_observed_data = stats.percentileofscore(shuffled_data, observed_data)
            percentiles_of_observed_bars[bar] = percentile_of_observed_data
        percentile_observed_data_bars.append(percentiles_of_observed_bars)
    spatial_firing['percentile_of_observed_data'] = percentile_observed_data_bars
    return spatial_firing


#  convert percentile to p value by subtracting the percentile from 100 when it is > than 50
def convert_percentile_to_p_value(spatial_firing):
    p_values = []
    for index, cell in spatial_firing.iterrows():
        percentile_values = cell.percentile_of_observed_data
        percentile_values[percentile_values > 50] = 100 - percentile_values[percentile_values > 50]
        p_values.append(percentile_values)
    spatial_firing['shuffle_p_values'] = p_values
    return spatial_firing


# perform Benjamini/Hochberg correction on p values calculated from the percentile of observed data relative to shuffled
def calculate_corrected_p_values(spatial_firing, type='bh'):
    corrected_p_values = []
    for index, cell in spatial_firing.iterrows():
        p_values = cell.shuffle_p_values
        if type == 'bh':
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, alpha=0.05, method='fdr_bh')
            corrected_p_values.append(pvals_corrected)
        if type == 'holm':
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, alpha=0.05, method='holm')
            corrected_p_values.append(pvals_corrected)

    field_name = 'p_values_corrected_bars_' + type
    spatial_firing[field_name] = corrected_p_values
    return spatial_firing


def plot_bar_chart_for_cells(spatial_firing, path, animal, shuffle_type='occupancy'):
    counter = 0
    for index, cell in spatial_firing.iterrows():
        mean = cell['shuffled_means']
        std = cell['shuffled_std']
        cell_spikes_hd = np.array(cell['hd'])
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        x_pos = np.arange(shuffled_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.bar(x_pos, mean, yerr=std*2, align='center', alpha=0.7, color='black', ecolor='grey', capsize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        plt.scatter(x_pos, cell.hd_histogram_real_data_hz, marker='o', color='red', s=40)
        plt.savefig(local_path + 'shuffle_analysis_' + animal + '_' + shuffle_type + '/' + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + str(index) + '_SD')
        plt.close()
        counter += 1


def plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, path, animal, shuffle_type='occupancy'):
    counter = 0
    for index, cell in spatial_firing.iterrows():
        mean = cell['shuffled_means']
        percentile_95 = cell['error_bar_95']
        percentile_5 = cell['error_bar_5']
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        x_pos = np.arange(shuffled_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.errorbar(x_pos, mean, yerr=[percentile_5, percentile_95], alpha=0.7, color='black', ecolor='grey', capsize=10, fmt='o', markersize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        plt.scatter(x_pos, cell.hd_histogram_real_data_hz, marker='o', color='navy', s=40)
        plt.title('Number of spikes ' + str(cell.number_of_spikes))
        plt.savefig(local_path + 'shuffle_analysis_' + animal + '_' + shuffle_type + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile')
        plt.close()
        counter += 1


def plot_bar_chart_for_cells_percentile_error_bar_polar(spatial_firing, sampling_rate_video, animal, path, colors=None):
    counter = 0
    for index, cell in spatial_firing.iterrows():
        observed_data_color = 'navy'

        mean = np.append(cell['shuffled_means'], cell['shuffled_means'][0])
        percentile_95 = np.append(cell['error_bar_95'], cell['error_bar_95'][0])
        percentile_5 = np.append(cell['error_bar_5'], cell['error_bar_5'][0])
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        real_data_hz = cell.hd_histogram_real_data_hz
        max_rate = np.round(real_data_hz[~np.isnan(real_data_hz)].max(), 2)
        x_pos = np.array(np.linspace(0, 2 * np.pi, real_data_hz.shape[0] + 1.5))
        significant_bins_to_mark = np.where(cell.p_values_corrected_bars_bh < 0.05)  # indices
        significant_bins_to_mark = x_pos[significant_bins_to_mark[0]]
        y_value_markers = [max_rate + 0.5] * len(significant_bins_to_mark)
        plt.cla()
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        ax.fill_between(x_pos, mean - percentile_5, percentile_95 + mean, color='grey', alpha=0.4)
        ax.plot(x_pos, mean, color='grey', linewidth=5, alpha=0.7)
        observed_data = np.append(real_data_hz, real_data_hz[0])
        ax.plot(x_pos, observed_data, color=observed_data_color, linewidth=5)
        plt.title('\n' + str(max_rate) + ' Hz', fontsize=20, y=1.08)
        if (cell.p_values_corrected_bars_bh < 0.05).sum() > 0:
            ax.scatter(significant_bins_to_mark, y_value_markers, c='red',  marker='*', zorder=3, s=100)
        plt.subplots_adjust(top=0.85)
        plt.savefig(local_path + 'shuffle_analysis_' + animal + '_' + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile_polar')
        plt.close()
        counter += 1


def get_random_indices_for_shuffle(cell, number_of_times_to_shuffle, shuffle_type='occupancy'):
    number_of_spikes_in_field = cell['number_of_spikes']
    length_of_recording = len(cell.trajectory_hd)
    if shuffle_type == 'occupancy':
        shuffle_indices = np.random.randint(0, length_of_recording, size=(number_of_times_to_shuffle, number_of_spikes_in_field))
    else:
        rates = cell.rate_map_values_session  # normalize to make sure it adds up to 1
        rates = np.nan_to_num(rates)  # assign 0 probability to sampling unsampled ranges
        rates /= sum(rates)
        shuffle_indices = np.random.choice(range(0, length_of_recording), size=(number_of_times_to_shuffle, number_of_spikes_in_field), p=rates)
    return shuffle_indices


def interpolate_nans(array_in):
    nans, x = array_utility.nan_helper(array_in)
    array_in[nans] = np.interp(x(nans), x(~nans), array_in[~nans])
    return array_in


# find firing rate on rate map for each sampling point and add to field df
def add_rate_map_values(spatial_firing, cell):
    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size(prm)
    pixel_ratio = prm.get_pixel_ratio()
    spike_data = pd.DataFrame()
    spike_data['x'] = cell.trajectory_x * pixel_ratio / 100
    spike_data['y'] = cell.trajectory_y * pixel_ratio / 100
    spike_data['hd'] = cell.trajectory_hd
    spike_data['synced_time'] = cell.trajectory_times

    spike_data.x = interpolate_nans(spike_data.x)
    spike_data.y = interpolate_nans(spike_data.y)

    spike_data['rate_map_x'] = (spike_data.x // bin_size_pixels).astype(int)
    spike_data['rate_map_y'] = (spike_data.y // bin_size_pixels).astype(int)
    rates = np.zeros(len(spike_data))
    cluster = spatial_firing.cluster_id == cell.cluster_id
    rate_map = cell.firing_maps
    for sample in range(len(spike_data)):
        rate = rate_map[spike_data.rate_map_x.iloc[sample], spike_data.rate_map_y.iloc[sample]]
        rates[sample] = rate
        # plt.scatter(spike_data.rate_map_x.iloc[sample], spike_data.rate_map_y.iloc[sample], color='red', s=50)
        # plt.scatter(spike_data_field.rate_map_x.iloc[sample], spike_data_field.rate_map_y.iloc[sample], color='red',s=50)
    all_rates = np.round(rates, 2)
    return all_rates


def plot_example_shuffle(cell, shuffle, shuffle_indices, iteration_num, shuffle_type, animal):
    plt.cla()
    shuffled_spikes = plt.figure()
    plt.plot(cell.trajectory_x, cell.trajectory_y, color='gray', alpha=0.6)
    plt.scatter(cell['trajectory_x'][shuffle_indices[shuffle]], cell['trajectory_y'][shuffle_indices[shuffle]], s=10)
    shuffled_spikes.set_size_inches(5, 5, forward=True)
    plt.savefig(local_path + 'shuffle_analysis_' + animal + '_' + shuffle_type + '/' + str(iteration_num) + str(cell.session_id) + str(cell.cluster_id) + str(shuffle) + 'shuffled')
    plt.close()

    plt.cla()
    real_spikes = plt.figure()
    plt.plot(cell.trajectory_x, cell.trajectory_y, color='gray', alpha=0.6)
    plt.scatter(cell.position_x, cell. position_y, color='red', s=10)
    real_spikes.set_size_inches(5, 5, forward=True)
    real_spikes.set_size_inches(5, 5, forward=True)
    plt.savefig(local_path + 'shuffle_analysis_' + animal + '_' + shuffle_type + '/' + str(iteration_num) + str(cell.session_id) + str(cell.cluster_id) + str(shuffle) + 'real')
    plt.close()

    hd_shuffle = cell['trajectory_hd'][shuffle_indices[shuffle]]


def downsample_simulated(spatial_firing, downsample_by=33):
    print('Simulated data is downsampled')
    xs = []
    ys = []
    hds = []
    times = []
    for index, cell in spatial_firing.iterrows():
        xs.append(cell.trajectory_x[::downsample_by].values)
        ys.append(cell.trajectory_y[::downsample_by].values)
        hds.append(cell.trajectory_hd[::downsample_by].values)
        times.append(cell.trajectory_times[::downsample_by].values)
    spatial_firing['trajectory_x'] = xs
    spatial_firing['trajectory_y'] = ys
    spatial_firing['trajectory_hd'] = hds
    spatial_firing['trajectory_times'] = times
    return spatial_firing


# add shuffled data to data frame as a new column for each cell
def shuffle_data(spatial_firing, number_of_bins, number_of_times_to_shuffle=1000, animal='mouse', shuffle_type='occupancy'):
    if 'shuffled_data' in spatial_firing:
        return spatial_firing
    if animal == 'simulated':
        downsample_simulated(spatial_firing, downsample_by=33)
    if os.path.exists(local_path + 'shuffle_analysis_' + animal + '_' + shuffle_type) is True:
        shutil.rmtree(local_path + 'shuffle_analysis_' + animal + '_' + shuffle_type)
    os.makedirs(local_path + 'shuffle_analysis_' + animal + '_' + shuffle_type)

    shuffled_histograms_all = []
    iteration_num = 0
    for index, cell in spatial_firing.iterrows():
        iteration_num += 1
        print('I will shuffle data.')
        field_rates = add_rate_map_values(spatial_firing, cell)
        cell['rate_map_values_session'] = field_rates
        shuffled_histograms = np.zeros((number_of_times_to_shuffle, number_of_bins))
        shuffle_indices = get_random_indices_for_shuffle(cell, number_of_times_to_shuffle, shuffle_type='distributive')
        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = cell['trajectory_hd'][shuffle_indices[shuffle]]
            shuffled_hd = (shuffled_hd + 180) * np.pi / 180
            hist, bin_edges = np.histogram(shuffled_hd, bins=number_of_bins, range=(0, 6.28))  # from 0 to 2pi
            shuffled_histograms[shuffle, :] = hist
            if shuffle == 0:
                plot_example_shuffle(cell, shuffle, shuffle_indices, iteration_num, shuffle_type, animal)
        shuffled_histograms_all.append(shuffled_histograms)
    spatial_firing['shuffled_data'] = shuffled_histograms_all

    if animal == 'mouse':
        spatial_firing.to_pickle(local_path_mouse)

    if animal == 'rat':
        spatial_firing.to_pickle(local_path_rat)

    # if animal == 'simulated':
        # spatial_firing.to_pickle(local_path_simulated)

    return spatial_firing


def analyze_shuffled_data(spatial_firing, save_path, sampling_rate_video, animal, number_of_bins=20, shuffle_type='occupancy'):
    if 'number_of_different_bins_shuffled_corrected_p' in spatial_firing:
        return spatial_firing
    print('Analyze shuffled data.')
    spatial_firing = add_mean_and_std_to_df(spatial_firing, sampling_rate_video, number_of_bins)
    spatial_firing = add_percentile_values_to_df(spatial_firing, sampling_rate_video, number_of_bins=20)
    spatial_firing = test_if_real_hd_differs_from_shuffled(spatial_firing)  # is the observed data within 95th percentile of the shuffled?
    spatial_firing = test_if_shuffle_differs_from_other_shuffles(spatial_firing)

    spatial_firing = calculate_percentile_of_observed_data(spatial_firing, sampling_rate_video, number_of_bins)  # this is relative to shuffled data
    # field_data = calculate_percentile_of_shuffled_data(field_data, number_of_bars=20)
    spatial_firing = convert_percentile_to_p_value(spatial_firing)  # this is needed to make it 2 tailed so diffs are picked up both ways
    spatial_firing = calculate_corrected_p_values(spatial_firing, type='bh')  # BH correction on p values from previous function
    spatial_firing = calculate_corrected_p_values(spatial_firing, type='holm')  # Holm correction on p values from previous function
    spatial_firing = count_number_of_significantly_different_bars_per_field(spatial_firing, significance_level=95, type='bh')
    spatial_firing = count_number_of_significantly_different_bars_per_field(spatial_firing, significance_level=95, type='holm')
    spatial_firing = test_if_shuffle_differs_from_other_shuffles_corrected_p_values(spatial_firing, sampling_rate_video, number_of_bars=20)
    # plot_bar_chart_for_cells(spatial_firing, save_path, animal, shuffle_type=shuffle_type)
    plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, save_path, animal, shuffle_type=shuffle_type)
    plot_bar_chart_for_cells_percentile_error_bar_polar(spatial_firing, sampling_rate_video, animal, save_path, colors=None)
    # spatial_firing.to_pickle(save_path)
    return spatial_firing


def find_tail_of_shuffled_distribution_of_rejects(shuffled_field_data):
    number_of_rejects = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects:
        flat_shuffled.extend(field)
    tail = max(flat_shuffled)
    percentile_95 = np.percentile(flat_shuffled, 95)
    percentile_99 = np.percentile(flat_shuffled, 99)
    return tail, percentile_95, percentile_99


def plot_histogram_of_number_of_rejected_bars(shuffled_field_data, animal='mouse', shuffle_type='occupancy'):
    number_of_rejects = shuffled_field_data.number_of_different_bins
    fig, ax = plt.subplots()
    plt.hist(number_of_rejects)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim(0, 20.5)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Proportion', size=30)
    plt.savefig(local_path + 'distribution_of_rejects_' + animal + shuffle_type + '.png', bbox_inches="tight")
    plt.close()


def plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_data, animal='mouse', shuffle_type='occupancy'):
    number_of_rejects = shuffled_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for cell in number_of_rejects:
        flat_shuffled.extend(cell)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_xlim(0, 20.5)
    plt.savefig(local_path + '/distribution_of_rejects_shuffled' + animal + shuffle_type + '.png', bbox_inches="tight")
    plt.close()


def make_combined_plot_of_distributions(shuffled_data, tag='grid', shuffle_type = 'occupancy'):
    tail, percentile_95, percentile_99 = find_tail_of_shuffled_distribution_of_rejects(shuffled_data)

    number_of_rejects_shuffled = shuffled_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for cell in number_of_rejects_shuffled:
        flat_shuffled.extend(cell)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, normed=True, color='black', alpha=0.5)

    number_of_rejects_real = shuffled_data.number_of_different_bins
    plt.hist(number_of_rejects_real, normed=True, color='navy', alpha=0.5)

    # plt.axvline(x=tail, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(x=percentile_95, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(x=percentile_99, color='red', alpha=0.5, linestyle='dashed')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_xlim(0, 20.5)
    plt.savefig(local_path + 'distribution_of_rejects_combined_all_' + tag + shuffle_type + '.png', bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    plt.ylim(0, 1.01)
    plt.yticks([0, 1])
    ax = plot_utility.format_bar_chart(ax, 'Rejected bars / cell', 'Cumulative probability')
    values, base = np.histogram(flat_shuffled, bins=40)
    cumulative = np.cumsum(values / len(flat_shuffled))
    plt.plot(base[:-1], cumulative, c='gray', linewidth=5)

    values, base = np.histogram(number_of_rejects_real, bins=40)
    cumulative = np.cumsum(values / len(number_of_rejects_real))
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim(0, 20.5)
    ax.set_xlabel('Rejected bars / cell', size=30)
    ax.set_ylabel('Cumulative probability', size=30)
    plt.savefig(local_path + 'distribution_of_rejects_' + tag + shuffle_type + '_cumulative.png', bbox_inches="tight")
    plt.close()


def plot_number_of_significant_p_values(spatial_firing, type='bh', shuffle_type='occupancy'):
    if type == 'bh':
        number_of_significant_p_values = spatial_firing.number_of_different_bins_bh
    else:
        number_of_significant_p_values = spatial_firing.number_of_different_bins_holm

    fig, ax = plt.subplots()
    plt.hist(number_of_significant_p_values, normed='True', color='navy', alpha=0.5)
    flat_shuffled = []
    for cell in spatial_firing.number_of_different_bins_shuffled_corrected_p:
        flat_shuffled.extend(cell)
    plt.hist(flat_shuffled, normed='True', color='gray', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Significant bars / cell', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_ylim(0, 0.2)
    ax.set_xlim(0, 20.5)
    plt.savefig(local_path + 'distribution_of_rejects_significant_p_ ' + type + shuffle_type + '.png', bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    # plt.xscale('log')
    plt.yticks([0, 1])
    plt.ylim(0, 1.01)
    ax = plot_utility.format_bar_chart(ax, 'Significant bars / cell', 'Cumulative probability')
    values, base = np.histogram(flat_shuffled, bins=40)
    cumulative = np.cumsum(values / len(flat_shuffled))
    plt.plot(base[:-1], cumulative, c='gray', linewidth=5)

    values, base = np.histogram(number_of_significant_p_values, bins=40)
    cumulative = np.cumsum(values / len(number_of_significant_p_values))
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim(0, 20.5)
    ax.set_ylabel('Cumulative probability', size=30)
    plt.savefig(local_path + 'distribution_of_rejects_signigicant_p' + type + shuffle_type + '_cumulative.png')
    plt.close()


def compare_distributions(x, y):
    # stat, p = scipy.stats.mannwhitneyu(x, y)
    stat, p = scipy.stats.ranksums(x, y)
    print('p value and test statistic for MW U test:')
    print(p)
    print(stat)
    return p


def compare_shuffled_to_real_data_mw_test(spatial_firing, analysis_type='bh'):
    num_bins = 20
    if analysis_type == 'bh':
        flat_shuffled = []
        for cell in spatial_firing.number_of_different_bins_shuffled_corrected_p:
            flat_shuffled.extend(cell)
        p_bh = compare_distributions(spatial_firing.number_of_different_bins_bh, flat_shuffled)
        print('Number of cells: ' + str(len(spatial_firing)))
        print('p value for comparing shuffled distribution to B-H corrected p values: ' + str(p_bh))
        number_of_significant_bins = spatial_firing.number_of_different_bins_bh.sum()
        total_number_of_bins = len(spatial_firing.number_of_different_bins_bh) * num_bins
        print(str(number_of_significant_bins) + ' out of ' + str(total_number_of_bins) + ' are significant')
        print(str(np.mean(spatial_firing.number_of_different_bins_bh)) + ' number of bins per cell +/- ' + str(np.std(spatial_firing.number_of_different_bins_bh)) + ' SD')
        print('shuffled: ')
        print(str(np.mean(flat_shuffled)) + ' number of bins per cell +/- ' + str(np.std(flat_shuffled)) + ' SD')
        return p_bh

    if analysis_type == 'percentile':
        flat_shuffled = []
        for cell in spatial_firing.number_of_different_bins_shuffled:
            flat_shuffled.extend(cell)
        p_percentile = compare_distributions(spatial_firing.number_of_different_bins, flat_shuffled)
        print('p value for comparing shuffled distribution to percentile thresholded p values: ' + str(p_percentile))
        number_of_significant_bins = spatial_firing.number_of_different_bins.sum()
        total_number_of_bins = len(spatial_firing.number_of_different_bins) * num_bins
        print(str(number_of_significant_bins) + ' out of ' + str(total_number_of_bins) + ' are different')
        return p_percentile


def plot_distributions_for_shuffled_vs_real_cells(shuffled_spatial_firing_data, tag='grid', animal='mouse', shuffle_type='occupancy'):
    plot_histogram_of_number_of_rejected_bars(shuffled_spatial_firing_data, animal, shuffle_type=shuffle_type)
    plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_spatial_firing_data, animal)
    plot_number_of_significant_p_values(shuffled_spatial_firing_data, type='bh_' + tag + '_' + animal, shuffle_type=shuffle_type)
    plot_number_of_significant_p_values(shuffled_spatial_firing_data, type='holm_' + tag + '_' + animal, shuffle_type=shuffle_type)
    make_combined_plot_of_distributions(shuffled_spatial_firing_data, tag=tag + '_' + animal, shuffle_type=shuffle_type)


def process_data(spatial_firing, sampling_rate_video, local_path, animal='mouse', shuffle_type='occupancy'):
    if animal == 'mouse':
        spatial_firing = tag_false_positives(spatial_firing)
    else:
        spatial_firing['false_positive'] = False
    if animal == 'simulated':
        downsample_by = 33
        sampling_rate_video /= downsample_by

    good_cell = spatial_firing.false_positive == False
    spatial_firing = shuffle_data(spatial_firing[good_cell], 20, number_of_times_to_shuffle=1000, animal=animal, shuffle_type=shuffle_type)
    spatial_firing = analyze_shuffled_data(spatial_firing, local_path, sampling_rate_video, animal, number_of_bins=20, shuffle_type=shuffle_type)
    print('I finished the shuffled analysis on ' + animal + ' data.\n')

    grid = spatial_firing.grid_score >= 0.4
    hd = spatial_firing.hd_score >= 0.5
    not_classified = np.logical_and(np.logical_not(grid), np.logical_not(hd))
    hd_cells = np.logical_and(np.logical_not(grid), hd)
    grid_cells = np.logical_and(grid, np.logical_not(hd))

    shuffled_spatial_firing_grid = spatial_firing[grid_cells & good_cell]
    shuffled_spatial_firing_not_classified = spatial_firing[not_classified & good_cell]

    get_number_of_directional_cells(shuffled_spatial_firing_grid, tag='grid')
    plot_distributions_for_shuffled_vs_real_cells(shuffled_spatial_firing_grid, 'grid', animal=animal, shuffle_type=shuffle_type)
    if len(shuffled_spatial_firing_not_classified) > 0:
        plot_distributions_for_shuffled_vs_real_cells(shuffled_spatial_firing_not_classified, 'not_classified', animal=animal, shuffle_type=shuffle_type)

    print(animal + ' data:')
    print('Grid cells:')
    print(shuffle_type)
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_grid, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_grid, analysis_type='percentile')
    print('Not classified cells:')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_not_classified, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_not_classified, analysis_type='percentile')


def get_number_of_directional_cells(cells, tag='grid'):
    print('HEAD DIRECTION')
    percentiles_no_correction = []
    percentiles_correction = []
    for index, cell in cells.iterrows():
        percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled, cell.number_of_different_bins)
        percentiles_no_correction.append(percentile)

        percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled_corrected_p, cell.number_of_different_bins_bh)
        percentiles_correction.append(percentile)

    print(tag)
    print('Number of fields: ' + str(len(cells)))
    print('Number of directional cells [without correction]: ')
    print(np.sum(np.array(percentiles_no_correction) > 95))
    cells['directional_no_correction'] = np.array(percentiles_no_correction) > 95

    print('Number of directional cells [with BH correction]: ')
    print(np.sum(np.array(percentiles_correction) > 95))
    cells['directional_correction'] = np.array(percentiles_correction) > 95
    cells.to_pickle(local_path + tag + 'cells.pkl')


def process_downsampled_data(spatial_firing, sampling_rate_video, local_path, animal='mouse', shuffle_type='occupancy'):
    if animal == 'mouse':
        spatial_firing = tag_false_positives(spatial_firing)
    else:
        spatial_firing['false_positive'] = False
    if animal == 'simulated':
        downsample_by = 33
        sampling_rate_video /= downsample_by

    good_cell = spatial_firing.false_positive == False

    spatial_firing = shuffle_data(spatial_firing[good_cell], 20, number_of_times_to_shuffle=1000, animal=animal, shuffle_type=shuffle_type)
    spatial_firing = analyze_shuffled_data(spatial_firing, local_path, sampling_rate_video, animal, number_of_bins=20, shuffle_type=shuffle_type)
    print('I finished the shuffled analysis on ' + animal + ' data.\n')

    grid = spatial_firing.grid_score >= 0.4
    hd = spatial_firing.hd_score >= 0.5
    not_classified = np.logical_and(np.logical_not(grid), np.logical_not(hd))
    hd_cells = np.logical_and(np.logical_not(grid), hd)
    grid_cells = np.logical_and(grid, np.logical_not(hd))

    shuffled_spatial_firing_grid = spatial_firing[grid_cells & good_cell]
    shuffled_spatial_firing_not_classified = spatial_firing[not_classified & good_cell]

    plot_distributions_for_shuffled_vs_real_cells(shuffled_spatial_firing_grid, 'grid', animal=animal, shuffle_type=shuffle_type)
    if len(shuffled_spatial_firing_not_classified) > 0:
        plot_distributions_for_shuffled_vs_real_cells(shuffled_spatial_firing_not_classified, 'not_classified', animal=animal, shuffle_type=shuffle_type)

    print(animal + ' data:')
    print('Grid cells:')
    print(shuffle_type)
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_grid, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_grid, analysis_type='percentile')
    print('Not classified cells:')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_not_classified, analysis_type='bh')
    compare_shuffled_to_real_data_mw_test(shuffled_spatial_firing_not_classified, analysis_type='percentile')


def main():

    spatial_firing_all_rats = load_data_frame_spatial_firing(local_path_rat, server_path_rat, spike_sorter='')
    prm.set_pixel_ratio(100)
    process_data(spatial_firing_all_rats, 50, local_path_rat, animal='rat', shuffle_type='distributive')
    prm.set_pixel_ratio(440)
    spatial_firing_all_mice = load_data_frame_spatial_firing(local_path_mouse, server_path_mouse, spike_sorter='/MountainSort')
    process_data(spatial_firing_all_mice, 30, local_path_mouse, animal='mouse', shuffle_type='distributive')
    '''

    local_path_df_ventral_narrow = local_path + 'all_simulated_df_ventral_narrow.pkl'
    spatial_firing_all_simulated = load_data_frame_spatial_firing(local_path_df_ventral_narrow, server_path_simulated + 'ventral_narrow/', spike_sorter='', df_path='')
    prm.set_pixel_ratio(100)
    process_data(spatial_firing_all_simulated, 1000, local_path_df_ventral_narrow, animal='simulated', shuffle_type='distributive_narrow')
    local_path_df_control_narrow = local_path + 'all_simulated_df_control_narrow.pkl'
    spatial_firing_all_simulated = load_data_frame_spatial_firing(local_path_df_control_narrow, server_path_simulated + 'control_narrow/', spike_sorter='', df_path='')
    process_data(spatial_firing_all_simulated, 1000, local_path_df_control_narrow, animal='simulated', shuffle_type='distributive_control_narrow')
    '''


if __name__ == '__main__':
    main()
