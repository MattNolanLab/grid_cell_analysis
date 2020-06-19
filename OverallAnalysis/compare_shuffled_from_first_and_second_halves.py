import glob
import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.false_positives
import OverallAnalysis.open_field_firing_maps_processed_data
import pandas as pd
import plot_utility
import OverallAnalysis.shuffle_cell_analysis
import PostSorting.compare_first_and_second_half
import PostSorting.open_field_head_direction
import PostSorting.open_field_firing_maps
import PostSorting.parameters
import scipy.stats
from scipy import signal

from statsmodels.sandbox.stats.multicomp import multipletests

prm = PostSorting.parameters.Parameters()

local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/compare_first_and_second_shuffled/'
local_path_mouse = local_path + 'all_mice_df.pkl'
local_path_mouse_down_sampled = local_path + 'all_mice_df_down_sampled.pkl'
local_path_rat = local_path + 'all_rats_df.pkl'

server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
path_to_simulated = local_path + 'simulated.pkl'


def load_simulated_data():
    path = OverallAnalysis.folder_path_settings.get_local_path() + 'simulated_25min/25/'
    spatial_firing_combined = pd.DataFrame()
    position_combined = pd.DataFrame()
    for recording_folder in glob.glob(path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/spatial_firing.pkl'
        data_frame_path_position = recording_folder + '/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            position = pd.read_pickle(data_frame_path_position)
            spatial_firing['trajectory_x'] = [position.position_x]
            spatial_firing['trajectory_y'] = [position.position_y]
            spatial_firing['trajectory_times'] = [position.synced_time]
            spatial_firing['trajectory_hd'] = [position.hd]
            spatial_firing_combined = spatial_firing_combined.append(spatial_firing)
            position_combined = position_combined.append(position)
    spatial_firing_combined.to_pickle(local_path + 'all_simulated_df.pkl')
    return spatial_firing_combined


def make_summary_figures(tag):
    if os.path.exists(local_path + tag + '_aggregated_data.pkl'):
        stats = pd.read_pickle(local_path + tag + '_aggregated_data.pkl')
        percentiles_vs_shuffled_plot = plt.figure()
        percentiles_vs_shuffled_plot.set_size_inches(5, 5, forward=True)
        ax = percentiles_vs_shuffled_plot.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        # ax = plot_utility.plot_cumulative_histogram(stats.shuffled_corr_median / 100, ax, color='gray', number_of_bins=100)
        # ax = plot_utility.plot_cumulative_histogram_from_zero(stats.shuffled_percentiles / 100, ax, color='grey', number_of_bins=100)
        # diagonal_line = np.linspace(0, 1, len(stats.percentiles))
        # ax = plot_utility.plot_cumulative_histogram_from_zero(diagonal_line, ax, color='grey', number_of_bins=100)
        ax = plot_utility.plot_cumulative_histogram_from_zero(stats.percentiles / 100, ax, color='navy', number_of_bins=100)
        plt.savefig(local_path + tag + 'percentiles_corr_vs_median_of_shuffled.png')

        '''
        d, p = scipy.stats.ks_2samp(stats.percentiles, diagonal_line * 100)
        print('KS test between observed and shuffled percentiles for correlation (D, p):')
        print(d)
        print(p)

        plt.cla()
        stats = pd.read_pickle(local_path + tag + '_aggregated_data.pkl')
        percentiles_vs_shuffled_plot = plt.figure()
        percentiles_vs_shuffled_plot.set_size_inches(5, 5, forward=True)
        ax = percentiles_vs_shuffled_plot.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax = plot_utility.plot_cumulative_histogram(stats.percentiles / 100, ax, color='navy', number_of_bins=100)
        plt.savefig(local_path + tag + 'percentiles_corr.png')
        '''

        percentile_values = stats.percentiles
        percentile_values[percentile_values > 50] = 100 - percentile_values[percentile_values > 50]
        reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(percentile_values, alpha=0.05, method='fdr_bh')
        print('Number of significantly correlation cells after BH correction:')
        print((pvals_corrected < 0.05).sum())


def add_cell_types_to_data_frame(spatial_firing):
    cell_type = []
    for index, cell in spatial_firing.iterrows():
        if cell.hd_score >= 0.5 and cell.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif cell.hd_score >= 0.5:
            cell_type.append('hd')
        elif cell.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    spatial_firing['cell type'] = cell_type

    return spatial_firing


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


def load_data(path):
    first_half_spatial_firing = None
    second_half_spatial_firing = None
    first_position = None
    second_position = None
    if os.path.exists(path + '/first_half/DataFrames/spatial_firing.pkl'):
        first_half_spatial_firing = pd.read_pickle(path + '/first_half/DataFrames/spatial_firing.pkl')
    else:
        return None, None, None, None
    if os.path.exists(path + '/second_half/DataFrames/spatial_firing.pkl'):
        second_half_spatial_firing = pd.read_pickle(path + '/second_half/DataFrames/spatial_firing.pkl')
    else:
        return None, None, None, None

    if os.path.exists(path + '/first_half/DataFrames/position.pkl'):
        first_position = pd.read_pickle(path + '/first_half/DataFrames/position.pkl')
    else:
        return None, None, None, None
    if os.path.exists(path + '/second_half/DataFrames/position.pkl'):
        second_position = pd.read_pickle(path + '/second_half/DataFrames/position.pkl')
    else:
        return None, None, None, None
    return first_half_spatial_firing, second_half_spatial_firing, first_position, second_position


def split_in_two(cell):
    cell['position_x_pixels'] = [np.array(cell.position_x.iloc[0]) * prm.get_pixel_ratio() / 100]
    cell['position_y_pixels'] = [np.array(cell.position_y.iloc[0]) * prm.get_pixel_ratio() / 100]
    spike_data_in = cell
    synced_spatial_data_in = pd.DataFrame()
    synced_spatial_data_in['position_x'] = cell.trajectory_x.iloc[0]
    synced_spatial_data_in['position_y'] = cell.trajectory_y.iloc[0]
    synced_spatial_data_in['synced_time'] = cell.trajectory_times.iloc[0]
    synced_spatial_data_in['hd'] = cell.trajectory_hd.iloc[0]
    spike_data_cluster_first, synced_spatial_data_first_half = PostSorting.compare_first_and_second_half.get_half_of_the_data_cell(prm, spike_data_in, synced_spatial_data_in, half='first_half')
    spike_data_cluster_second, synced_spatial_data_second_half = PostSorting.compare_first_and_second_half.get_half_of_the_data_cell(prm, spike_data_in, synced_spatial_data_in, half='second_half')

    synced_spatial_data_first_half['position_x_pixels'] = np.array(synced_spatial_data_first_half.position_x) * prm.get_pixel_ratio() / 100
    synced_spatial_data_first_half['position_y_pixels'] = np.array(synced_spatial_data_first_half.position_y) * prm.get_pixel_ratio() / 100
    synced_spatial_data_second_half['position_x_pixels'] = np.array(synced_spatial_data_second_half.position_x) * prm.get_pixel_ratio() / 100
    synced_spatial_data_second_half['position_y_pixels'] = np.array(synced_spatial_data_second_half.position_y) * prm.get_pixel_ratio() / 100

    first = pd.DataFrame()
    first['session_id'] = [cell.session_id.iloc[0]]
    first['cluster_id'] = [cell.cluster_id.iloc[0]]
    first['number_of_spikes'] = [len(spike_data_cluster_first.firing_times)]
    first['firing_times'] = [spike_data_cluster_first.firing_times]
    first['position_x'] = [spike_data_cluster_first.position_x]
    first['position_y'] = [spike_data_cluster_first.position_y]
    first['position_x_pixels'] = [spike_data_cluster_first.position_x_pixels]
    first['position_y_pixels'] = [spike_data_cluster_first.position_y_pixels]
    first['hd'] = [spike_data_cluster_first.hd]

    first['trajectory_x'] = [synced_spatial_data_first_half.position_x]
    first['trajectory_y'] = [synced_spatial_data_first_half.position_y]
    first['trajectory_hd'] = [synced_spatial_data_first_half.hd]
    first['trajectory_times'] = [synced_spatial_data_first_half.synced_time]

    second = pd.DataFrame()
    second['session_id'] = [cell.session_id.iloc[0]]
    second['cluster_id'] = [cell.cluster_id.iloc[0]]
    second['number_of_spikes'] = [len(spike_data_cluster_second.firing_times)]
    second['firing_times'] = [spike_data_cluster_second.firing_times]
    second['position_x'] = [spike_data_cluster_second.position_x]
    second['position_y'] = [spike_data_cluster_second.position_y]
    second['position_x_pixels'] = [spike_data_cluster_second.position_x_pixels]
    second['position_y_pixels'] = [spike_data_cluster_second.position_y_pixels]
    second['hd'] = [spike_data_cluster_second.hd]

    second['trajectory_x'] = [synced_spatial_data_second_half.position_x.reset_index(drop=True)]
    second['trajectory_y'] = [synced_spatial_data_second_half.position_y.reset_index(drop=True)]
    second['trajectory_hd'] = [synced_spatial_data_second_half.hd.reset_index(drop=True)]
    second['trajectory_times'] = [synced_spatial_data_second_half.synced_time.reset_index(drop=True)]
    return first, second, synced_spatial_data_first_half, synced_spatial_data_second_half


def plot_observed_vs_shuffled_correlations(observed, shuffled, cell):
    hd_polar_fig = plt.figure()
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    plt.hist(shuffled.flatten(), color='gray', alpha=0.8)
    ax.axvline(observed, color='navy', linewidth=3)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.xlim(-1, 1)
    plt.xticks([-1, 0, 1])
    plt.yticks([0, 150000, 300000])
    ax.set_yticklabels([0, '', 3])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('r', fontsize=24)
    plt.ylabel('N (shuffles)', fontsize=24)
    plt.tight_layout()
    plt.savefig(local_path + cell.session_id[0] + str(cell.cluster_id[0]) + '_corr_coefs.png')
    plt.close()


def plot_summary_stats(animal, grid_data, percentiles, hd_scores, number_of_spikes):
    plt.cla()
    plt.scatter(hd_scores, percentiles, color='navy')
    plt.xlabel('Head direction score', fontsize=20)
    plt.ylabel('Percentile of correlation coef.', fontsize=20)
    plt.tight_layout()
    plt.savefig(local_path + animal + 'pearson_coef_percentile_vs_hd_score.png')
    plt.close()

    plt.cla()
    plt.scatter(number_of_spikes, percentiles, color='navy')
    plt.xlabel('Number of spikes', fontsize=20)
    plt.ylabel('Percentile of correlation coef.', fontsize=20)
    plt.tight_layout()
    plt.savefig(local_path + animal + 'pearson_coef_percentile_vs_number_of_spikes.png')
    plt.close()


def print_summary_stat_results(corr_coefs_mean, percentiles, tag):
    print('***********' + tag + '***************')
    print('avg corr correlations ')
    print(corr_coefs_mean)
    print('mean:')
    print(np.mean(corr_coefs_mean))
    print('std:')
    print(np.std(corr_coefs_mean))

    print('percentiles')
    print(percentiles)
    print('mean percentiles: ' + str(np.mean(percentiles)))
    print('sd percentiles: ' + str(np.std(percentiles)))
    print('number of cells in 95 percentile: ' + str(len(np.where(np.array(percentiles) > 95)[0])))
    print('number of all grid cells: ' + str(len(percentiles)))


def add_hd_histogram_of_observed_data_to_df(cells, sampling_rate_video, number_of_bins=20, binning='not_smooth'):
    if binning == 'not_smooth':
        angles_session = np.array(cells.trajectory_hd[0])
        low_end = angles_session[~np.isnan(angles_session)].min()
        high_end = angles_session[~np.isnan(angles_session)].max()
        hd_hist_session = np.histogram(angles_session, bins=number_of_bins, range=(low_end, high_end))[0]
        angles_spike = cells.hd[0]
        low_end = angles_spike[~np.isnan(angles_spike)].min()
        high_end = angles_spike[~np.isnan(angles_spike)].max()
        real_data_hz = np.histogram(angles_spike, bins=number_of_bins, range=(low_end, high_end))[0] * sampling_rate_video / hd_hist_session
        cells['hd_histogram_real_data_hz'] = [real_data_hz]
    else:
        angles_session = np.array(cells.trajectory_hd[0])
        hd_hist_session = PostSorting.open_field_head_direction.get_hd_histogram(angles_session)
        hd_hist_session /= prm.get_sampling_rate()
        angles_spike = cells.hd[0]
        hd_hist_spikes = PostSorting.open_field_head_direction.get_hd_histogram(angles_spike)
        cells['hd_histogram_real_data_hz'] = [hd_hist_spikes / hd_hist_session / 1000]
    return cells


def process_data(server_path, spike_sorter='/MountainSort', df_path='/DataFrames', sampling_rate_video=30, tag='mouse'):
    all_data = pd.read_pickle(local_path + 'all_' + tag + '_df.pkl')
    all_data = add_cell_types_to_data_frame(all_data)
    grid_cells = all_data['cell type'] == 'grid'
    grid_data = all_data[grid_cells]

    iterator = 0
    corr_coefs_mean = []
    corr_stds = []
    percentiles = []
    shuffled_percentiles = [] # this will be used as a baseline for expected percentiles
    number_of_spikes = []
    hd_scores = []
    col_names = ['session_id', 'cluster_id', 'corr_coefs_mean', 'shuffled_corr_median', 'corr_stds', 'percentiles', 'shuffled_percentiles', 'hd_scores_all',
                 'number_of_spikes_all']
    aggregated_data = pd.DataFrame(columns=col_names)
    for iterator in range(len(grid_data)):
        try:
            print(iterator)
            print(grid_data.iloc[iterator].session_id)
            first_half, second_half, position_first, position_second = split_in_two(grid_data.iloc[iterator:iterator + 1])
            # add rate map to dfs
            # shuffle
            position_heat_map_first, first_half = OverallAnalysis.open_field_firing_maps_processed_data.make_firing_field_maps(position_first, first_half, prm)
            spatial_firing_first = OverallAnalysis.shuffle_cell_analysis.shuffle_data(first_half, 20, number_of_times_to_shuffle=1000, animal=tag + '_first_half', shuffle_type='distributive')
            spatial_firing_first = OverallAnalysis.shuffle_cell_analysis.add_mean_and_std_to_df(spatial_firing_first, sampling_rate_video, number_of_bins=20)
            spatial_firing_first = OverallAnalysis.shuffle_cell_analysis.analyze_shuffled_data(spatial_firing_first, local_path, sampling_rate_video, tag + str(iterator) + 'first',
                                                   number_of_bins=20, shuffle_type='distributive')

            # OverallAnalysis.shuffle_cell_analysis.plot_distributions_for_shuffled_vs_real_cells(spatial_firing_first, 'grid', animal=tag + str(iterator) + 'first', shuffle_type='distributive')

            position_heat_map_second, second_half = OverallAnalysis.open_field_firing_maps_processed_data.make_firing_field_maps(position_second, second_half, prm)
            spatial_firing_second = OverallAnalysis.shuffle_cell_analysis.shuffle_data(second_half, 20, number_of_times_to_shuffle=1000, animal=tag + '_second_half', shuffle_type='distributive')
            spatial_firing_second = OverallAnalysis.shuffle_cell_analysis.add_mean_and_std_to_df(spatial_firing_second, sampling_rate_video, number_of_bins=20)
            spatial_firing_second = OverallAnalysis.shuffle_cell_analysis.analyze_shuffled_data(spatial_firing_second, local_path, sampling_rate_video, tag + str(iterator) + 'second',
                                                   number_of_bins=20, shuffle_type='distributive')
            # OverallAnalysis.shuffle_cell_analysis.plot_distributions_for_shuffled_vs_real_cells(spatial_firing_second, 'grid', animal=tag + str(iterator) + 'second', shuffle_type='distributive')

            print('shuffled')
            # compare
            time_spent_in_bins_first = spatial_firing_first.time_spent_in_bins  # based on trajectory
            # normalize shuffled data
            shuffled_histograms_hz_first = spatial_firing_first.shuffled_data * sampling_rate_video / time_spent_in_bins_first
            time_spent_in_bins_second = spatial_firing_second.time_spent_in_bins  # based on trajectory
            # normalize shuffled data
            shuffled_histograms_hz_second = spatial_firing_second.shuffled_data * sampling_rate_video / time_spent_in_bins_second

            # look at correlations between rows of the two arrays above to get a distr of correlations for the shuffled data
            corr = np.corrcoef(shuffled_histograms_hz_first[0], shuffled_histograms_hz_second[0])[1000:, :1000]
            corr_mean = corr.mean()
            corr_std = corr.std()
            shuffled_corr_median = np.median(corr)
            # check what percentile real value is relative to distribution of shuffled correlations
            spatial_firing_first = add_hd_histogram_of_observed_data_to_df(spatial_firing_first, sampling_rate_video,
                                                                           number_of_bins=20, binning='not_smooth')
            spatial_firing_second = add_hd_histogram_of_observed_data_to_df(spatial_firing_second, sampling_rate_video,
                                                                            number_of_bins=20, binning='not_smooth')
            corr_observed = scipy.stats.pearsonr(spatial_firing_first.hd_histogram_real_data_hz[0], spatial_firing_second.hd_histogram_real_data_hz[0])[0]

            plot_observed_vs_shuffled_correlations(corr_observed, corr, spatial_firing_first)

            percentile = scipy.stats.percentileofscore(corr.flatten(), corr_observed)
            shuffled_percentile = scipy.stats.percentileofscore(corr.flatten(), corr[0][0])
            percentiles.append(percentile)
            shuffled_percentiles.append(shuffled_percentile)
            number_of_spikes.append(grid_data.iloc[iterator].number_of_spikes)
            hd_scores.append(grid_data.iloc[iterator].hd_score)

            corr_coefs_mean.append(corr_mean)
            corr_stds.append(corr_std)

            aggregated_data = aggregated_data.append({
                "session_id": grid_data.iloc[iterator].session_id,
                "cluster_id":  grid_data.iloc[iterator].cluster_id,
                "corr_coefs_mean": corr_mean,
                "shuffled_corr_median": shuffled_corr_median,
                "corr_stds": corr_std,
                "percentiles": percentile,
                "shuffled_percentiles": shuffled_percentile,
                "hd_scores_all": grid_data.iloc[iterator].hd_score,
                "number_of_spikes_all": grid_data.iloc[iterator].number_of_spikes
            }, ignore_index=True)

        except:
            print('i failed...')

    print_summary_stat_results(corr_coefs_mean, percentiles, tag)
    plot_summary_stats(tag, grid_data, percentiles, hd_scores, number_of_spikes)
    aggregated_data.to_pickle(local_path + tag + '_aggregated_data.pkl')


def main():
    # load_simulated_data()
    # prm.set_pixel_ratio(100)
    # prm.set_sampling_rate(1000)
    # process_data(path_to_simulated, tag='simulated')
    prm.set_pixel_ratio(100)
    prm.set_sampling_rate(1)  # firing times are in seconds for rat data
    process_data(server_path_rat, tag='rats')
    make_summary_figures('mice')
    make_summary_figures('rats')
    prm.set_pixel_ratio(440)
    prm.set_sampling_rate(30000)
    process_data(server_path_mouse, tag='mice')



if __name__ == '__main__':
    main()