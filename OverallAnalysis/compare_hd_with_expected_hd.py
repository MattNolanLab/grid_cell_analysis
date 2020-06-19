import glob
import matplotlib.pylab as plt
import math_utility
import math
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots
import plot_utility
from rpy2 import robjects as robj
from scipy.stats import circstd
from rpy2.robjects import pandas2ri
import scipy.stats
import seaborn
import PostSorting.compare_first_and_second_half
import PostSorting.open_field_head_direction
import PostSorting.parameters

prm = PostSorting.parameters.Parameters()
prm.set_sorter_name('MountainSort')


local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/compare_hd_with_expected_hd/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def load_field_data(output_path, server_path, spike_sorter, animal):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data
    field_data_combined = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/shuffled_fields.pkl'
        spatial_firing_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)
            spatial_firing = pd.read_pickle(spatial_firing_path)
            position = pd.read_pickle(position_path)
            prm.set_file_path(recording_folder)
            # spatial_firing = PostSorting.compare_first_and_second_half.analyse_first_and_second_halves(prm, position, spatial_firing)
            if 'shuffled_data' in field_data:
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'position_x_spikes',
                                                    'position_y_spikes', 'position_x_session', 'position_y_session',
                                                    'indices_rate_map', 'hd_in_field_spikes',
                                                    'hd_in_field_session', 'spike_times', 'times_session',
                                                    'time_spent_in_field', 'number_of_spikes_in_field']].copy()
                field_data_to_combine['normalized_hd_hist'] = field_data.hd_hist_spikes / field_data.hd_hist_session
                if 'hd_score' in field_data:
                    field_data_to_combine['hd_score'] = field_data.hd_score
                if 'grid_score' in field_data:
                    field_data_to_combine['grid_score'] = field_data.grid_score
                rate_maps = []
                length_recording = []
                position_xs = []
                position_ys = []
                synced_times = []
                hds = []

                for cluster in range(len(field_data.cluster_id)):
                    rate_map = spatial_firing[field_data.cluster_id.iloc[cluster] == spatial_firing.cluster_id].firing_maps
                    rate_maps.append(rate_map)
                    length_of_recording = (position.synced_time.max() - position.synced_time.min())
                    length_recording.append(length_of_recording)
                    position_xs.append(position.position_x_pixels)
                    position_ys.append(position.position_y_pixels)
                    synced_times.append(position.synced_time)
                    hds.append(position.hd)

                field_data_to_combine['rate_map'] = rate_maps
                field_data_to_combine['recording_length'] = length_recording
                field_data_to_combine['position_x_pixels'] = position_xs
                field_data_to_combine['position_y_pixels'] = position_ys
                field_data_to_combine['synced_time'] = synced_times
                field_data_to_combine['hd'] = hds
                field_data_combined = field_data_combined.append(field_data_to_combine)
                print(field_data_combined.head())
    field_data_combined.to_pickle(output_path)
    return field_data_combined


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_mouse(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    field_data['unique_id'] = unique_id
    unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    accepted_fields['unique_id'] = unique_id
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_rat(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    unique_cell_id = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['unique_id'] = unique_id
    field_data['unique_cell_id'] = unique_cell_id
    if 'Session ID' in accepted_fields:
        unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    else:
        unique_id = accepted_fields['SessionID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)

    accepted_fields['unique_id'] = unique_id
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# add cell type tp rat data frame
def add_cell_types_to_data_frame(field_data):
    cell_type = []
    for index, field in field_data.iterrows():
        if field.hd_score >= 0.5 and field.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif field.hd_score >= 0.5:
            cell_type.append('hd')
        elif field.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    field_data['cell type'] = cell_type

    return field_data


# get head-direction hist from bins of field
def get_hd_in_field_spikes(rate_map_indices, spatial_data, prm):
    hd_in_field_hist = np.zeros((len(rate_map_indices), 40))
    for index, bin_in_field in enumerate(rate_map_indices):
        inside_bin = PostSorting.open_field_head_direction.get_indices_for_bin(bin_in_field, spatial_data, prm)
        hd = inside_bin.hd.values + 180
        hd_hist = np.histogram(hd, bins=40, range=(0, 360))[0]
        # hd_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd)
        hd_in_field_hist[index] = hd_hist
    return hd_in_field_hist


def get_rate_map_values_for_bins(rate_map_indices, rate_map):
    rates = np.zeros((len(rate_map_indices), 1))
    for index, bin_in_field in enumerate(rate_map_indices):
        rate = rate_map[bin_in_field[0], bin_in_field[1]]
        rates[index] = rate
    return rates


def get_rate_map_values_for_bins_raw(rate_map_indices, spatial_data, spike_data, prm):
    rates = np.zeros((len(rate_map_indices), 1))
    for index, bin_in_field in enumerate(rate_map_indices):
        inside_bin = PostSorting.open_field_head_direction.get_indices_for_bin(bin_in_field, spatial_data, prm)
        inside_bin_spikes = PostSorting.open_field_head_direction.get_indices_for_bin(bin_in_field, spike_data, prm)

        if len(inside_bin) == 0:
            rate = 0
        else:
            rate = len(inside_bin_spikes.hd) / len(inside_bin.hd)
        rates[index] = rate
    return rates


def get_estimated_hd(field):
    spatial_data_field = pd.DataFrame()
    spatial_data_field['x'] = field.position_x_pixels
    spatial_data_field['y'] = field.position_y_pixels
    spatial_data_field['hd'] = field.hd
    spatial_data_field['synced_time'] = field.synced_time

    spike_data_field = pd.DataFrame()
    spike_data_field['x'] = field.position_x_spikes
    spike_data_field['y'] = field.position_y_spikes
    spike_data_field['hd'] = field.hd_in_field_spikes
    spike_data_field['synced_time'] = field.spike_times

    rate_map_indices = field.indices_rate_map
    hd_in_field_histograms = get_hd_in_field_spikes(rate_map_indices, spatial_data_field, prm)
    rates_for_bins = get_rate_map_values_for_bins_raw(rate_map_indices, spatial_data_field, spike_data_field, prm)
    weighed_hists = hd_in_field_histograms * rates_for_bins
    weighed_hist_sum = np.sum(weighed_hists, axis=0)
    # weighed_hist_sum_smooth = PostSorting.open_field_head_direction.get_rolling_sum(weighed_hist_sum, window=23) / 23
    return weighed_hist_sum


def get_estimated_hd_shuffled(field):
    spatial_data_field = pd.DataFrame()
    spatial_data_field['x'] = field.position_x_pixels[0]
    spatial_data_field['y'] = field.position_y_pixels[0]
    spatial_data_field['hd'] = field.hd[0]
    spatial_data_field['synced_time'] = field.synced_time[0]

    spike_data_field = pd.DataFrame()
    spike_data_field['x'] = field.position_x_spikes[0]
    spike_data_field['y'] = field.position_y_spikes[0]
    spike_data_field['hd'] = field.hd_in_field_spikes[0]
    spike_data_field['synced_time'] = field.spike_times[0]

    rate_map_indices = field.indices_rate_map[0]
    hd_in_field_histograms = get_hd_in_field_spikes(rate_map_indices, spatial_data_field, prm)
    rates_for_bins = get_rate_map_values_for_bins_raw(rate_map_indices, spatial_data_field, spike_data_field, prm)
    weighed_hists = hd_in_field_histograms * rates_for_bins
    weighed_hist_sum = np.sum(weighed_hists, axis=0)
    # weighed_hist_sum_smooth = PostSorting.open_field_head_direction.get_rolling_sum(weighed_hist_sum, window=23) / 23
    return weighed_hist_sum


# plot polar hd histograms without needing the whole df as an input
def plot_polar_hd_hist(hist_1, hist_2, cluster, save_path, color1='lime', color2='navy', title=''):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta = np.linspace(0, 2*np.pi, 41)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    ax.plot(theta[:-1], hist_1, color=color1, linewidth=2)
    ax.plot(theta[:-1], hist_2, color=color2, linewidth=2)
    plt.title(title)
    # ax.plot(theta[:-1], hist_2 * (max(hist_1) / max(hist_2)), color='navy', linewidth=2)
    plt.tight_layout()
    plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
    # plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
    plt.close()


def plot_real_vs_estimated(field, norm_hist_real, estimate, animal, ratio, color='red'):
    plt.cla()
    fig, ax = plt.subplots()
    ax.plot(norm_hist_real, color=color, alpha=0.4, linewidth=5)
    ax.plot(estimate, color='black', alpha=0.4, linewidth=5)
    plt.title('hd ' + str(round(field.hd_score, 1)) + ' grid ' + str(round(field.grid_score, 1)) + ' ratio ' + str(round(ratio, 4)))
    # legend = plt.legend()
    # legend.get_frame().set_facecolor('none')
    plt.savefig(local_path + animal + field.session_id + str(field.cluster_id) + str(field.field_id) + 'estimated_hd_rate_vs_real.png')
    plt.close()

    save_path = local_path + animal + field.session_id + str(field.cluster_id) + str(field.field_id) + 'estimated_hd_rate_vs_real'
    plot_polar_hd_hist(norm_hist_real, estimate, field.cluster_id, save_path, color1=color, color2='black', title='')


# generate more random colors if necessary
def generate_colors(number_of_firing_fields):
    colors = [[0, 1, 0], [1, 0.6, 0.3], [0, 1, 1], [1, 0, 1], [0.7, 0.3, 1], [0.6, 0.5, 0.4], [0.6, 0, 0]]  # green, orange, cyan, pink, purple, grey, dark red
    if number_of_firing_fields > len(colors):
        for i in range(number_of_firing_fields):
            colors.append(plot_utility.generate_new_color(colors, pastel_factor=0.9))
    return colors


def calculate_ratio(observed, predicted):
    small_number = 0.00001
    ratio = np.nanmean(np.abs(np.log((small_number + observed) / (small_number + predicted))))
    # print(ratio)
    return ratio


def plot_histograms_of_ratios(animal, field_data):
    grid_cells = field_data['cell type'] == 'grid'
    conjunctive_cells = field_data['cell type'] == 'conjunctive'
    accepted_fields = field_data.accepted_field == True

    ratio_scores_grid = field_data[grid_cells & accepted_fields].ratio_measure
    ratio_scores_conjunctive = field_data[conjunctive_cells & accepted_fields].ratio_measure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if 'ratio_measure_shuffle' in field_data:
        shuffled_scores_grid = field_data[grid_cells & accepted_fields].ratio_measure_shuffle
        shuffled_scores_grid_flat = [item for sublist in shuffled_scores_grid for item in sublist]
        plt.hist(shuffled_scores_grid_flat, color='gray', alpha=0.4)

    ax.set_yscale('log')
    plt.hist(ratio_scores_grid, color='navy', alpha=0.6)
    plt.hist(ratio_scores_conjunctive, color='red', alpha=0.6)

    plt.savefig(local_path + animal + '_ratio_measure_estimate.png')
    plt.close()


def distributive_hypothesis_analysis_observed(field_data, animal, output_path):
    print('analyzing observed ' + animal)
    if 'ratio_measure' in field_data:
        plot_histograms_of_ratios(animal, field_data)
        return field_data
    colors = generate_colors(20)
    ratios = []
    for index, field in field_data.iterrows():
        weighed_hist_sum = get_estimated_hd(field)
        # hd_session_real_hist = PostSorting.open_field_head_direction.get_hd_histogram(field.hd_in_field_session)
        hd_session_real_hist = np.histogram(field.hd_in_field_session * 180 / np.pi, bins=40, range=(0, 360))[0]
        estimate = weighed_hist_sum / hd_session_real_hist
        # hd_spikes_real_hist = PostSorting.open_field_head_direction.get_hd_histogram(field.hd_in_field_spikes)
        hd_spikes_real_hist = np.histogram(field.hd_in_field_spikes * 180 / np.pi, bins=40, range=(0, 360))[0]
        norm_hist_real = np.nan_to_num(hd_spikes_real_hist / hd_session_real_hist)
        ratio = calculate_ratio(norm_hist_real, estimate)
        ratios.append(ratio)
        plot_real_vs_estimated(field, norm_hist_real, estimate, animal, ratio, colors[field.field_id])
    field_data['ratio_measure'] = np.array(ratios)
    field_data.to_pickle(output_path)
    plot_histograms_of_ratios(animal, field_data)
    return field_data


def get_random_indices_for_shuffle(field, number_of_times_to_shuffle):
    number_of_spikes_in_field = field['number_of_spikes_in_field']
    time_spent_in_field = field['time_spent_in_field']
    shuffle_indices = np.random.randint(0, time_spent_in_field, size=(number_of_times_to_shuffle, number_of_spikes_in_field))
    return shuffle_indices


def distributive_hypothesis_analysis_shuffled(field_data, animal, output_path, number_of_shuffles=100):
    print('analyzing shuffled ' + animal)
    ratios_all = []
    if 'ratio_measure_shuffle' in field_data:
        plot_histograms_of_ratios(animal, field_data)
        return field_data

    for index, field in field_data.iterrows():
        ratios = []
        shuffle_indices = get_random_indices_for_shuffle(field, number_of_shuffles)
        for shuffle in range(number_of_shuffles):
            shuffled_data = pd.DataFrame()
            shuffled_data['index_shuffle'] = [0]
            shuffled_data['hd'] = [list(field['hd'])]
            shuffled_data['position_x_pixels'] = [list(field['position_x_pixels'])]
            shuffled_data['position_y_pixels'] = [list(field['position_y_pixels'])]
            shuffled_data['synced_time'] = [list(field['synced_time'])]

            shuffled_data['position_x_spikes'] = [list(field['position_x_session'][shuffle_indices[shuffle]])]
            shuffled_data['position_y_spikes'] = [list(field['position_y_session'][shuffle_indices[shuffle]])]
            shuffled_data['hd_in_field_spikes'] = [list(field['hd_in_field_session'][shuffle_indices[shuffle]])]
            shuffled_data['spike_times'] = [list(np.array(field['times_session'])[shuffle_indices[shuffle]])]
            shuffled_data['indices_rate_map'] = [field.indices_rate_map]

            weighed_hist_sum = get_estimated_hd_shuffled(shuffled_data)
            hd_session_real_hist = np.histogram(field.hd_in_field_session * 180 / np.pi, bins=40, range=(0, 360))[0]
            estimate = weighed_hist_sum / hd_session_real_hist
            hd_spikes_real_hist = np.histogram(np.array(shuffled_data.hd_in_field_spikes[0]) * 180 / np.pi, bins=40, range=(0, 360))[0]
            norm_hist_real = np.nan_to_num(hd_spikes_real_hist / hd_session_real_hist)
            ratio = calculate_ratio(norm_hist_real, estimate)
            ratios.append(ratio)

        ratios_all.append(ratios)

    field_data['ratio_measure_shuffle'] = ratios_all
    plot_histograms_of_ratios(animal, field_data)
    field_data.to_pickle(output_path)
    return field_data


def process_data(animal):
    if animal == 'mouse':
        output_path = local_path + 'mouse_data.pkl'
        server_path = server_path_mouse
        spike_sorter = '/MountainSort'
        prm.set_pixel_ratio(440)
        accepted_fields = pd.read_excel(local_path + 'list_of_accepted_fields.xlsx')
    else:
        server_path = server_path_rat
        spike_sorter = ''
        output_path = local_path + 'rat_data.pkl'
        prm.set_pixel_ratio(100)
        accepted_fields = pd.read_excel(local_path + 'included_fields_detector2_sargolini.xlsx')
    field_data = load_field_data(output_path, server_path, spike_sorter, animal)
    field_data = add_cell_types_to_data_frame(field_data)
    if animal == 'mouse':
        field_data = tag_accepted_fields_mouse(field_data, accepted_fields)
    if animal == 'rat':
        field_data = tag_accepted_fields_rat(field_data, accepted_fields)

    good_fields = field_data.accepted_field == True

    field_data = distributive_hypothesis_analysis_observed(field_data[good_fields], animal, output_path)
    field_data = distributive_hypothesis_analysis_shuffled(field_data[good_fields], animal, output_path)



def main():
    process_data('mouse')
    process_data('rat')


if __name__ == '__main__':
    main()