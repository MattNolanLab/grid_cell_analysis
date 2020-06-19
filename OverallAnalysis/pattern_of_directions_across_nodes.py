import sys
print(sys.path)
import glob
from scipy.stats import circstd
import scipy.stats
import seaborn
import math_utility
import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots
import plot_utility
import PostSorting.compare_first_and_second_half
import PostSorting.parameters

from scipy.stats import linregress

prm = PostSorting.parameters.Parameters()
prm.set_sorter_name('MountainSort')


local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/pattern_of_directions/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()


def remove_zeros(first, second):
    zeros_in_first_indices = first == 0
    zeros_in_second_indices = second == 0
    combined = zeros_in_first_indices + zeros_in_second_indices
    first_out = first[~combined]
    second_out = second[~combined]
    return first_out, second_out


def load_field_data(output_path, server_path, spike_sorter, animal, df_path='/DataFrames'):
    if animal == 'mouse':
        ephys_sampling_rate = 30000
    elif animal == 'rat':
        ephys_sampling_rate = 1  # this is because the rat data is already in seconds
    else:
        ephys_sampling_rate = 1000  # simulated is in ms
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data
    field_data_combined = pd.DataFrame()
    loaded_session_count = 0
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + df_path + '/shuffled_fields_distributive.pkl'
        spatial_firing_path = recording_folder + spike_sorter + df_path + '/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + df_path + '/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)
            spatial_firing = pd.read_pickle(spatial_firing_path)
            position = pd.read_pickle(position_path)
            prm.set_file_path(recording_folder)
            # spatial_firing = PostSorting.compare_first_and_second_half.analyse_first_and_second_halves(prm, position, spatial_firing)
            if 'shuffled_data' in field_data:
                loaded_session_count += 1
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'position_x_spikes',
                                                    'position_y_spikes', 'position_x_session', 'position_y_session',
                                                    'field_histograms_hz', 'indices_rate_map', 'hd_in_field_spikes',
                                                    'hd_in_field_session', 'spike_times', 'times_session',
                                                    'time_spent_in_field', 'number_of_spikes_in_field']].copy()
                field_data_to_combine['normalized_hd_hist'] = field_data.hd_hist_spikes / field_data.hd_hist_session
                if 'hd_score' in field_data:
                    field_data_to_combine['hd_score'] = field_data.hd_score
                if 'grid_score' in field_data:
                    field_data_to_combine['grid_score'] = field_data.grid_score
                rate_maps = []
                length_recording = []
                length_of_recording = 0
                field_data_to_combine.session_id = field_data_to_combine.session_id + '_' + str(loaded_session_count)

                for cluster in range(len(field_data.cluster_id)):
                    rate_map = spatial_firing[field_data.cluster_id.iloc[cluster] == spatial_firing.cluster_id].firing_maps
                    rate_maps.append(rate_map)
                    length_of_recording = (position.synced_time.max() - position.synced_time.min())
                    length_recording.append(length_of_recording)

                field_data_to_combine['rate_map'] = rate_maps
                field_data_to_combine['recording_length'] = length_recording
                field_data_to_combine = add_histograms_for_half_recordings(field_data_to_combine, position, spatial_firing, length_of_recording, ephys_sampling_rate)
                field_data_combined = field_data_combined.append(field_data_to_combine)
                print(field_data_combined.head())
    field_data_combined.to_pickle(output_path)
    return field_data_combined


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_mouse(field_data, accepted_fields):
    if len(field_data.session_id.iloc[0]) > 20:
        # need to remove end of session id string
        split_df = field_data.session_id.str.split('_', expand=True)
        session_id = split_df[0] + '_' + split_df[1] + '_' + split_df[2] + '_' + split_df[3]
    else:
        session_id = field_data.session_id

    unique_id = session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)

    field_data['unique_id'] = unique_id
    unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    accepted_fields['unique_id'] = unique_id
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_rat(field_data, accepted_fields):
    session_id = field_data.session_id.str.split('_', expand=True)[0]
    unique_id = session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    unique_cell_id = session_id + '_' + field_data.cluster_id.apply(str)
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


def clean_data(coefs):
    flat_coefs = [item for sublist in coefs for item in sublist]
    return [x for x in flat_coefs if str(x) != 'nan']


def format_bar_chart(ax, x_label, y_label):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(x_label, fontsize=25)
    ax.set_ylabel(y_label, fontsize=25)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    return ax


def plot_pearson_coefs_of_field_hist(coefs_grid, coefs_conjunctive, animal, tag=''):
    grid_coefs = clean_data(coefs_grid)
    fig, ax = plt.subplots()
    ax = format_bar_chart(ax, 'r', 'Proportion')
    plt.axvline(x=0, linewidth=3, color='gray')
    plt.hist(grid_coefs, color='navy', alpha=0.7, normed=True)
    print(animal + ' ' + tag + 'median correlation coefs in between fields [grid cells]')
    coefs_grid = [x for x in coefs_grid if ~np.isnan(x)]
    print(str(np.median(coefs_grid)))
    print(str(np.std(coefs_grid)))

    if coefs_conjunctive is not None:
        conj_coefs = clean_data(coefs_conjunctive)
        plt.hist(conj_coefs, color='red', alpha=0.7, normed='True')
    plt.savefig(local_path + animal + tag + '_correlation_of_field_histograms.png')
    plt.close()

    fig, ax = plt.subplots()
    ax = format_bar_chart(ax, 'r', 'Proportion')
    plt.axvline(x=0, linewidth=3, color='red')
    plot_utility.plot_cumulative_histogram(grid_coefs, ax, color='navy')
    if coefs_conjunctive is not None:
        plot_utility.plot_cumulative_histogram(conj_coefs, ax, color='red')
        print(animal + ' ' + tag + 'median correlation coefs in between fields [conjunctive cells]')
        print(str(np.median(conj_coefs)))
        print(str(np.std(conj_coefs)))
    plt.savefig(local_path + animal + tag + '_correlation_of_field_histograms_cumulative.png')
    plt.close()


def plot_pearson_coefs_of_field_hist_centre_border(coefs_centre, coefs_border, animal, tag=''):
    centre_coefs = clean_data(coefs_centre)
    border_coefs = clean_data(coefs_border)
    fig, ax = plt.subplots()
    plt.xlim(-1, 1)
    ax = format_bar_chart(ax, 'r', 'Proportion')
    plt.axvline(x=0, linewidth=3, color='red')
    plt.hist(centre_coefs, color='black', alpha=0.7, normed=True)
    plt.hist(border_coefs, color='gray', alpha=0.4, normed=True)

    plt.savefig(local_path + animal + tag + '_correlation_of_field_histograms.png')
    plt.close()

    fig, ax = plt.subplots()
    plt.axvline(x=0, linewidth=3, color='red')
    plot_utility.plot_cumulative_histogram(centre_coefs, ax, color='black')
    plot_utility.plot_cumulative_histogram(border_coefs, ax, color='gray')
    plt.savefig(local_path + animal + tag + '_correlation_of_field_histograms_cumulative.png')
    plt.close()

    print(animal + '' + tag)
    print(str(len(centre_coefs)) + ' number of centre coefs')
    print(str(len(border_coefs)) + ' number of border coefs')


def remove_nans(field1, field2):
    not_nans_in_field1 = ~np.isnan(field1)
    not_nans_in_field2 = ~np.isnan(field2)
    field1 = field1[not_nans_in_field1 & not_nans_in_field2]
    field2 = field2[not_nans_in_field1 & not_nans_in_field2]
    return field1, field2


def compare_hd_histograms(field_data, type='cell'):
    if len(field_data) > 0:
        field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
        list_of_cells = np.unique(list(field_data.unique_cell_id))
        pearson_coefs_all = []
        pearson_coefs_avg = []
        for cell in range(len(list_of_cells)):
            cell_id = list_of_cells[cell]
            field_histograms = field_data.loc[field_data['unique_cell_id'] == cell_id].normalized_hd_hist
            pearson_coefs_cell = []
            for index1, field1 in enumerate(field_histograms):
                for index2, field2 in enumerate(field_histograms):
                    if index1 != index2:
                        field1_clean_z, field2_clean_z = remove_nans(field1, field2)
                        # field1_clean_z, field2_clean_z = remove_zeros(field1_clean, field2_clean)
                        if len(field1_clean_z) > 1:
                            pearson_coef = scipy.stats.pearsonr(field1_clean_z, field2_clean_z)[0]
                            pearson_coefs_cell.append(pearson_coef)
            pearson_coefs_all.append([pearson_coefs_cell])
            pearson_coefs_avg.append([np.mean(pearson_coefs_cell)])

        t, p = scipy.stats.wilcoxon([item for sublist in pearson_coefs_avg for item in sublist])
        coefs_list = np.asanyarray([item for sublist in pearson_coefs_avg for item in sublist])
        coefs_list = coefs_list[~np.isnan(coefs_list)]
        print('Wilcoxon p value for correlations in between fields is (for avg of cell) ' + str(p) + ' T is ' + str(t) + type)
        print('nedian for in between fields: ' + str(np.median(coefs_list)) + 'sd ' + str(np.std(coefs_list)))
        print('number of fields: ' + str(len(coefs_list)))
        return pearson_coefs_avg


def get_pearson_coefs_all(field_data):
    if len(field_data) > 0:
        field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
        list_of_cells = np.unique(list(field_data.unique_cell_id))
        pearson_coefs_all = []
        for cell in range(len(list_of_cells)):
            cell_id = list_of_cells[cell]
            field_histograms = field_data.loc[field_data['unique_cell_id'] == cell_id].normalized_hd_hist
            pearson_coefs_cell = []
            for index1, field1 in enumerate(field_histograms):
                for index2, field2 in enumerate(field_histograms):
                    if index1 < index2:
                        field1_clean_z, field2_clean_z = remove_nans(field1, field2)
                        # field1_clean_z, field2_clean_z = remove_zeros(field1_clean, field2_clean)
                        if len(field1_clean_z) > 1:
                            pearson_coef = scipy.stats.pearsonr(field1_clean_z, field2_clean_z)[0]
                            pearson_coefs_cell.append(pearson_coef)
            pearson_coefs_all.extend([pearson_coefs_cell])
        return pearson_coefs_all


def get_distances_all(field_data):
    if len(field_data) > 0:
        field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
        list_of_cells = np.unique(list(field_data.unique_cell_id))
        distances_all = []
        for cell in range(len(list_of_cells)):
            cell_id = list_of_cells[cell]
            indices_rate_map = field_data.loc[field_data['unique_cell_id'] == cell_id].indices_rate_map
            distances_cell = []
            for index1, field1 in enumerate(indices_rate_map):
                for index2, field2 in enumerate(indices_rate_map):
                    if index1 < index2:
                        # calculate distance between fields
                        x1 = int(np.median(np.unique(field1[:, 0])))
                        y1 = int(np.median(np.unique(field1[:, 1])))

                        x2 = int(np.median(np.unique(field2[:, 0])))
                        y2 = int(np.median(np.unique(field2[:, 1])))

                        distance = np.sqrt(np.square(np.abs(x2 - x1)) + np.square(np.abs(y2 - y1)))
                        distances_cell.append(distance)
            distances_all.extend([distances_cell])
        return distances_all


# calculate rotation angle (rotating one of the fields) where the correlation is highest between the fields
def calculate_highest_correlating_angle(field_hist1, field_hist2):
    max = -1  # to find the maximum this is initialized at the lowest possible value
    angle_of_max = None
    for rotation in range(len(field_hist1)):
        field1_clean, field2_clean = remove_nans(field_hist1, field_hist2)
        if len(field1_clean) > 1:
            pearson_coef = scipy.stats.pearsonr(field1_clean, field2_clean)[0]
            if pearson_coef > max:
                max = pearson_coef
                angle_of_max = rotation * 360 / len(field_hist1)
        field_hist1 = np.roll(field_hist1, 1)
    return angle_of_max


def get_highest_correlation_angles(field_data):
    if len(field_data) > 0:
        field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
        list_of_cells = np.unique(list(field_data.unique_cell_id))
        highest_correlation_angles = []
        for cell in range(len(list_of_cells)):
            cell_id = list_of_cells[cell]
            field_histograms = field_data.loc[field_data['unique_cell_id'] == cell_id].normalized_hd_hist
            angles_cell = []
            for index1, field1 in enumerate(field_histograms):
                for index2, field2 in enumerate(field_histograms):
                    if index1 < index2:
                        angle = calculate_highest_correlating_angle(field1, field2)
                        angles_cell.append(angle)
            highest_correlation_angles.extend([angles_cell])

        return highest_correlation_angles


def get_distance_vs_correlations(field_data, type='grid cells'):
    pearson_coefs_all = get_pearson_coefs_all(field_data)
    distances = get_distances_all(field_data)
    highest_correlation_angles = get_highest_correlation_angles(field_data)
    coefs_list = np.asanyarray([item for sublist in pearson_coefs_all for item in sublist])
    distances_list = np.asanyarray([item for sublist in distances for item in sublist])
    highest_correlation_angles_list = np.asanyarray([item for sublist in highest_correlation_angles for item in sublist])
    # corr_clean, dist_clean = remove_nans(coefs_list, distances_list)
    corr, p = scipy.stats.pearsonr(distances_list, coefs_list)
    print('number of fields: ' + str(len(coefs_list)))
    print('Correlation between distance between fields and correlation of field shape:')
    print(corr)
    print('p: ' + str(p))

    # corr_lin, p_lin = scipy.stats.pearsonr(distances_list, highest_correlation_angles_list)
    corr, p = math_utility.circ_corrcc(np.radians(highest_correlation_angles_list), distances_list)
    print('number of fields: ' + str(len(distances_list)))
    print('Correlation between distance between fields and highest correlating rotation angle:')
    print(corr)
    print('p: ' + str(p))
    return distances_list, coefs_list, highest_correlation_angles_list


def plot_distances_vs_field_correlations(distances, in_between_coefs, tag):
    plt.cla()
    f, ax = plt.subplots(figsize=(11, 9))
    distances *= 2.5
    plt.scatter(distances, in_between_coefs)
    ax.set_xlabel('Distance between fields (cm)', fontsize=30)
    # Pearson correlation between fields
    ax.set_ylabel('R', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    slope, intercept, r_value, p_value, std_err = linregress(distances, in_between_coefs)
    plt.plot(distances, distances * slope + intercept, 'r')
    print('distances vs field corr')
    print('R = ' + str(r_value) + ' p = ' + str(p_value))

    plt.savefig(local_path + 'distance_between_fields_vs_correlation' + tag + '.png')
    plt.close()


# plot distance between fields vs highest correlating rotation (one of the fields is rotated to find the highest corr)
def plot_distances_vs_most_correlating_angle(distances, highest_corr_angles, tag):
    plt.cla()
    distances *= 2.5 # convert to cm
    f, ax = plt.subplots(figsize=(11, 9))
    plt.scatter(distances, highest_corr_angles)
    ax.set_xlabel('Distance between fields(cm)', fontsize=30)
    ax.set_ylabel('Highest correlating rotation (deg)', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    slope, intercept, r_value, p_value, std_err = linregress(distances, highest_corr_angles)
    plt.plot(distances, distances * slope + intercept, 'r')
    print('distances vs most correlating angle')
    print('R = ' + str(r_value) + ' p = ' + str(p_value))

    plt.savefig(local_path + 'distance_between_fields_vs_highest_correlating_angle' + tag + '.png')
    plt.close()


def save_correlation_plot(corr, animal, cell_type, tag=''):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    plt.cla()
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    # cmap = seaborn.color_palette("RdBu_r", 10)
    cmap = seaborn.color_palette("coolwarm", 10)

    # Draw the heatmap with the mask and correct aspect ratio
    seaborn.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.savefig(local_path + animal + '_' + cell_type + tag + '_correlation_heatmap.png')
    plt.close()


def get_field_df_to_correlate(histograms):
    histograms_df = pd.DataFrame()
    field_number = 0
    for index, cell in histograms.iterrows():
        field_number += 1
        histograms_df[str(field_number)] = cell.normalized_hd_hist
    return histograms_df


def get_field_df_to_correlate_halves(histograms):
    histograms_df = pd.DataFrame()
    field_number = 0
    for index, cell in histograms.iterrows():
        field_number += 1
        histograms_df[str(field_number)] = cell.hd_hist_first_half

    for index, cell in histograms.iterrows():
        field_number += 1
        histograms_df[str(field_number + int(len(histograms)/2))] = cell.hd_hist_second_half
    return histograms_df


def plot_correlation_matrix(field_data, animal):
    grid_histograms = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score < 0.5) & (field_data.accepted_field == True)]
    grid_histograms_df = get_field_df_to_correlate(grid_histograms)
    # Compute the correlation matrix
    corr = grid_histograms_df.corr()
    save_correlation_plot(corr, animal, 'grid')

    grid_histograms_centre = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score < 0.5) & (field_data.accepted_field == True) & (field_data.border_field == False)]
    grid_histograms_df_centre = get_field_df_to_correlate(grid_histograms_centre)
    # Compute the correlation matrix
    corr = grid_histograms_df_centre.corr()
    save_correlation_plot(corr, animal, 'grid', 'centre_fields')

    grid_histograms_border = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score < 0.5) & (field_data.accepted_field == True) & (field_data.border_field == True)]
    grid_histograms_df_border = get_field_df_to_correlate(grid_histograms_border)
    # Compute the correlation matrix
    corr = grid_histograms_df_border.corr()
    save_correlation_plot(corr, animal, 'grid', 'border_fields')

    conjunctive_histograms = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score >= 0.5) & (field_data.accepted_field == True)]
    conjunctive_histograms_df = get_field_df_to_correlate(conjunctive_histograms)
    # Compute the correlation matrix
    corr = conjunctive_histograms_df.corr()
    save_correlation_plot(corr, animal, 'conjunctive')

    grid_histograms = field_data.loc[(field_data.grid_score >= 0.4) & (field_data.hd_score < 0.5) & (field_data.accepted_field == True)]
    grid_histograms_df_halves = get_field_df_to_correlate_halves(grid_histograms)
    # Compute the correlation matrix
    corr = grid_histograms_df_halves.corr()
    save_correlation_plot(corr, animal, 'grid_halves')


def plot_correlation_matrix_individual_cells(field_data, animal):
    field_data = field_data[field_data.accepted_field == True]
    field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
    list_of_cells = np.unique(list(field_data.unique_cell_id))
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        field_histograms = field_data.loc[field_data['unique_cell_id'] == cell_id]
        field_df = get_field_df_to_correlate(field_histograms)
        corr = field_df.corr()  # correlation matrix
        save_correlation_plot(corr, animal, '', tag=cell_id)

    field_data_centre = field_data[(field_data.accepted_field == True) & (field_data.border_field == False)]
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        field_histograms = field_data_centre.loc[field_data_centre['unique_cell_id'] == cell_id]
        field_df = get_field_df_to_correlate(field_histograms)
        corr = field_df.corr()  # correlation matrix
        save_correlation_plot(corr, animal, '', tag=cell_id + '_centre')

    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        field_histograms = field_data_centre.loc[field_data_centre['unique_cell_id'] == cell_id]
        field_df = get_field_df_to_correlate_halves(field_histograms)
        corr = field_df.corr()  # correlation matrix
        save_correlation_plot(corr, animal, '', tag=cell_id + '_halves')


# if it touches the border it's a border field
def tag_border_and_middle_fields(field_data):
    border_tag = []
    for index, field in field_data.iterrows():
        rate_map = field.rate_map.iloc[0]
        field_indices = field.indices_rate_map
        y_max = len(rate_map)
        x_max = len(rate_map[0])
        border = False
        if (field_indices[:, 0] < 1).sum() + (field_indices[:, 0] == x_max).sum() > 0:
            border = True
        if (field_indices[:, 1] < 1).sum() + (field_indices[:, 1] == y_max).sum() > 0:
            border = True
        border_tag.append(border)

    field_data['border_field'] = border_tag
    return field_data


# this is just here to test the analysis by plotting the data
def plot_half_spikes(spatial_firing, position, field, half_time, firing_times_cluster, sampling_rate_ephys):
    plt.cla()
    position_x = spatial_firing[field.cluster_id == spatial_firing.cluster_id].position_x
    position_y = spatial_firing[field.cluster_id == spatial_firing.cluster_id].position_y
    spike_x_first_half = np.array(position_x.values[0])[
        firing_times_cluster.values[0] < (half_time * sampling_rate_ephys)]
    spike_y_first_half = np.array(position_y.values[0])[
        firing_times_cluster.values[0] < (half_time * sampling_rate_ephys)]
    position_x_second_half = np.array(position_x.values[0])[
        firing_times_cluster.values[0] >= (half_time * sampling_rate_ephys)]
    position_y_second_half = np.array(position_y.values[0])[
        firing_times_cluster.values[0] >= (half_time * sampling_rate_ephys)]

    # firing events first half
    plt.scatter(spike_x_first_half, spike_y_first_half, color='red', s=50)
    # firing events second half
    plt.scatter(position_x_second_half, position_y_second_half, color='lime', s=50)
    # all session data from field
    plt.scatter(field.position_x_session / 440 * 100, field.position_y_session / 440 * 100, color='navy')

    firing_times_cluster_first_half = firing_times_cluster.values[0][
        firing_times_cluster.values[0] < (half_time * sampling_rate_ephys)]
    firing_times_cluster_second_half = firing_times_cluster.values[0][
        firing_times_cluster.values[0] >= (half_time * sampling_rate_ephys)]
    x_cluster = spatial_firing[field.cluster_id == spatial_firing.cluster_id].position_x
    y_cluster = spatial_firing[field.cluster_id == spatial_firing.cluster_id].position_y
    x_cluster_first_half = np.array(x_cluster.values[0])[
        firing_times_cluster.values[0] < (half_time * sampling_rate_ephys)]
    x_cluster_second_half = np.array(x_cluster.values[0])[
        firing_times_cluster.values[0] >= (half_time * sampling_rate_ephys)]
    y_cluster_first_half = np.array(y_cluster.values[0])[
        firing_times_cluster.values[0] < (half_time * sampling_rate_ephys)]
    y_cluster_second_half = np.array(y_cluster.values[0])[
        firing_times_cluster.values[0] >= (half_time * sampling_rate_ephys)]
    mask_firing_times_in_field_first = np.in1d(firing_times_cluster_first_half, field.spike_times)
    mask_firing_times_in_field_second = np.in1d(firing_times_cluster_second_half, field.spike_times)

    x_field_first_half_spikes = x_cluster_first_half[mask_firing_times_in_field_first]
    y_field_first_half_spikes = y_cluster_first_half[mask_firing_times_in_field_first]
    x_field_second_half_spikes = x_cluster_second_half[mask_firing_times_in_field_second]
    y_field_second_half_spikes = y_cluster_second_half[mask_firing_times_in_field_second]

    plt.scatter(x_field_first_half_spikes, y_field_first_half_spikes, color='yellow')
    plt.scatter(x_field_second_half_spikes, y_field_second_half_spikes, color='black')

    # get half of session data
    times_field_first_half = np.take(field.times_session, np.where(field.times_session < half_time))
    mask_times_in_field_first_half = np.in1d(position.synced_time, times_field_first_half)
    x_field_first_half = position.position_x[mask_times_in_field_first_half]
    y_field_first_half = position.position_y[mask_times_in_field_first_half]

    times_field_second_half = np.take(field.times_session, np.where(field.times_session >= half_time))
    mask_times_in_field_second_half = np.in1d(position.synced_time, times_field_second_half)
    x_field_second_half = position.position_x[mask_times_in_field_second_half]
    y_field_second_half = position.position_y[mask_times_in_field_second_half]

    plt.plot(x_field_first_half, y_field_first_half, color='red', alpha=0.5)
    plt.plot(x_field_second_half, y_field_second_half, color='lime', alpha=0.5)


def get_halves_for_spike_data(length_of_recording, spatial_firing, field, sampling_rate_ephys):
    half_time = length_of_recording / 2
    firing_times_cluster = spatial_firing[field.cluster_id == spatial_firing.cluster_id].firing_times
    firing_times_cluster_first_half = firing_times_cluster.values[0][
        firing_times_cluster.values[0] < (half_time * sampling_rate_ephys)]
    firing_times_cluster_second_half = firing_times_cluster.values[0][
        firing_times_cluster.values[0] >= (half_time * sampling_rate_ephys)]
    hd_cluster = spatial_firing[field.cluster_id == spatial_firing.cluster_id].hd
    hd_cluster_first_half = np.array(hd_cluster.values[0])[
        firing_times_cluster.values[0] < (half_time * sampling_rate_ephys)]
    hd_cluster_second_half = np.array(hd_cluster.values[0])[
        firing_times_cluster.values[0] >= (half_time * sampling_rate_ephys)]
    mask_firing_times_in_field_first = np.in1d(firing_times_cluster_first_half, field.spike_times)
    mask_firing_times_in_field_second = np.in1d(firing_times_cluster_second_half, field.spike_times)
    hd_field_first_half_spikes = hd_cluster_first_half[mask_firing_times_in_field_first]
    hd_field_first_half_spikes_rad = (hd_field_first_half_spikes + 180) * np.pi / 180
    hd_field_hist_first_spikes = PostSorting.open_field_head_direction.get_hd_histogram(hd_field_first_half_spikes_rad)
    hd_field_second_half_spikes = hd_cluster_second_half[mask_firing_times_in_field_second]
    hd_field_second_half_spikes_rad = (hd_field_second_half_spikes + 180) * np.pi / 180
    hd_field_hist_second_spikes = PostSorting.open_field_head_direction.get_hd_histogram(
        hd_field_second_half_spikes_rad)
    return hd_field_hist_first_spikes, hd_field_hist_second_spikes, hd_field_first_half_spikes, hd_field_second_half_spikes


def get_halves_for_session_data(position, field, length_of_recording):
    half_time = length_of_recording / 2
    times_field_first_half = np.take(field.times_session, np.where(field.times_session < half_time))
    mask_times_in_field_first_half = np.in1d(position.synced_time, times_field_first_half)
    hd_field_first_half = position.hd[mask_times_in_field_first_half]
    hd_field_first_half_rad = (hd_field_first_half + 180) * np.pi / 180
    hd_field_hist_first_session = PostSorting.open_field_head_direction.get_hd_histogram(hd_field_first_half_rad)

    times_field_second_half = np.take(field.times_session, np.where(field.times_session >= half_time))
    mask_times_in_field_second_half = np.in1d(position.synced_time, times_field_second_half)
    hd_field_second_half = position.hd[mask_times_in_field_second_half]
    hd_field_second_half_rad = (hd_field_second_half + 180) * np.pi / 180
    hd_field_hist_second_session = PostSorting.open_field_head_direction.get_hd_histogram(hd_field_second_half_rad)
    return hd_field_hist_first_session, hd_field_hist_second_session, hd_field_first_half, hd_field_second_half


def add_histograms_for_half_recordings(field_data, position, spatial_firing, length_of_recording, sampling_rate_ephys):
    first_halves = []
    hd_first_spikes = []
    hd_first_session = []
    second_halves = []
    hd_second_spikes = []
    hd_second_session = []
    # pearson_coefs = []
    # pearson_ps = []
    for field_index, field in field_data.iterrows():
        # get half of spike data
        hd_field_hist_first_spikes, hd_field_hist_second_spikes, hd_field_first_half_spikes, hd_field_second_half_spikes = get_halves_for_spike_data(length_of_recording, spatial_firing, field, sampling_rate_ephys)

        # get half of session data
        hd_field_hist_first_session, hd_field_hist_second_session, hd_field_first_half, hd_field_second_half = get_halves_for_session_data(position, field, length_of_recording)

        hd_hist_first_half = np.divide(hd_field_hist_first_spikes, hd_field_hist_first_session)
        hd_hist_second_half = np.divide(hd_field_hist_second_spikes, hd_field_hist_second_session)
        # pearson_coef, pearson_p = scipy.stats.pearsonr(hd_hist_first_half, hd_hist_second_half)
        first_halves.append(hd_hist_first_half)
        second_halves.append(hd_hist_second_half)
        # pearson_coefs.append(pearson_coef)
        # pearson_ps.append(pearson_p)
        hd_first_spikes.append(hd_field_first_half_spikes)
        hd_second_spikes.append(hd_field_second_half_spikes)
        hd_first_session.append(hd_field_first_half)
        hd_second_session.append(hd_field_second_half)

    field_data['hd_hist_first_half'] = first_halves
    field_data['hd_hist_second_half'] = second_halves
    # field_data['pearson_coef_halves'] = pearson_coefs
    # field_data['pearson_p_halves'] = pearson_ps
    field_data['hd_first_half_session'] = hd_first_session
    field_data['hd_second_half_session'] = hd_second_session
    field_data['hd_first_half_spikes'] = hd_first_spikes
    field_data['hd_second_half_spikes'] = hd_second_spikes

    return field_data


def get_correlation_values_in_between_fields(field_data):
    if 'unique_cell_id' not in field_data:
        field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
    list_of_cells = np.unique(list(field_data.unique_cell_id))
    # first_halves = field_data.hd_hist_first_half
    # second_halves = field_data.hd_hist_second_half
    correlation_values = []
    correlation_p = []
    count_f1 = 0
    count_f2 = 0
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        first_halves = field_data.loc[field_data['unique_cell_id'] == cell_id].hd_hist_first_half
        second_halves = field_data.loc[field_data['unique_cell_id'] == cell_id].hd_hist_second_half
        for index, field1 in enumerate(first_halves):
            for index2, field2 in enumerate(second_halves):
                if count_f1 != count_f2:
                    # field1_clean, field2_clean = remove_zeros(field1, field2)
                    field1_clean, field2_clean = remove_nans(field1, field2)
                    if len(field1_clean) > 1:   # if there is any data left after removing nans
                        pearson_coef, corr_p = scipy.stats.pearsonr(field1_clean, field2_clean)
                        correlation_values.append(pearson_coef)
                        correlation_p.append(corr_p)
                    else:
                        correlation_values.append(np.nan)
                        correlation_p.append(np.nan)

                count_f2 += 1
            count_f1 += 1

    correlation_values_in_between = np.array(correlation_values)
    return correlation_values_in_between, correlation_p


def get_correlation_values_within_fields(field_data):
    first_halves = field_data.hd_hist_first_half
    second_halves = field_data.hd_hist_second_half
    correlation_values = []
    correlation_p = []
    for field in range(len(field_data)):
        first = first_halves.iloc[field]
        second = second_halves.iloc[field]
        # first, second = remove_zeros(first, second)
        first, second = remove_nans(first, second)

        if len(first) > 1:
            pearson_coef, corr_p = scipy.stats.pearsonr(first, second)
            correlation_values.append(pearson_coef)
            correlation_p.append(corr_p)
        else:
            correlation_values.append(np.nan)
            correlation_p.append(np.nan)

    correlation_values_within = np.array(correlation_values)
    correlation_p = np.array(correlation_p)
    return correlation_values_within, correlation_p


def save_half_fields_as_csv(field_data, file_name):
    field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
    list_of_cells = np.unique(list(field_data.unique_cell_id))

    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        field_histograms_first_half = field_data.loc[field_data['unique_cell_id'] == cell_id].hd_hist_first_half
        field_histograms_second_half = field_data.loc[field_data['unique_cell_id'] == cell_id].hd_hist_second_half
        for index, field in enumerate(field_histograms_first_half):
            np.savetxt(local_path + '/data_for_r/' + file_name + '_' + cell_id + '_' + str(index) + '_first.csv', field, delimiter=',')
        for index, field in enumerate(field_histograms_second_half):
            np.savetxt(local_path + '/data_for_r/' + file_name + '_' + cell_id + '_' + str(index) + '_second.csv', field, delimiter=',')


def compare_within_field_with_other_fields(field_data, animal):
    if len(field_data) > 1:
        in_between_fields, correlation_p = get_correlation_values_in_between_fields(field_data)
        within_field_corr, correlation_p_within = get_correlation_values_within_fields(field_data)

        fig, ax = plt.subplots()
        plt.axvline(x=0, linewidth=3, color='red')
        ax = format_bar_chart(ax, 'r', 'Proportion')
        plot_utility.plot_cumulative_histogram(in_between_fields[~np.isnan(in_between_fields)], ax, color='navy')
        plot_utility.plot_cumulative_histogram(within_field_corr[~np.isnan(within_field_corr)], ax, color='gray')
        plt.xlim(-1, 1)
        plt.savefig(local_path + animal + 'half_session_correlations_cumulative2.png')
        plt.close()

        # plot only within field comparisons
        fig, ax = plt.subplots()
        plt.axvline(x=0, linewidth=3, color='red')
        ax = format_bar_chart(ax, 'r', 'Proportion')
        plot_utility.plot_cumulative_histogram(within_field_corr[~np.isnan(within_field_corr)], ax, color='gray')
        plt.xlim(-1, 1)
        plt.savefig(local_path + animal + 'half_session_correlations_cumulative_within_field_only.png')
        plt.close()
        t, p = scipy.stats.wilcoxon(in_between_fields)
        print('all fields included')
        print('Wilcoxon p value for correlations in between fields (all fields)' + str(p) + ' T is ' + str(t) + animal)
        t, p = scipy.stats.wilcoxon(within_field_corr)
        print('Wilcoxon p value for correlations within fields (all fields)' + str(p) + ' T is ' + str(t) + animal)
        print('median of in-between fields: ' + str(np.median(in_between_fields[~np.isnan(in_between_fields)])) + ' sd: ' + str(np.std(in_between_fields[~np.isnan(in_between_fields)])))
        print('median of within fields: ' + str(np.median(within_field_corr[~np.isnan(within_field_corr)])) + ' sd: ' + str(np.std(within_field_corr[~np.isnan(within_field_corr)])))


def save_corr_coef_in_csv(good_grid_coef, good_grid_cells_p, file_name):
    correlation_data = pd.DataFrame()
    correlation_data['R'] = good_grid_coef
    correlation_data['p'] = good_grid_cells_p
    correlation_data.to_csv(OverallAnalysis.folder_path_settings.get_local_path() + '/field_histogram_shapes/' + file_name + '.csv')


def compare_within_field_with_other_fields_stat(field_data, animal):
    correlation_values_in_between, correlation_p = get_correlation_values_in_between_fields(field_data)
    # save_corr_coef_in_csv(correlation_values_in_between, correlation_p, 'in_between_fields_all_' + animal)
    print('% of significant p values for in between field correlations:')
    print(sum(np.array(correlation_p) < 0.01) / len(correlation_p) * 100)
    within_field_corr, correlation_p_within = get_correlation_values_within_fields(field_data)
    # save_corr_coef_in_csv(within_field_corr, correlation_p_within, 'within_fields_all_' + animal)
    print('% of significant p values for within field correlations:')
    print('number of fields included: ' + str(len(correlation_p_within)))
    print(sum(correlation_p_within < 0.01) / len(correlation_p_within) * 100)
    stat, p = scipy.stats.ks_2samp(correlation_values_in_between, within_field_corr)
    print('Kolmogorov-Smirnov result to compare in between and within field correlations for ' + animal)
    print(stat)
    print(p)

    t, p = scipy.stats.wilcoxon(correlation_values_in_between)
    print('Wilcoxon p value for correlations in between fields is ' + str(p) + ' T is ' + str(t))

    t, p = scipy.stats.wilcoxon(within_field_corr)
    print('Wilcoxon p value for within field correlations is ' + str(p) + ' T is ' + str(t))


def compare_within_field_with_other_fields_correlating_fields(field_data, animal):
    if 'unique_cell_id' not in field_data:
        field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)
    # correlation_within, p = get_correlation_values_within_fields(field_data)
    list_of_cells = np.unique(list(field_data.unique_cell_id))
    correlation_values = []
    correlation_p = []
    count_f1 = 0
    count_f2 = 0
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        first_halves = field_data.loc[field_data['unique_cell_id'] == cell_id].hd_hist_first_half
        second_halves = field_data.loc[field_data['unique_cell_id'] == cell_id].hd_hist_second_half
        correlation_within, p_within = get_correlation_values_within_fields(field_data.loc[field_data['unique_cell_id'] == cell_id])
        for index, field1 in enumerate(first_halves):
            for index2, field2 in enumerate(second_halves):
                if count_f1 != count_f2:
                    if (correlation_within[index] >= 0.4) & (correlation_within[index2] >= 0.4):
                        # field_1_clean, field_2_clean = remove_zeros(field1, field2)
                        field_1_clean, field_2_clean = remove_nans(field1, field2)
                        if len(field_1_clean) > 1:
                            pearson_coef, corr_p = scipy.stats.pearsonr(field_1_clean, field_2_clean)
                            correlation_values.append(pearson_coef)
                            correlation_p.append(corr_p)
                        else:
                            correlation_values.append(np.nan)
                            correlation_p.append(np.nan)

                count_f2 += 1
            count_f1 += 1

    in_between_fields = np.array(correlation_values)
    in_between_fields_p = np.array(correlation_p)

    within_field, p_within_field = get_correlation_values_within_fields(field_data)
    save_corr_coef_in_csv(within_field, p_within_field, 'within_fields_correlating_only_' + animal)
    within_field = within_field[within_field >= 0.4]

    fig, ax = plt.subplots()
    plt.axvline(x=0, linewidth=3, color='red')
    ax = format_bar_chart(ax, 'r', 'Proportion')
    plot_utility.plot_cumulative_histogram(in_between_fields[~np.isnan(in_between_fields)], ax, color='navy')
    plot_utility.plot_cumulative_histogram(within_field[~np.isnan(within_field)], ax, color='gray')
    plt.xlim(-1, 1)
    plt.savefig(local_path + animal + 'half_session_correlations_internally_correlating_only_r04_cumulative.png')
    plt.close()

    stat, p = scipy.stats.ks_2samp(in_between_fields, within_field)
    print('for Pearson r >= 0.4')
    print('median of in-between fields: ' + str(np.median(in_between_fields)) + ' sd: ' + str(np.std(in_between_fields)))
    print('median of within fields: ' + str(np.median(within_field)) + ' sd: ' + str(np.std(within_field)))
    print('Kolmogorov-Smirnov result to compare in between and within field correlations for ' + animal)
    print(stat)
    print(p)
    print('number of fields ' + str(len(within_field)))

    print('% of coefficients with significant p for in between field correlations:')
    print(sum(in_between_fields_p < 0.01) / len(in_between_fields_p) * 100)
    save_corr_coef_in_csv(in_between_fields, in_between_fields_p, 'in_between_fields_correlating_only_' + animal)

    print('% of coefficients with significant p for within field correlations:')
    print(sum(p_within_field < 0.01) / len(p_within_field) * 100)
    t, p = scipy.stats.wilcoxon(in_between_fields)
    print('Wilcoxon p value for correlations in between fields (all, R>0.4)' + str(p) + ' T is ' + str(t) + animal)
    t, p = scipy.stats.wilcoxon(within_field)
    print('Wilcoxon p value for correlations within fields (all, R>0.4)' + str(p) + ' T is ' + str(t) + animal)


def plot_half_fields(field_data, animal):
    correlation, p = get_correlation_values_within_fields(field_data)
    field_num = 0
    for index, field in field_data.iterrows():
        plt.cla()
        fig, ax = plt.subplots()
        corr = str(round(correlation[field_num], 4))
        title = ('r= ' + corr)
        save_path = local_path + '/first_vs_second_fields/' + animal + field.session_id + str(field.cluster_id) + str(field.field_id)
        PostSorting.open_field_make_plots.plot_polar_hd_hist(field.hd_hist_first_half, field.hd_hist_second_half, field.cluster_id, save_path, color1='lime', color2='navy', title=title)

        plt.close()
        field_num += 1


def remove_nans_from_both(first, second):
    nans_in_first_indices = np.isnan(first)
    nans_in_second_indices = np.isnan(second)
    nans_combined = nans_in_first_indices + nans_in_second_indices
    first_out = first[~nans_combined]
    second_out = second[~nans_combined]
    return first_out, second_out


def get_server_path_and_load_accepted_fields(animal, tag):
    print('I will analyze data from this animal: ' + animal + ' ' + tag)
    animal_path = local_path + 'field_data_modes_' + animal + tag + '.pkl'
    if animal == 'mouse':
        server_path = server_path_mouse
        accepted_fields = pd.read_excel(local_path + 'list_of_accepted_fields.xlsx')
        field_data = load_field_data(animal_path, server_path, '/MountainSort', animal)
    elif animal == 'rat':
        server_path = server_path_rat
        accepted_fields = pd.read_excel(local_path + 'included_fields_detector2_sargolini.xlsx')
        field_data = load_field_data(animal_path, server_path, '', animal)

    else:
        if tag == 'ventral_narrow':
            server_path = server_path_simulated + '/' + tag + '/'
            field_data = load_field_data(animal_path, server_path, '', animal, df_path='')
            accepted_fields = True
        else:
            server_path = server_path_simulated + '/' + tag + '/'
            field_data = load_field_data(animal_path, server_path, '', animal, df_path='')
            accepted_fields = True

    return field_data, accepted_fields


def process_circular_data(animal, tag=''):
    print('*************************' + animal + tag + '***************************')
    field_data, accepted_fields = get_server_path_and_load_accepted_fields(animal, tag)
    if animal == 'mouse':
        field_data = tag_accepted_fields_mouse(field_data, accepted_fields)
    elif animal == 'rat':
        field_data = tag_accepted_fields_rat(field_data, accepted_fields)
    else:
        field_data['accepted_field'] = True

    field_data = add_cell_types_to_data_frame(field_data)
    field_data = tag_border_and_middle_fields(field_data)

    all_accepted_grid_cells_df = field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid')]
    save_amount_of_time_and_number_of_spikes_in_fields_csv(all_accepted_grid_cells_df, animal + '_' + tag)

    distances, in_between_coefs, highest_corr_angles = get_distance_vs_correlations(all_accepted_grid_cells_df, type='grid cells ' + animal)
    plot_distances_vs_field_correlations(distances, in_between_coefs, tag='grid_cells_' + animal)
    plot_distances_vs_most_correlating_angle(distances, highest_corr_angles, tag='grid_cells_' + animal)

    grid_cell_pearson = compare_hd_histograms(all_accepted_grid_cells_df, type='grid cells ' + animal)

    centre_fields_only_df = field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid') & (field_data.border_field == False)]
    grid_pearson_centre = compare_hd_histograms(centre_fields_only_df, type='grid cells, centre ' + animal)

    border_fields_only_df = field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid') & (field_data.border_field == True)]
    grid_pearson_border = compare_hd_histograms(border_fields_only_df, type='grid cells, border ' + animal)

    conjunctive_cells_df = field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'conjunctive')]
    # compare_within_field_with_other_fields_correlating_fields(all_accepted_grid_cells_df, 'grid_' + animal + tag)
    compare_within_field_with_other_fields(all_accepted_grid_cells_df, 'grid_' + animal + tag)

    conjunctive_cell_pearson = compare_hd_histograms(conjunctive_cells_df, type='conjunctive cells ' + animal)
    conjunctive_pearson_centre = compare_hd_histograms(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'conjunctive') & (field_data.border_field == False)])
    compare_within_field_with_other_fields(conjunctive_cells_df, 'conj_' + animal + tag)
    plot_pearson_coefs_of_field_hist(grid_cell_pearson, conjunctive_cell_pearson, animal + tag)
    plot_pearson_coefs_of_field_hist(grid_pearson_centre, conjunctive_pearson_centre, animal, tag='_centre')

    compare_within_field_with_other_fields_stat(field_data[(field_data.accepted_field == True) & (field_data['cell type'] == 'grid')], 'grid_' + animal + tag)
    plot_pearson_coefs_of_field_hist_centre_border(grid_pearson_centre, grid_pearson_border, 'animal', tag='_centre_vs_border')
    # plot_correlation_matrix(field_data, animal)
    # plot_correlation_matrix_individual_cells(field_data, animal)
    # plot_half_fields(field_data, 'mouse')


def compare_correlations_from_different_experiments():
    pearson_grid_simulated_narrow = pd.read_csv(local_path + 'grid_cell_pearson_avg_for_cellsimulatedventral_narrow.csv', header=None)
    pearson_grid_simulated_control = pd.read_csv(local_path + 'grid_cell_pearson_avg_for_cellsimulatedcontrol_narrow.csv', header=None)

    print('**************************')
    print('I will now compare results from different experiments.')
    stat, p = scipy.stats.ks_2samp(pearson_grid_simulated_narrow[0].values, pearson_grid_simulated_control[0].values)
    print('Kolmogorov-Smirnov result to compare simulted control and ventral grid cell within field pearson ' + str(stat) + ' ' + str(p))

    between_field_ventral_narrow_04 = pd.read_csv(local_path + 'in_between_fields_correlating_only_grid_simulatedventral_narrow.csv').R.values
    between_field_control_narrow_04 = pd.read_csv(local_path + 'in_between_fields_correlating_only_grid_simulatedcontrol_narrow.csv').R.values
    between_field_ventral_narrow = pd.read_csv(local_path + 'in_between_fields_all_grid_simulatedventral_narrow.csv').R.values
    between_field_control_narrow = pd.read_csv(local_path + 'in_between_fields_all_grid_simulatedcontrol_narrow.csv').R.values

    within_field_ventral_narrow_04 = pd.read_csv(local_path + 'within_fields_correlating_only_grid_simulatedventral_narrow.csv').R.values
    within_field_control_narrow_04 = pd.read_csv(local_path + 'within_fields_correlating_only_grid_simulatedcontrol_narrow.csv').R.values
    within_field_ventral_narrow = pd.read_csv(local_path + 'within_fields_all_grid_simulatedventral_narrow.csv').R.values
    within_field_control_narrow = pd.read_csv(local_path + 'within_fields_all_grid_simulatedcontrol_narrow.csv').R.values

    stat, p = scipy.stats.ks_2samp(between_field_ventral_narrow_04, between_field_control_narrow_04)
    print('Kolmogorov-Smirnov result between_field_ventral_narrow_04 vs control ' + str(stat) + ' ' + str(p))

    stat, p = scipy.stats.ks_2samp(between_field_ventral_narrow, between_field_control_narrow)
    print('Kolmogorov-Smirnov result between_field_ventral_narrow vs control ' + str(stat) + ' ' + str(p))

    stat, p = scipy.stats.ks_2samp(within_field_ventral_narrow_04, within_field_control_narrow_04)
    print('Kolmogorov-Smirnov result within_field_ventral_narrow_04 vs control ' + str(stat) + ' ' + str(p))

    stat, p = scipy.stats.ks_2samp(within_field_ventral_narrow, within_field_control_narrow)
    print('Kolmogorov-Smirnov result within_field_ventral_narrow vs control' + str(stat) + ' ' + str(p))

    print('****************')
    stat, p = scipy.stats.ks_2samp(between_field_control_narrow, within_field_control_narrow)
    print('Kolmogorov-Smirnov result within_field vs in between, control ' + str(stat) + ' ' + str(p))

    stat, p = scipy.stats.ks_2samp(between_field_ventral_narrow, within_field_ventral_narrow)
    print('Kolmogorov-Smirnov result within_field vs in between, ventral ' + str(stat) + ' ' + str(p))


def save_amount_of_time_and_number_of_spikes_in_fields_csv(field_data, tag):
    if not os.path.isdir(local_path + 'session_length_and_spikes'):
        os.mkdir(local_path + 'session_length_and_spikes')

    df_to_save = field_data[['session_id', 'cluster_id', 'field_id', 'time_spent_in_field', 'number_of_spikes_in_field']]
    df_to_save.to_csv(local_path + 'session_length_and_spikes/' + tag + '.csv')


def main():
    process_circular_data('mouse')
    process_circular_data('rat')
    # process_circular_data('simulated', 'ventral_narrow')
    # process_circular_data('simulated', 'control_narrow')
    # compare_correlations_from_different_experiments()


if __name__ == '__main__':
    main()

