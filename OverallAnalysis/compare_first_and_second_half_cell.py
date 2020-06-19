import glob
import pandas as pd
import plot_utility
import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.false_positives
import OverallAnalysis.folder_path_settings
import OverallAnalysis.analyze_field_correlations
import os
import scipy.stats

import rpy2.robjects as ro
from rpy2.robjects.packages import importr


local_path_mouse = OverallAnalysis.folder_path_settings.get_local_path() + '/correlation_cell/all_mice_df.pkl'
local_path_rat = OverallAnalysis.folder_path_settings.get_local_path() + '/correlation_cell/all_rats_df.pkl'
local_path_simulated = OverallAnalysis.folder_path_settings.get_local_path() + '/correlation_cell/all_simulated_df.pkl'
path_to_data = OverallAnalysis.folder_path_settings.get_local_path() + '/correlation_cell/'
save_output_path = OverallAnalysis.folder_path_settings.get_local_path() + '/correlation_cell/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()


def add_cell_types_to_data_frame(cells):
    cell_type = []
    for index, field in cells.iterrows():
        if field.hd_score >= 0.5 and field.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif field.hd_score >= 0.5:
            cell_type.append('hd')
        elif field.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    cells['cell type'] = cell_type

    return cells


def tag_false_positives(all_cells, animal):
    if animal == 'mouse':
        false_positives_path = path_to_data + 'false_positives_all.txt'
        list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
        all_cells = add_combined_id_to_df(all_cells)
        all_cells['false_positive'] = all_cells['false_positive_id'].isin(list_of_false_positives)
    else:
        all_cells['false_positive'] = np.full(len(all_cells), False)
    return all_cells


def load_spatial_firing(output_path, server_path, animal, spike_sorter='', df_path='/DataFrames'):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + df_path + '/spatial_firing.pkl'
        position_data_path = recording_folder + spike_sorter + df_path + '/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            if 'hd_correlation_first_vs_second_half' in spatial_firing:
                if animal == 'rat':
                    spatial_firing = spatial_firing[['session_id', 'cell_id', 'cluster_id', 'firing_times',
                                                    'number_of_spikes', 'hd', 'speed', 'mean_firing_rate',
                                                     'hd_spike_histogram', 'max_firing_rate_hd', 'preferred_HD',
                                                     'grid_spacing', 'field_size', 'grid_score', 'hd_score', 'firing_fields']].copy()
                if animal == 'mouse':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'firing_times',
                                                     'number_of_spikes', 'hd', 'speed', 'mean_firing_rate',
                                                     'hd_spike_histogram', 'max_firing_rate_hd', 'preferred_HD',
                                                     'grid_spacing', 'field_size', 'grid_score', 'hd_score',
                                                     'firing_fields', 'hd_correlation_first_vs_second_half', 'hd_correlation_first_vs_second_half_p']].copy()
                if animal == 'simulated':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'firing_times',
                                                    'hd', 'hd_spike_histogram', 'max_firing_rate_hd', 'preferred_HD',
                                                     'grid_spacing', 'field_size', 'grid_score', 'hd_score', 'firing_fields']].copy()
                    downsample = True

                spatial_firing_data = spatial_firing_data.append(spatial_firing)

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def add_combined_id_to_df(df_all_mice):
    animal_ids = [session_id.split('_')[0] for session_id in df_all_mice.session_id.values]
    dates = [session_id.split('_')[1] for session_id in df_all_mice.session_id.values]
    tetrode = df_all_mice.tetrode.values
    cluster = df_all_mice.cluster_id.values

    combined_ids = []
    for cell in range(len(df_all_mice)):
        id = animal_ids[cell] + '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    df_all_mice['false_positive_id'] = combined_ids
    return df_all_mice


def save_corr_coef_in_csv(good_grid_coef, good_grid_cells_p):
    correlation_data = pd.DataFrame()
    correlation_data['R'] = good_grid_coef
    correlation_data['p'] = good_grid_cells_p
    correlation_data.to_csv(OverallAnalysis.folder_path_settings.get_local_path() + '/correlation_cell/whole_cell_correlations.csv')


def correlation_between_first_and_second_halves_of_session(df_all_animals, animal='mouse'):
    good_cluster = df_all_animals.false_positive == False
    grid_cell = df_all_animals['cell type'] == 'grid'

    is_hd_cell = df_all_animals.hd_score >= 0.5
    print('number of grid: ' + str(len(df_all_animals[grid_cell])))
    print('number of conj cells: ' + str(len(df_all_animals[grid_cell & is_hd_cell])))

    print('mean and sd pearson r of correlation between first and second half for grid cells')
    print(df_all_animals.hd_correlation_first_vs_second_half[good_cluster & grid_cell].mean())
    print(df_all_animals.hd_correlation_first_vs_second_half[good_cluster & grid_cell].std())

    print('% of significant correlation values for grid cells: ')
    good_grid_coef = df_all_animals.hd_correlation_first_vs_second_half[good_cluster & grid_cell]
    good_grid_cells_p = df_all_animals.hd_correlation_first_vs_second_half_p[good_cluster & grid_cell]
    number_of_significant_ps = (good_grid_cells_p < 0.01).sum()
    all_ps = len(good_grid_cells_p)
    proportion = number_of_significant_ps / all_ps * 100
    print(proportion)
    save_corr_coef_in_csv(good_grid_coef, good_grid_cells_p)
    t, p = scipy.stats.wilcoxon(df_all_animals.hd_correlation_first_vs_second_half[good_cluster & grid_cell])
    print('Wilcoxon p value is ' + str(p) + ' T is ' + str(t))

    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_animals.hd_correlation_first_vs_second_half[good_cluster & grid_cell], save_output_path + 'correlation_hd_session_' + animal + '.png', y_axis_label='Cumulative probability')


def process_data(animal):
    print('-------------------------------------------------------------')
    if animal == 'mouse':
        spike_sorter = '/MountainSort'
        local_path_animal = local_path_mouse
        server_path_animal = server_path_mouse
        df_path = '/DataFrames'
    elif animal == 'rat':
        spike_sorter = ''
        local_path_animal = local_path_rat
        server_path_animal = server_path_rat
        df_path = '/DataFrames'
    else:
        spike_sorter = ''
        local_path_animal = local_path_simulated
        server_path_animal = server_path_simulated
        df_path = ''

    all_cells = load_spatial_firing(local_path_animal, server_path_animal, animal, spike_sorter, df_path=df_path)
    all_cells = tag_false_positives(all_cells, animal)
    all_cells = add_cell_types_to_data_frame(all_cells)
    if animal == 'mouse':
        correlation_between_first_and_second_halves_of_session(all_cells)


def main():
    process_data('mouse')
    process_data('rat')
    process_data('simulated')


if __name__ == '__main__':
    main()