import data_frame_utility
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis
import OverallAnalysis.compare_shuffled_from_first_and_second_halves_fields
import OverallAnalysis.false_positives
import pandas as pd
import PostSorting.parameters

import scipy


local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/basic_info_included_data/'

prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)
prm.set_sampling_rate(30000)


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
    spatial_firing['animal'] = animal_ids

    dates = [session_id.split('_')[1] for session_id in spatial_firing.session_id.values]

    cluster = spatial_firing.cluster_id.values
    combined_ids = []
    for cell in range(len(spatial_firing)):
        id = animal_ids[cell] + '-' + dates[cell] + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    spatial_firing['false_positive_id'] = combined_ids
    return spatial_firing


def tag_false_positives(spatial_firing):
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(analysis_path + 'false_positives_all.txt')
    spatial_firing = add_combined_id_to_df(spatial_firing)
    spatial_firing['false_positive'] = spatial_firing['false_positive_id'].isin(list_of_false_positives)
    return spatial_firing


def get_time_spent_and_num_spikes(df_grid):
    print('Avg length of recording:')
    print((df_grid.number_of_spikes / df_grid.mean_firing_rate / 60).mean())
    print('sd')
    print((df_grid.number_of_spikes / df_grid.mean_firing_rate / 60).std())

    print('Avg number of spikes:')
    print(df_grid.number_of_spikes.mean())
    print('sd')
    print(df_grid.number_of_spikes.std())


def print_basic_info(df, animal):
    if animal == 'mouse':
        df = tag_false_positives(df)
    else:
        df['false_positive'] = False

    good_cells = df.false_positive == False
    df_good_cells = df[good_cells]
    df = add_cell_types_to_data_frame(df_good_cells)
    grid_cells = df['cell type'] == 'grid'
    df_grid = df[grid_cells]
    conj_cells = df['cell type'] == 'conjunctive'
    df_conj = df[conj_cells]
    print('Number of grid cells:')
    print(len(df_grid))
    print('Number of conjunctive cells:')
    print(len(df_conj))

    get_time_spent_and_num_spikes(df_grid)


    animals_with_grid_cells = df_grid.animal.unique()
    animals_with_conj_cells = df_conj.animal.unique()

    animals_with_grid_or_conj = np.unique(np.concatenate((animals_with_grid_cells, animals_with_conj_cells), axis=0))
    print('Number of animals:')
    print(len(animals_with_grid_or_conj))

    included_cells = df_good_cells[df_good_cells.animal.isin(animals_with_grid_or_conj)]
    print('Number of included cells:')
    print(len(included_cells))

    print('Number_of_included sessions:')
    print(len(included_cells.session_id.unique()))

    print('Number of recording days per animal:')
    print(included_cells.groupby('animal').session_id.nunique())
    print('mean')
    print(included_cells.groupby('animal').session_id.nunique().mean())
    print('std')
    print(included_cells.groupby('animal').session_id.nunique().std())
    print('min')
    print(included_cells.groupby('animal').session_id.nunique().min())
    print('max')
    print(included_cells.groupby('animal').session_id.nunique().max())


def main():
    mouse_df_path = analysis_path + 'all_mice_df.pkl'
    mouse_df = pd.read_pickle(mouse_df_path)
    rat_df_path = analysis_path + 'all_rats_df.pkl'
    rat_df = pd.read_pickle(rat_df_path)

    print('mouse')
    print_basic_info(mouse_df, 'mouse')
    print('rat')
    print_basic_info(rat_df, 'rat')


if __name__ == '__main__':
    main()