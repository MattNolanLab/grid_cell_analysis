import glob
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import plot_utility
import PostSorting.make_plots
import PostSorting.speed
import OverallAnalysis.folder_path_settings
import os
import OverallAnalysis.false_positives
import OverallAnalysis.shuffle_cell_analysis


local_path_mouse = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/all_mice_df.pkl'
local_path_rat = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/all_rats_df.pkl'
local_path_simulated = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/all_simulated_df.pkl'
path_to_data = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/'
save_output_path = OverallAnalysis.folder_path_settings.get_local_path() + '/speed/'

server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()


def add_speed_score_to_spatial_firing(output_path, server_path, animal, video_sampling, ephys_sample, spike_sorter='', df_path='/DataFrames'):
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
            position_data = pd.read_pickle(position_data_path)
            if 'grid_score' in spatial_firing:
                if animal == 'rat':
                    spatial_firing = spatial_firing[['session_id', 'cell_id', 'cluster_id', 'firing_times',
                                                    'number_of_spikes', 'hd', 'speed', 'mean_firing_rate', 'grid_score', 'hd_score']].copy()
                if animal == 'mouse':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'firing_times',
                                                     'number_of_spikes', 'hd', 'speed', 'mean_firing_rate', 'grid_score', 'hd_score']].copy()
                if animal == 'simulated':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'firing_times',
                                                    'hd', 'hd_spike_histogram', 'speed', 'max_firing_rate_hd', 'grid_score', 'hd_score']].copy()

                spatial_firing = PostSorting.speed.calculate_speed_score(position_data, spatial_firing, gauss_sd=250, sampling_rate_conversion=ephys_sample)

                save_path = save_output_path + animal + 'grid_speed_scatter_plots/'
                if os.path.exists(save_path) is False:
                    os.makedirs(save_path)
                PostSorting.speed.plot_speed_vs_firing_rate_grid(position_data, spatial_firing, ephys_sample, video_sampling, save_path)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)
    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


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


def plot_speed_dependence(spatial_firing, animal, tag, color='navy'):
    # plt.hist(spatial_firing.speed_score, alpha=0.5, normed=True, color='gray')
    plt.cla()
    fig, ax = plt.subplots()
    ax = plot_utility.format_bar_chart(ax, 'Speed score', 'Number of ' + tag + ' cells')
    plt.hist(spatial_firing.speed_score, alpha=0.8, color=color)
    plt.xlim(-1, 1)
    plt.savefig(save_output_path + animal + '_' + tag + '_cell_speed_scores.png')
    plt.close()


def analyze_all_mouse_grid_cells(spatial_firing):
    spatial_firing = OverallAnalysis.shuffle_cell_analysis.tag_false_positives(spatial_firing)
    spatial_firing = add_cell_types_to_data_frame(spatial_firing)
    grid_cells = spatial_firing['cell type'] == 'grid'
    good_cell = spatial_firing.false_positive == False
    print('number of cells ' + str(len(spatial_firing[good_cell].speed_score)))
    median_speed_score = np.median(spatial_firing[good_cell].speed_score)
    print('[mouse] median speed score for all cells: ' + str(median_speed_score))
    print(np.std(spatial_firing[good_cell].speed_score))
    median_speed_score_grid = np.median(spatial_firing[grid_cells & good_cell].speed_score)
    print('number of grid cells ' + str(len(spatial_firing[good_cell & grid_cells].speed_score)))
    print('median speed score for grid cells: ' + str(median_speed_score_grid))
    print('proportion of significant speed scores:')
    print((spatial_firing[grid_cells & good_cell].speed_score_p_values < 0.05).sum() / len(
        spatial_firing[good_cell & grid_cells].speed_score))

    print('Number of cells with grid score < 0.1:')
    print(str((spatial_firing[good_cell & grid_cells].speed_score < 0.1).sum()))

    print('range of speed scores for grid cells: ' + str(spatial_firing[grid_cells & good_cell].speed_score.min()) + '-' + str(spatial_firing[grid_cells & good_cell].speed_score.max()))
    print(np.std(spatial_firing[grid_cells & good_cell].speed_score))
    plot_speed_dependence(spatial_firing[grid_cells & good_cell], 'mouse', 'grid')
    plot_speed_dependence(spatial_firing[good_cell], 'mouse', 'all', color='gray')


def analyze_all_rat_grid_cells(spatial_firing):
    print('----------------------------------------------------------------------------------------------------')
    spatial_firing = add_speed_score_to_spatial_firing(local_path_rat, server_path_rat, 'rat', 50, 1, spike_sorter='', df_path='/DataFrames')
    spatial_firing['false_positive'] = False
    spatial_firing = add_cell_types_to_data_frame(spatial_firing)
    grid_cells = spatial_firing['cell type'] == 'grid'
    good_cell = spatial_firing.false_positive == False
    median_speed_score = np.median(spatial_firing[good_cell].speed_score)
    print('number of cells ' + str(len(spatial_firing[good_cell].speed_score)))
    print('[rat] median speed score for all cells: ' + str(median_speed_score))
    print('range of speed scores for grid cells: ' + str(spatial_firing[grid_cells & good_cell].speed_score.min()) + '-' + str(spatial_firing[grid_cells & good_cell].speed_score.max()))
    print(np.std(spatial_firing[good_cell].speed_score))
    median_speed_score_grid = np.median(spatial_firing[grid_cells & good_cell].speed_score)
    print('number of grid cells ' + str(len(spatial_firing[good_cell & grid_cells].speed_score)))
    print('median speed score for grid cells: ' + str(median_speed_score_grid))
    print(np.std(spatial_firing[grid_cells & good_cell].speed_score))

    print('Number of cells with grid score < 0.1:')
    print(str((spatial_firing[good_cell & grid_cells].speed_score < 0.1).sum()))

    print('proportion of significant speed scores:')
    print((spatial_firing[grid_cells & good_cell].speed_score_p_values < 0.05).sum() / len(spatial_firing[good_cell & grid_cells].speed_score))
    plot_speed_dependence(spatial_firing[grid_cells & good_cell], 'rat', 'grid')
    plot_speed_dependence(spatial_firing[good_cell], 'rat', 'all', color='gray')


def process_data():
    accepted_fields = pd.read_excel(path_to_data + 'list_of_accepted_fields.xlsx')
    # mouse_speed_scores = pd.read_pickle(path_to_data + 'mouse_speed_scores.pkl')
    spatial_firing = add_speed_score_to_spatial_firing(local_path_mouse, server_path_mouse, 'mouse', 30, 30000, spike_sorter='/MountainSort', df_path='/DataFrames')
    # spatial_firing = mouse_speed_scores[mouse_speed_scores['session_id'].isin(spatial_firing.session_id)]
    print('all mouse grid cells')
    analyze_all_mouse_grid_cells(spatial_firing)
    has_fields = spatial_firing[spatial_firing.session_id.isin(accepted_fields['Session ID'])]
    print('included in field analysis')
    analyze_all_mouse_grid_cells(has_fields)

    accepted_fields = pd.read_excel(path_to_data + 'included_fields_detector2_sargolini.xlsx')
    spatial_firing = add_speed_score_to_spatial_firing(local_path_rat, server_path_rat, 'rat', 50, 1, spike_sorter='', df_path='/DataFrames')
    print('all rat grid cells')
    analyze_all_rat_grid_cells(spatial_firing)
    has_fields = spatial_firing[spatial_firing.session_id.isin(accepted_fields['Session ID'])]
    print('included in field analysis')
    analyze_all_rat_grid_cells(has_fields)


def main():
    process_data()


if __name__ == '__main__':
    main()