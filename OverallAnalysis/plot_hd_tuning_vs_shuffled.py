import data_frame_utility
import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_cell_analysis
import OverallAnalysis.compare_shuffled_from_first_and_second_halves_fields
import OverallAnalysis.false_positives
import pandas as pd
import PostSorting.parameters
import plot_utility

import scipy
import scipy.stats


local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/plot_hd_tuning_vs_shuffled/'

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


def plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, path, animal, shuffle_type='occupancy'):
    counter = 0
    for index, cell in spatial_firing.iterrows():
        mean = np.append(cell['shuffled_means'], cell['shuffled_means'][0])
        percentile_95 = np.append(cell['error_bar_95'], cell['error_bar_95'][0])
        percentile_5 = np.append(cell['error_bar_5'], cell['error_bar_5'][0])
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        max_rate = np.round(cell.hd_histogram_real_data_hz.max(), 2)
        x_pos = np.linspace(0, 2*np.pi, shuffled_histograms_hz.shape[1] + 1)

        significant_bins_to_mark = np.where(cell.p_values_corrected_bars_bh < 0.05)  # indices
        significant_bins_to_mark = x_pos[significant_bins_to_mark[0]]
        y_value_markers = [max_rate + 0.5] * len(significant_bins_to_mark)

        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        ax.fill_between(x_pos, mean - percentile_5, percentile_95 + mean, color='grey', alpha=0.4)
        ax.plot(x_pos, mean, color='grey', linewidth=5, alpha=0.7)
        observed_data = np.append(cell.hd_histogram_real_data_hz, cell.hd_histogram_real_data_hz[0])
        ax.plot(x_pos, observed_data, color='navy', linewidth=5)
        plt.title('\n' + str(max_rate) + ' Hz', fontsize=20, y=1.08)
        if (cell.p_values_corrected_bars_bh < 0.05).sum() > 0:
            ax.scatter(significant_bins_to_mark, y_value_markers, c='red',  marker='*', zorder=3, s=100)
        plt.subplots_adjust(top=0.85)
        plt.savefig(analysis_path + animal + '_' + shuffle_type + '/' + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile_polar_' + str(cell.percentile_value) + '.png')
        plt.close()
        counter += 1


def get_number_of_directional_cells(cells, tag='grid'):
    percentiles_no_correction = []
    percentiles_correction = []
    for index, cell in cells.iterrows():
        percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled, cell.number_of_different_bins)
        percentiles_no_correction.append(percentile)

        percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled_corrected_p, cell.number_of_different_bins_bh)
        percentiles_correction.append(percentile)

    cells['percentile_value'] = percentiles_correction
    print(tag)
    print('Number of fields: ' + str(len(cells)))
    print('Number of directional cells [without correction]: ')
    print(np.sum(np.array(percentiles_no_correction) > 95))
    cells['directional_no_correction'] = np.array(percentiles_no_correction) > 95

    print('Number of directional cells [with BH correction]: ')
    print(np.sum(np.array(percentiles_correction) > 95))
    cells['directional_correction'] = np.array(percentiles_correction) > 95
    cells.to_pickle(local_path + tag + 'cells.pkl')


def plot_hd_vs_shuffled():
    mouse_df_path = analysis_path + 'all_mice_df.pkl'
    mouse_df = pd.read_pickle(mouse_df_path)
    df = tag_false_positives(mouse_df)
    good_cells = df.false_positive == False
    df_good_cells = df[good_cells]
    df = add_cell_types_to_data_frame(df_good_cells)
    grid_cells = df['cell type'] == 'grid'
    df_grid = df[grid_cells]
    print('mouse')
    get_number_of_directional_cells(df_grid, tag='grid')
    plot_bar_chart_for_cells_percentile_error_bar(df_grid, analysis_path, 'mouse', shuffle_type='distributive')


def main():
    plot_hd_vs_shuffled()


if __name__ == '__main__':
    main()