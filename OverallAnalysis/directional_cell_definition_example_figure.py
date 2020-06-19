import numpy as np
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis_heading
import OverallAnalysis.shuffle_cell_analysis
import pandas as pd
import plot_utility
import PostSorting.open_field_heading_direction
import PostSorting.open_field_firing_maps
import matplotlib.pylab as plt
import scipy.stats
import os
import glob

local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/methods_directional_cell/'


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
    return cells


# plot shuffled vs shuffled
def plot_bar_chart_for_cells_percentile_error_bar_polar(spatial_firing, path, animal, shuffle_type='occupancy'):
    plt.cla()
    counter = 0
    for index, cell in spatial_firing.iterrows():
        mean = np.append(cell['shuffled_means'], cell['shuffled_means'][0])
        percentile_95 = np.append(cell['error_bar_95'], cell['error_bar_95'][0])
        percentile_5 = np.append(cell['error_bar_5'], cell['error_bar_5'][0])
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        max_rate = np.round(cell.hd_histogram_real_data_hz.max(), 2)
        x_pos = np.linspace(0, 2*np.pi, shuffled_histograms_hz.shape[1] + 1)
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        ax.fill_between(x_pos, mean - percentile_5, percentile_95 + mean, color='grey', alpha=0.4)
        ax.plot(x_pos, mean, color='grey', linewidth=5, alpha=0.7)
        observed_data = np.append(cell.shuffled_histograms_hz[0], cell.shuffled_histograms_hz[0][0])
        ax.plot(x_pos, observed_data, color='black', linewidth=5, alpha=0.9)
        plt.ylim(0, max_rate + 1.5)
        plt.title('\n' + str(max_rate) + ' Hz', fontsize=20, y=1.08)
        plt.subplots_adjust(top=0.85)
        plt.savefig(analysis_path + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile_polar_' + str(cell.percentile_value) + '_polar.png')
        plt.close()
        counter += 1


def plot_bar_chart_for_cells_percentile_error_bar_polar_observed(spatial_firing, path, animal, shuffle_type='occupancy'):
    plt.cla()
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
        y_value_markers = [max_rate + 1] * len(significant_bins_to_mark)

        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        ax.fill_between(x_pos, mean - percentile_5, percentile_95 + mean, color='grey', alpha=0.4)
        observed_data = np.append(cell.hd_histogram_real_data_hz, cell.hd_histogram_real_data_hz[0])
        ax.plot(x_pos, observed_data, color='navy', linewidth=7)
        plt.ylim(0, max_rate + 1.5)
        plt.title('\n' + str(max_rate) + ' Hz', fontsize=20, y=1.08)
        if (cell.p_values_corrected_bars_bh < 0.05).sum() > 0:
            ax.scatter(significant_bins_to_mark, y_value_markers, c='red', marker='*', zorder=3, s=100)
        plt.subplots_adjust(top=0.85)
        plt.savefig(analysis_path + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile_polar_' + str(cell.percentile_value) + '_polar_observed.png')
        plt.close()
        counter += 1


def plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, path, animal, shuffle_type='distributive'):
    counter = 0
    for index, cell in spatial_firing.iterrows():
        for shuffle_example in range(3):
            mean = cell['shuffled_means']
            percentile_95 = cell['error_bar_95']
            percentile_5 = cell['error_bar_5']
            shuffled_histograms_hz = cell['shuffled_histograms_hz']
            x_pos = np.arange(shuffled_histograms_hz.shape[1])
            fig, ax = plt.subplots()
            ax = OverallAnalysis.shuffle_cell_analysis.format_bar_chart(ax)
            ax.errorbar(x_pos, mean, yerr=[percentile_5, percentile_95], alpha=0.7, color='black', ecolor='grey', capsize=10, fmt='o', markersize=10)
            x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
            plt.xticks(x_pos, x_labels)
            plt.scatter(x_pos, cell.shuffled_histograms_hz[shuffle_example], marker='o', color='grey', s=40)
            plt.ylim(2, 6.5)
            plt.yticks([2, 3, 4, 5, 6])
            # plt.title('Number of spikes ' + str(cell.number_of_spikes))
            plt.savefig(analysis_path + 'shuffle_analysis_' + animal + '_' + shuffle_type + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile' + str(shuffle_example))
            plt.close()
            counter += 1


def plot_bar_chart_for_cells_percentile_error_bar_observed(spatial_firing, path, animal, shuffle_type='distributive'):
    for index, cell in spatial_firing.iterrows():
        mean = cell['shuffled_means']
        percentile_95 = cell['error_bar_95']
        percentile_5 = cell['error_bar_5']
        shuffled_histograms_hz = cell['shuffled_histograms_hz']
        max_rate = np.round(cell.hd_histogram_real_data_hz.max(), 2)
        x_pos = np.arange(shuffled_histograms_hz.shape[1])
        significant_bins_to_mark = np.where(cell.p_values_corrected_bars_bh < 0.05)  # indices
        significant_bins_to_mark = x_pos[significant_bins_to_mark[0]]
        y_value_markers = [max_rate + 0.5] * len(significant_bins_to_mark)
        fig, ax = plt.subplots()
        ax = OverallAnalysis.shuffle_cell_analysis.format_bar_chart(ax)
        ax.errorbar(x_pos, mean, yerr=[percentile_5, percentile_95], alpha=0.7, color='black', ecolor='grey', capsize=10, fmt='o', markersize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        plt.scatter(x_pos, cell.hd_histogram_real_data_hz, marker='o', color='navy', s=40)
        plt.ylim(2, 6.5)
        plt.yticks([2, 3, 4, 5, 6])
        if (cell.p_values_corrected_bars_bh < 0.05).sum() > 0:
            ax.scatter(significant_bins_to_mark, y_value_markers, c='red', marker='*', zorder=3, s=100)
        # plt.title('Number of spikes ' + str(cell.number_of_spikes))
        plt.savefig(analysis_path + 'shuffle_analysis_' + animal + '_' + shuffle_type + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile_observed')
        plt.close()


def plot_shuffled_number_of_bins_vs_observed(cell):
    # percentile = scipy.stats.percentileofscore(cell.number_of_different_bins_shuffled_corrected_p.iloc[0], cell.number_of_different_bins_bh.iloc[0])
    shuffled_distribution = cell.number_of_different_bins_shuffled_corrected_p.iloc[0]
    plt.cla()
    fig = plt.figure(figsize=(6, 3))
    plt.yticks([0, 500, 1000])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yticklabels([0, '', 1000])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.hist(shuffled_distribution, bins=range(20), color='gray')
    ax.axvline(x=cell.number_of_different_bins_bh.iloc[0], color='navy', linewidth=3)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    # plt.xscale('log')
    max_x_value = max(cell.number_of_different_bins_bh.iloc[0], shuffled_distribution.max())
    plt.xlim(0, max_x_value + 1)
    plt.ylabel('N (shuffles)', fontsize=24)
    plt.xlabel('N (significant bins)', fontsize=24)
    plt.tight_layout()
    plt.savefig(analysis_path + 'number_of_significant_bars_shuffled_vs_real_example.png')
    plt.close()


def make_example_plot():
    session_id = 'M12_2018-04-10_14-22-14_of'
    # load shuffled hd data
    spatial_firing = pd.read_pickle(analysis_path + 'all_mice_df.pkl')
    '''
    ['session_id', 'cluster_id', 'hd_score', 'position_x', 'position_y',
       'hd', 'firing_maps', 'number_of_spikes_in_fields', 'firing_times',
       'trajectory_hd', 'trajectory_x', 'trajectory_y', 'trajectory_times',
       'number_of_spikes', 'rate_map_autocorrelogram', 'grid_spacing',
       'field_size', 'grid_score', 'false_positive_id', 'false_positive',
       'shuffled_data', 'shuffled_means', 'shuffled_std',
       'hd_histogram_real_data_hz', 'time_spent_in_bins',
       'shuffled_histograms_hz', 'shuffled_percentile_threshold_95',
       'shuffled_percentile_threshold_5', 'error_bar_95', 'error_bar_5',
       'real_and_shuffled_data_differ_bin', 'number_of_different_bins',
       'number_of_different_bins_shuffled', 'percentile_of_observed_data',
       'shuffle_p_values', 'p_values_corrected_bars_bh',
       'p_values_corrected_bars_holm', 'number_of_different_bins_bh',
       'number_of_different_bins_holm',
       'number_of_different_bins_shuffled_corrected_p']
    
    '''
    example_session = spatial_firing.session_id == session_id
    example_cell = spatial_firing[example_session]
    example_cell = get_number_of_directional_cells(example_cell, tag='grid')
    plot_shuffled_number_of_bins_vs_observed(example_cell)
    plot_bar_chart_for_cells_percentile_error_bar(example_cell, '', 'mouse', shuffle_type='distributive')
    plot_bar_chart_for_cells_percentile_error_bar_observed(example_cell, '', 'mouse', shuffle_type='distributive')
    plot_bar_chart_for_cells_percentile_error_bar_polar(example_cell, '', 'mouse', shuffle_type='distributive')
    plot_bar_chart_for_cells_percentile_error_bar_polar_observed(example_cell, '', 'mouse', shuffle_type='distributive')


def main():
    make_example_plot()


if __name__ == '__main__':
    main()