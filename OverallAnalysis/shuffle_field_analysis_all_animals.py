import numpy as np
import OverallAnalysis.folder_path_settings
import pandas as pd
import plot_utility
import matplotlib.pylab as plt
import scipy.stats
import os
import glob

analysis_path = OverallAnalysis.folder_path_settings.get_local_path() + '/shuffled_analysis/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()

local_path_to_shuffled_field_data_mice = analysis_path + 'shuffled_field_data_all_mice.pkl'
local_path_to_shuffled_field_data_rats = analysis_path + 'shuffled_field_data_all_rats.pkl'
# local_path_to_shuffled_field_data_simulated = analysis_path + 'shuffled_field_data_all_simulated.pkl'


# loads shuffle analysis results for field data
def load_data_frame_field_data(output_path, server_path, spike_sorter, df_path='/DataFrames', shuffle_type='occupancy'):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data

    else:
        field_data_combined = pd.DataFrame()
        for recording_folder in glob.glob(server_path + '*'):
            os.path.isdir(recording_folder)
            if shuffle_type == 'occupancy':
                data_frame_path = recording_folder + spike_sorter + df_path + '/shuffled_fields.pkl'
            else:
                data_frame_path = recording_folder + spike_sorter + df_path + '/shuffled_fields_distributive.pkl'
            if os.path.exists(data_frame_path):
                print('I found a field data frame.')
                field_data = pd.read_pickle(data_frame_path)
                if 'field_id' in field_data:
                    field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'indices_rate_map',
                                                        'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
                                                        'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
                                                        'times_session', 'time_spent_in_field', 'position_x_session',
                                                        'position_y_session', 'hd_in_field_session', 'hd_hist_session',
                                                        'hd_histogram_real_data', 'time_spent_in_bins',
                                                        'field_histograms_hz', 'hd_score', 'grid_score', 'shuffled_means', 'shuffled_std',
                                                        'real_and_shuffled_data_differ_bin', 'number_of_different_bins',
                                                        'number_of_different_bins_shuffled', 'number_of_different_bins_bh',
                                                        'number_of_different_bins_holm', 'number_of_different_bins_shuffled_corrected_p']].copy()

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
    field_data['unique_cell_id'] = field_data.session_id + '_' + field_data.cluster_id.apply(str)
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
    field_data['unique_cell_id'] = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


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
    ax.set_xlim(0, 20)
    ax.set_xlabel('Rejected bars / field', size=30)
    ax.set_ylabel('Proportion', size=30)
    plt.savefig(analysis_path + 'distribution_of_rejects_' + shuffle_type + animal + '.png', bbox_inches="tight")
    plt.close()


def plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_field_data, animal='mouse', shuffle_type='occupancy'):
    number_of_rejects = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects:
        flat_shuffled.extend(field)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Rejected bars / field', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_xlim(0, 20)
    plt.savefig(analysis_path + '/distribution_of_rejects_shuffled' + shuffle_type + animal + '.png', bbox_inches="tight")
    plt.close()


def make_combined_plot_of_distributions(shuffled_field_data, tag='grid', shuffle_type='occupancy'):
    tail, percentile_95, percentile_99 = find_tail_of_shuffled_distribution_of_rejects(shuffled_field_data)

    number_of_rejects_shuffled = shuffled_field_data.number_of_different_bins_shuffled
    flat_shuffled = []
    for field in number_of_rejects_shuffled:
        flat_shuffled.extend(field)
    fig, ax = plt.subplots()
    plt.hist(flat_shuffled, normed=True, color='black', alpha=0.5)

    number_of_rejects_real = shuffled_field_data.number_of_different_bins
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
    ax.set_xlabel('Rejected bars / field', size=30)
    ax.set_ylabel('Proportion', size=30)
    ax.set_xlim(0, 20)
    plt.savefig(analysis_path + 'distribution_of_rejects_combined_all_' + shuffle_type + tag + '.png', bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    plt.yticks([0, 1])
    plt.ylim(0, 1.01)
    ax = plot_utility.format_bar_chart(ax, 'Pearson correlation coef.', 'Cumulative probability')
    ax.set_xlim(0, 20)
    values, base = np.histogram(flat_shuffled, bins=40)
    # evaluate the cumulative
    cumulative = np.cumsum(values / len(flat_shuffled))
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='gray', linewidth=5)

    values, base = np.histogram(number_of_rejects_real, bins=40)
    # evaluate the cumulative
    cumulative = np.cumsum(values / len(number_of_rejects_real))
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)

    # plt.axvline(x=tail, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(x=percentile_95, color='red', alpha=0.5, linestyle='dashed')
    # plt.axvline(x=percentile_99, color='red', alpha=0.5, linestyle='dashed')

    ax.set_xlabel('Rejected bars / field', size=30)
    ax.set_ylabel('Cumulative probability', size=30)
    plt.savefig(analysis_path + 'distribution_of_rejects_combined_all_' + shuffle_type + tag + '_cumulative.png', bbox_inches="tight")
    plt.close()


def plot_number_of_significant_p_values(field_data, type='bh', shuffle_type='occupancy'):
    if type == 'bh':
        number_of_significant_p_values = field_data.number_of_different_bins_bh
    else:
        number_of_significant_p_values = field_data.number_of_different_bins_holm

    fig, ax = plt.subplots()
    plt.hist(number_of_significant_p_values, normed='True', color='navy', alpha=0.5)
    flat_shuffled = []
    for field in field_data.number_of_different_bins_shuffled_corrected_p:
        flat_shuffled.extend(field)
    plt.hist(flat_shuffled, normed='True', color='gray', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 20)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Significant bars / field', size=20)
    ax.set_ylabel('Proportion', size=20)
    ax.set_ylim(0, 0.2)
    ax.set_xlim(0, 20)
    plt.savefig(analysis_path + 'distribution_of_rejects_significant_p_ ' + shuffle_type + type + '.png', bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    plt.yticks([0, 1])
    plt.ylim(0, 1.01)
    values, base = np.histogram(flat_shuffled, bins=40)
    # evaluate the cumulative
    cumulative = np.cumsum(values / len(flat_shuffled))
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='gray', linewidth=5)

    values, base = np.histogram(number_of_significant_p_values, bins=40)
    # evaluate the cumulative
    cumulative = np.cumsum(values / len(number_of_significant_p_values))
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)
    ax.set_xlim(0, 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlabel('Significant bars / field', size=25)
    ax.set_ylabel('Cumulative probability', size=25)
    plt.savefig(analysis_path + 'distribution_of_rejects_significant_p_' + shuffle_type + type + '_cumulative.png', bbox_inches="tight")
    plt.close()


def compare_distributions(x, y):
    # stat, p = scipy.stats.mannwhitneyu(x, y)
    stat, p = scipy.stats.ranksums(x, y)
    print('p value and test statistic for MW U test:')
    print(p)
    print(stat)
    return p, stat


def compare_shuffled_to_real_data_mw_test(field_data, analysis_type='bh', shuffle_type='occupancy'):
    num_bins = 20
    if analysis_type == 'bh':
        flat_shuffled = []
        for field in field_data.number_of_different_bins_shuffled_corrected_p:
            flat_shuffled.extend(field)
        p_bh, stat_bh = compare_distributions(field_data.number_of_different_bins_bh, flat_shuffled)
        print(shuffle_type)
        print('p value for comparing shuffled distribution to B-H corrected p values: ' + str(p_bh))
        print('stat value for comparing shuffled distribution to B-H corrected p values: ' + str(stat_bh))
        number_of_significant_bins = field_data.number_of_different_bins_bh.sum()
        total_number_of_bins = len(field_data.number_of_different_bins_bh) * num_bins
        print(str(number_of_significant_bins) + ' out of ' + str(total_number_of_bins) + ' are significant')
        print(str(np.mean(field_data.number_of_different_bins_bh)) + ' number of bins per cell +/- ' + str(np.std(field_data.number_of_different_bins_bh)) + ' SD')
        print('shuffled: ')
        print(str(np.mean(flat_shuffled)) + ' number of bins per cell +/- ' + str(np.std(flat_shuffled)) + ' SD')
        return p_bh

    if analysis_type == 'percentile':
        flat_shuffled = []
        for field in field_data.number_of_different_bins_shuffled:
            flat_shuffled.extend(field)
        p_percentile, stat_percentile = compare_distributions(field_data.number_of_different_bins, flat_shuffled)
        print('p value for comparing shuffled distribution to percentile thresholded p values: ' + str(p_percentile))
        print('stat value for comparing shuffled distribution to percentile thresholded p values: ' + str(stat_percentile))
        number_of_significant_bins = field_data.number_of_different_bins.sum()
        total_number_of_bins = len(field_data.number_of_different_bins) * num_bins
        print(str(number_of_significant_bins) + ' out of ' + str(total_number_of_bins) + ' are different')
        print(str(np.mean(field_data.number_of_different_bins)) + ' number of bins per cell +/- ' + str(np.std(field_data.number_of_different_bins)) + ' SD')

        return p_percentile


def plot_distributions_for_fields(shuffled_field_data, tag='grid', animal='mouse', shuffle_type='occupancy'):
    plot_histogram_of_number_of_rejected_bars(shuffled_field_data, animal + tag, shuffle_type=shuffle_type)
    plot_histogram_of_number_of_rejected_bars_shuffled(shuffled_field_data, animal + tag, shuffle_type=shuffle_type)
    plot_number_of_significant_p_values(shuffled_field_data, type='bh_' + tag + '_' + animal, shuffle_type=shuffle_type)
    plot_number_of_significant_p_values(shuffled_field_data, type='holm_' + tag + '_' + animal, shuffle_type=shuffle_type)
    make_combined_plot_of_distributions(shuffled_field_data, tag=tag + '_' + animal, shuffle_type=shuffle_type)


def get_percentage_of_grid_cells_with_directional_nodes(fields):
    percentage_of_directional_no_corr = []
    percentage_of_directional_corr = []
    cell_ids = fields.unique_cell_id.unique()
    for cell in range(len(cell_ids)):
        fields_of_cell = fields[fields.unique_cell_id == cell_ids[cell]]
        number_of_directional_fields_no_correction = np.sum(fields_of_cell.directional_no_correction)
        percentage = number_of_directional_fields_no_correction / len(fields_of_cell) * 100
        percentage_of_directional_no_corr.append(percentage)
        number_of_directional_fields_correction = np.sum(fields_of_cell.directional_correction)
        percentage = number_of_directional_fields_correction / len(fields_of_cell) * 100
        percentage_of_directional_corr.append(percentage)

    print('Total number of cells: ' + str(len(cell_ids)))
    print('avg % of directional fields in grid cells no correction: ' + str(np.mean(percentage_of_directional_no_corr)))
    print(np.std(percentage_of_directional_no_corr))
    print('avg % of directional fields in grid cells BH correction: ' + str(np.mean(percentage_of_directional_corr)))
    print(np.std(percentage_of_directional_corr))


def plot_shuffled_number_of_bins_vs_observed(cells):
    for index, cell in cells.iterrows():
        shuffled_distribution = cell.number_of_different_bins_shuffled_corrected_p
        plt.cla()
        fig = plt.figure(figsize=(6, 3))
        plt.yticks([0, 1000])
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(shuffled_distribution, bins=range(20), color='gray')
        ax.axvline(x=cell.number_of_different_bins_bh, color='navy', linewidth=3)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        max_x_value = max(cell.number_of_different_bins_bh, shuffled_distribution.max())
        plt.xlim(0, max_x_value + 1)
        # plt.xscale('log')
        plt.ylabel('N', fontsize=24)
        plt.xlabel('Number of significant bins', fontsize=24)
        plt.tight_layout()
        plt.savefig(analysis_path + 'percentile/' + cell.session_id + str(cell.cluster_id) + str(cell.field_id) + 'number_of_significant_bars_shuffled_vs_real_' + str(cell.directional_percentile)  + '.png')
        plt.close()


def get_number_of_directional_fields(fields, tag='grid'):
    percentiles_no_correction = []
    percentiles_correction = []
    for index, field in fields.iterrows():
        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled, field.number_of_different_bins)
        percentiles_no_correction.append(percentile)

        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled_corrected_p, field.number_of_different_bins_bh)
        percentiles_correction.append(percentile)

    print(tag)
    print('Number of fields: ' + str(len(fields)))
    print('Number of directional fields [without correction]: ')
    print(np.sum(np.array(percentiles_no_correction) > 95))
    fields['directional_no_correction'] = np.array(percentiles_no_correction) > 95

    print('Number of directional fields [with BH correction]: ')
    print(np.sum(np.array(percentiles_correction) > 95))
    fields['directional_correction'] = np.array(percentiles_correction) > 95
    fields['directional_percentile'] = np.array(percentiles_correction)
    fields.to_pickle(analysis_path + tag + 'fields.pkl')

    get_percentage_of_grid_cells_with_directional_nodes(fields)
    plot_shuffled_number_of_bins_vs_observed(fields)


def analyze_data(animal, server_path, shuffle_type='occupancy'):
    if animal == 'mouse':
        local_path_to_field_data = local_path_to_shuffled_field_data_mice
        spike_sorter = '/MountainSort'
        accepted_fields = pd.read_excel(analysis_path + 'list_of_accepted_fields.xlsx')
        df_path = '/DataFrames'
    elif animal == 'rat':
        local_path_to_field_data = local_path_to_shuffled_field_data_rats
        spike_sorter = ''
        accepted_fields = pd.read_excel(analysis_path + 'included_fields_detector2_sargolini.xlsx')
        df_path = '/DataFrames'

    else:
        local_path_to_field_data = analysis_path + 'simulated_' + shuffle_type + '.pkl'
        spike_sorter = '/'
        df_path = ''

    shuffled_field_data = load_data_frame_field_data(local_path_to_field_data, server_path, spike_sorter, df_path=df_path, shuffle_type=shuffle_type)
    if animal == 'mouse':
        shuffled_field_data = tag_accepted_fields_mouse(shuffled_field_data, accepted_fields)
    elif animal == 'rat':
        shuffled_field_data = tag_accepted_fields_rat(shuffled_field_data, accepted_fields)
    else:
        shuffled_field_data['accepted_field'] = True
        unique_cell_id = shuffled_field_data.session_id + '_' + shuffled_field_data.cluster_id.apply(str)
        shuffled_field_data['unique_cell_id'] = unique_cell_id
    grid = shuffled_field_data.grid_score >= 0.4
    hd = shuffled_field_data.hd_score >= 0.5
    not_classified = np.logical_and(np.logical_not(grid), np.logical_not(hd))
    grid_cells = np.logical_and(grid, np.logical_not(hd))
    conj_cells = np.logical_and(grid, hd)

    accepted_field = shuffled_field_data.accepted_field == True

    shuffled_field_data_grid = shuffled_field_data[grid_cells & accepted_field]
    shuffled_field_data_not_classified = shuffled_field_data[not_classified & accepted_field]
    shuffled_field_data_conj = shuffled_field_data[conj_cells & accepted_field]

    get_number_of_directional_fields(shuffled_field_data_grid, tag='grid' + animal)
    get_number_of_directional_fields(shuffled_field_data_conj, tag='conjunctive' + animal)
    plot_distributions_for_fields(shuffled_field_data_grid, 'grid', animal=animal, shuffle_type=shuffle_type)
    plot_distributions_for_fields(shuffled_field_data_conj, 'conjunctive', animal=animal, shuffle_type=shuffle_type)
    if len(shuffled_field_data_not_classified) > 0:
        plot_distributions_for_fields(shuffled_field_data_not_classified, 'not_classified', animal=animal, shuffle_type=shuffle_type)

    print('*****')
    print(animal + ' data:')
    print(shuffle_type)
    print('Grid cells:')
    print('Number of grid fields: ' + str(len(shuffled_field_data_grid)))
    print('Number of grid cells: ' + str(len(np.unique(list(shuffled_field_data_grid.unique_cell_id)))))
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_grid, analysis_type='bh', shuffle_type=shuffle_type)
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_grid, analysis_type='percentile', shuffle_type=shuffle_type)
    print('__________________________________')
    print('Not classified cells: ')
    print('Number of not classified fields: ' + str(len(shuffled_field_data_not_classified)))
    print('Number of not classified cells: ' + str(len(np.unique(list(shuffled_field_data_not_classified.unique_cell_id)))))
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_not_classified, analysis_type='bh', shuffle_type=shuffle_type)
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_not_classified, analysis_type='percentile', shuffle_type=shuffle_type)
    print('__________________________________')
    print('__________________________________')
    print('Conjunctive cells: ')
    print('Number of conjunctive fields: ' + str(len(shuffled_field_data_conj)))
    print('Number of conjunctive cells: ' + str(len(np.unique(list(shuffled_field_data_conj.unique_cell_id)))))
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_conj, analysis_type='bh', shuffle_type=shuffle_type)
    compare_shuffled_to_real_data_mw_test(shuffled_field_data_conj, analysis_type='percentile', shuffle_type=shuffle_type)
    print('__________________________________')


def main():
    analyze_data('mouse', server_path_mouse, shuffle_type='distributive')
    analyze_data('rat', server_path_rat, shuffle_type='distributive')
    # server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated() + 'ventral_narrow/'
    # analyze_data('simulated', server_path_simulated, shuffle_type='distributive_narrow')
    # server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated() + 'control_narrow/'
    # analyze_data('simulated', server_path_simulated, shuffle_type='distributive_control_narrow')


if __name__ == '__main__':
    main()
