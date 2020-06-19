import glob
import pandas as pd
import plot_utility
import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.false_positives
import OverallAnalysis.folder_path_settings
import OverallAnalysis.analyze_field_correlations
import os
import PostSorting.open_field_head_direction

# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr


local_path_mouse = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/all_mice_df.pkl'
local_path_rat = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/all_rats_df.pkl'
local_path_simulated = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/all_simulated_df.pkl'
path_to_data = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/'
save_output_path = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_cells/'
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


# run 2 sample watson test and put it in df
def run_two_sample_watson_test(hd_cluster, hd_session, downsample=False):
    circular = importr("circular")
    if downsample:
        hd_session = hd_session[::20]
    watson_two_test = circular.watson_two_test
    hd_cluster = ro.FloatVector(hd_cluster)
    hd_session = ro.FloatVector(hd_session)
    stat = watson_two_test(hd_cluster, hd_session)
    return stat[0][0]  # this is the part of the return r object that is the U^2 stat


# call R to tun two sample watson test on HD from firing field when the cell fired vs HD when the mouse was in the field
def compare_hd_when_the_cell_fired_to_heading(cell_data, position, downsample=False):
    two_watson_stats = []
    for index, cell in cell_data.iterrows():
        print('two-sample watson test on ' + str(cell.session_id) + str(cell.cluster_id))
        if type(cell.hd) == list:
            hd_cluster = (np.asanyarray(cell.hd) + 180) * np.pi / 180
        else:
            hd_cluster = (cell.hd + 180) * np.pi / 180
        hd_session = (position.hd + 180) * np.pi / 180
        two_watson_stat = run_two_sample_watson_test(hd_cluster, hd_session, downsample)
        two_watson_stats.append(two_watson_stat)
    cell_data['watson_test_hd'] = two_watson_stats
    return cell_data


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
            position_data = pd.read_pickle(position_data_path)
            downsample = False
            if 'grid_score' in spatial_firing:
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
                                                     'firing_fields']].copy()
                if animal == 'simulated':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'firing_times',
                                                    'hd', 'hd_spike_histogram', 'max_firing_rate_hd', 'preferred_HD',
                                                     'grid_spacing', 'field_size', 'grid_score', 'hd_score', 'firing_fields']].copy()
                    downsample = True

                spatial_firing = compare_hd_when_the_cell_fired_to_heading(spatial_firing, position_data, downsample)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def plot_hd_vs_rayleigh(df_all_cells, animal):
    plt.cla()
    marker_size = 45
    good_cluster = df_all_cells.false_positive == False
    grid_cell = (df_all_cells.grid_score >= 0.4) & (df_all_cells.hd_score < 0.5)
    hd_cell = (df_all_cells.grid_score < 0.4) & (df_all_cells.hd_score >= 0.5)
    conjunctive_cell = (df_all_cells.grid_score >= 0.4) & (df_all_cells.hd_score >= 0.5)

    fig, ax = plt.subplots()
    plt.xscale('linear')
    plt.yscale('linear')
    hd_score = df_all_cells[good_cluster].hd_score
    rayleigh = df_all_cells[good_cluster].rayleigh_score
    plt.scatter(hd_score, rayleigh, color='gray', marker='o', s=marker_size, alpha=0.7, label='Non-spatial')
    hd_score_grid = df_all_cells[good_cluster & grid_cell].hd_score
    rayleigh_grid = df_all_cells[good_cluster & grid_cell].rayleigh_score

    hd_score_hd = df_all_cells[good_cluster & hd_cell].hd_score
    rayleigh_hd = df_all_cells[good_cluster & hd_cell].rayleigh_score

    hd_score_conj = df_all_cells[good_cluster & conjunctive_cell].hd_score
    rayleigh_conj = df_all_cells[good_cluster & conjunctive_cell].rayleigh_score
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.scatter(hd_score_hd, rayleigh_hd, color='navy', marker='o', s=marker_size, label='HD')
    plt.scatter(hd_score_grid, rayleigh_grid, color='red', marker='o', s=marker_size, label='Grid')
    plt.scatter(hd_score_conj, rayleigh_conj, color='orange', marker='o', s=marker_size, label='Conjunctive')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head-direction score', fontsize=30)
    ax.set_ylabel('Rayleigh score', fontsize=30)
    plt.legend(loc='upper left', scatterpoints=1, frameon=False, handletextpad=0.05, prop={'size': 20})
    plt.tight_layout()
    plt.savefig(save_output_path + 'hd_vs_rayleigh_all_cells_' + animal + '.png')
    plt.close()


def plot_watson_vs_rayleigh(df_all_cells, animal):
    plt.cla()
    marker_size = 45
    good_cluster = df_all_cells.false_positive == False
    grid_cell = (df_all_cells.grid_score >= 0.4) & (df_all_cells.hd_score < 0.5)
    hd_cell = (df_all_cells.grid_score < 0.4) & (df_all_cells.hd_score >= 0.5)
    conjunctive_cell = (df_all_cells.grid_score >= 0.4) & (df_all_cells.hd_score >= 0.5)

    fig, ax = plt.subplots()
    plt.xscale('linear')
    plt.yscale('linear')
    watson_score = df_all_cells[good_cluster].watson_test_hd
    rayleigh = df_all_cells[good_cluster].rayleigh_score
    plt.scatter(watson_score, rayleigh, color='gray', marker='o', s=marker_size, alpha=0.7, label='Non-spatial')
    watson_grid = df_all_cells[good_cluster & grid_cell].watson_test_hd
    rayleigh_grid = df_all_cells[good_cluster & grid_cell].rayleigh_score

    watson_hd = df_all_cells[good_cluster & hd_cell].watson_test_hd
    rayleigh_hd = df_all_cells[good_cluster & hd_cell].rayleigh_score

    watson_conj = df_all_cells[good_cluster & conjunctive_cell].watson_test_hd
    rayleigh_conj = df_all_cells[good_cluster & conjunctive_cell].rayleigh_score
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(watson_hd, rayleigh_hd, color='navy', marker='o', s=marker_size, label='HD')
    plt.scatter(watson_grid, rayleigh_grid, color='red', marker='o', s=marker_size, label='Grid')
    plt.scatter(watson_conj, rayleigh_conj, color='orange', marker='o', s=marker_size, label='Conjunctive')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Watson U', fontsize=30)
    ax.set_ylabel('Rayleigh score', fontsize=30)
    plt.legend(loc='upper left', scatterpoints=1, frameon=False, handletextpad=0.05, prop={'size': 20})
    plt.tight_layout()
    plt.savefig(save_output_path + 'watson_vs_rayleigh_all_cells_' + animal + '.png')
    plt.close()


def plot_hd_vs_watson_stat(df_all_cells, animal='mouse'):
    plt.cla()
    marker_size = 45
    good_cluster = df_all_cells.false_positive == False
    grid_cell = (df_all_cells.grid_score >= 0.4) & (df_all_cells.hd_score < 0.5)
    hd_cell = (df_all_cells.grid_score < 0.4) & (df_all_cells.hd_score >= 0.5)
    conjunctive_cell = (df_all_cells.grid_score >= 0.4) & (df_all_cells.hd_score >= 0.5)

    fig, ax = plt.subplots()
    hd_score = df_all_cells[good_cluster].hd_score
    watson_two_stat = df_all_cells[good_cluster].watson_test_hd
    plt.scatter(hd_score, watson_two_stat, color='gray', marker='o', s=marker_size, alpha=0.7, label='Non-spatial')

    hd_score_grid = df_all_cells[good_cluster & grid_cell].hd_score
    watson_two_stat_grid = df_all_cells[good_cluster & grid_cell].watson_test_hd

    hd_score_hd = df_all_cells[good_cluster & hd_cell].hd_score
    watson_two_stat_hd = df_all_cells[good_cluster & hd_cell].watson_test_hd

    hd_score_conj = df_all_cells[good_cluster & conjunctive_cell].hd_score
    watson_two_stat_conj = df_all_cells[good_cluster & conjunctive_cell].watson_test_hd
    if animal == 'mouse':
        plt.xlim([10**-1, 10**0])
        plt.ylim([10**-1, 10**3])
    if animal == 'rat':
        plt.xlim([10**-1, 10**0])
        plt.ylim([10**-1, 10**3])
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.scatter(hd_score_hd, watson_two_stat_hd, color='navy', marker='o', s=marker_size, label='HD')
    plt.scatter(hd_score_grid, watson_two_stat_grid, color='red', marker='o', s=marker_size, label='Grid')
    plt.scatter(hd_score_conj, watson_two_stat_conj, color='orange', marker='o', s=marker_size, label='Conjunctive')
    plt.xscale('log')
    plt.yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head-direction score', fontsize=30)
    ax.set_ylabel('Watson $U^2$', fontsize=30)
    plt.axhline(0.386, color='red', alpha=0.8, linewidth=3, linestyle='--')  # p < 0.001
    plt.axhline(0.268, color='red', alpha=0.8, linewidth=3)  # p < 0.01
    plt.legend(loc='upper left', scatterpoints=1, frameon=False, handletextpad=0.05, prop={'size': 20})
    plt.tight_layout()
    plt.savefig(save_output_path + 'hd_vs_watson_stat_all_cells_' + animal + '_log.png')
    plt.close()


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


def correlation_between_first_and_second_halves_of_session(df_all_animals, animal='mouse'):
    good_cluster = df_all_animals.false_positive == False
    grid_cell = df_all_animals['cell type'] == 'grid'
    # significant_corr = df_all_animals.hd_correlation_first_vs_second_half_p < 0.001
    watson_result_exists = df_all_animals.watson_test_hd.notnull()
    is_hd_cell = df_all_animals.hd_score >= 0.5
    print('Number of cells included in two sample watson test for head-direction from the whole session: ' + str(
        watson_result_exists.sum()))
    print('grid: ' + str(len(df_all_animals[grid_cell & watson_result_exists])))
    watson_significant = df_all_animals.watson_test_hd > 0.268  # p < 0.01
    print('Number of cells with significantly different HD distributions: ' + str(watson_significant.sum()))
    print('Number of excitatory neurons with significantly different HD: ' + str(len(df_all_animals[watson_significant & grid_cell])))

    print('Number of excitatory neurons with significantly different HD that are hd cells: ' + str(len(df_all_animals[watson_significant & grid_cell & is_hd_cell])))

    print('mean pearson r of correlation between first and second half')
    print(df_all_animals.hd_correlation_first_vs_second_half[good_cluster].mean())

    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_animals.hd_correlation_first_vs_second_half[good_cluster & watson_significant], save_output_path + 'correlation_hd_session_' + animal + '.png', y_axis_label='Number of cells')
    OverallAnalysis.analyze_field_correlations.plot_correlation_coef_hist(df_all_animals.hd_correlation_first_vs_second_half[good_cluster & watson_significant & grid_cell], save_output_path + 'correlation_hd_session_grid_cell_' + animal + '.png', y_axis_label='Number of cells')


def plot_results_of_watson_test(df_all_animals, cell_type='grid', animal='mouse', xlim=False):
    if xlim is True:
        tag = 'zoomed'
    else:
        tag = ''
    good_cluster = df_all_animals.false_positive == False
    if cell_type == 'grid':
        grid = df_all_animals.grid_score >= 0.4
        not_hd = df_all_animals.hd_score < 0.5
        cells_to_analyze = grid & not_hd
    elif cell_type == 'hd':
        not_grid = df_all_animals.grid_score < 0.4
        hd = df_all_animals.hd_score >= 0.5
        cells_to_analyze = not_grid & hd
    else:
        not_grid = df_all_animals.grid_score < 0.4
        not_hd = df_all_animals.hd_score < 0.5
        cells_to_analyze = not_grid & not_hd

    watson_test_stats = df_all_animals.watson_test_hd[good_cluster & cells_to_analyze]
    watson_test_stats = watson_test_stats[~np.isnan(watson_test_stats)]

    print('\n' + animal)
    print(cell_type)
    print('all cells: ' + str(len(watson_test_stats)))
    print('significant: ' + str(len(watson_test_stats > 0.268)))

    fig, ax = plt.subplots()
    if xlim == True:
        plt.xlim(0, 1)
    plt.hist(watson_test_stats, bins=30, color='navy', normed=True, alpha=0.7)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.axvline(x=0.268, linewidth=3, color='red')  # p < 0.01 based on r docs for watson two test
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Watson $U^2$', size=30)
    ax.set_ylabel('Proportion', size=30)
    plt.savefig(save_output_path + animal + '_two_sample_watson_stats_hist_all_spikes_' + cell_type + '_cells_' + tag + '.png', bbox_inches="tight")

    plt.close()
    fig, ax = plt.subplots()
    ax = plot_utility.format_bar_chart(ax, 'Watson $U^2$', 'Cumulative probability')
    plt.xscale('log')
    plt.yticks([0, 1])
    plt.xlim(0.1, 100)
    plt.axvline(x=0.268, linewidth=3, color='red')  # p < 0.01 based on r docs for watson two test
    values, base = np.histogram(watson_test_stats, bins=40)
    # evaluate the cumulative
    cumulative = np.cumsum(values / len(watson_test_stats))
    # plot the cumulative function
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)
    plt.savefig(save_output_path + animal + '_two_sample_watson_stats_hist_all_spikes_' + cell_type + '_cells_cumulative' + tag + '.png', bbox_inches="tight")
    plt.close()


def plot_hd_histograms(df_all_animals, cell_type='grid', animal='mouse'):
    good_cluster = df_all_animals.false_positive == False
    if cell_type == 'grid':
        grid = df_all_animals.grid_score >= 0.4
        not_hd = df_all_animals.hd_score < 0.5
        cells_to_analyze = grid & not_hd
    elif cell_type == 'hd':
        not_grid = df_all_animals.grid_score < 0.4
        hd = df_all_animals.hd_score >= 0.5
        cells_to_analyze = not_grid & hd
    else:
        not_grid = df_all_animals.grid_score < 0.4
        not_hd = df_all_animals.hd_score < 0.5
        cells_to_analyze = not_grid & not_hd

    hd_scores = df_all_animals.hd_score[good_cluster & cells_to_analyze]
    hd_scores = hd_scores[~np.isnan(hd_scores)]

    print('\n' + animal)
    print(cell_type)
    print('all cells: ' + str(len(hd_scores)))
    print('significant: ' + str(len(hd_scores > 0.268)))

    fig, ax = plt.subplots()
    plt.hist(hd_scores, bins=30, color='navy', normed=True)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    plt.xlim(0, 1)
    # plt.axvline(x=0.268, linewidth=5, color='red')  # p < 0.01 based on r docs for watson two test
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head-direction score', size=30)
    ax.set_ylabel('Proportion', size=30)
    plt.savefig(save_output_path + animal + '_head_direction_histogram_' + cell_type + '_cells.png', bbox_inches="tight")


def make_descriptive_plots(all_cells, animal):
    plot_watson_vs_rayleigh(all_cells, animal)
    plot_hd_vs_rayleigh(all_cells, animal)
    plot_hd_vs_watson_stat(all_cells, animal)
    plot_results_of_watson_test(all_cells, cell_type='grid', animal=animal)
    plot_results_of_watson_test(all_cells, cell_type='hd', animal=animal)
    plot_results_of_watson_test(all_cells, cell_type='nc')
    plot_results_of_watson_test(all_cells, cell_type='grid', animal=animal, xlim=True)
    plot_results_of_watson_test(all_cells, cell_type='hd', animal=animal, xlim=True)
    plot_results_of_watson_test(all_cells, cell_type='nc', xlim=True)
    plot_hd_histograms(all_cells, cell_type='grid', animal=animal)
    plot_hd_histograms(all_cells, cell_type='hd', animal=animal)
    plot_hd_histograms(all_cells, cell_type='nc', animal=animal)


def tag_false_positives(all_cells, animal):
    if animal == 'mouse':
        false_positives_path = path_to_data + 'false_positives_all.txt'
        list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
        all_cells = add_combined_id_to_df(all_cells)
        all_cells['false_positive'] = all_cells['false_positive_id'].isin(list_of_false_positives)
    else:
        all_cells['false_positive'] = np.full(len(all_cells), False)
    return all_cells


def calculate_watson_from_half_session(all_cells, server_path, sampling_rate):
    watson_results = []
    for index, cell in all_cells.iterrows():
        if ~np.isnan(cell.watson_test_hd):
            watson_results.append(cell.watson_test_hd)
        else:
            position = pd.read_pickle(server_path + cell.session_id + '/MountainSort/DataFrames/position.pkl')
            hd_session = (position.hd.dropna() + 180) * np.pi / 180
            hd_spikes = (np.array(cell.hd)[~np.isnan(cell.hd)] + 180) * np.pi / 180
            stat = run_two_sample_watson_test(hd_session, hd_spikes, downsample=True)
            print(stat)

            '''
            half_time = len(position.synced_time) / 2     # seconds
            last_index_first_half_spikes = len(cell.firing_times[cell.firing_times < half_time * sampling_rate])
            first_half_session = position.hd[:int(half_time)]
            second_half_session = position.hd[int(half_time):]

            first_half_spikes = cell.hd[:last_index_first_half_spikes]
            second_half_spikes = cell.hd[last_index_first_half_spikes:]

            watson_first_halves = run_two_sample_watson_test(first_half_spikes, first_half_session)
            watson_second_halves = run_two_sample_watson_test(second_half_spikes, second_half_session)
            avg_watson = (watson_first_halves + watson_second_halves) / 2
            '''
            watson_results.append(stat)

    all_cells['watson_test_hd'] = watson_results
    return all_cells


def add_rayleigh_score(spatial_firing):
    print('I will do the Rayleigh test to check if head-direction tuning is uniform.')
    rayleigh_ps = []
    for index, cell in spatial_firing.iterrows():
        hd_hist = cell.hd_spike_histogram.copy()
        p = PostSorting.open_field_head_direction.get_rayleigh_score_for_cluster(hd_hist)
        rayleigh_ps.append(p)
    spatial_firing['rayleigh_score'] = np.array(rayleigh_ps)
    return spatial_firing


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
    # correlation_between_first_and_second_halves_of_session(all_cells)
    # if animal == 'mouse':
        # all_cells = calculate_watson_from_half_session(all_cells, server_path_animal, sampling_rate=30000)
    all_cells = add_rayleigh_score(all_cells)

    make_descriptive_plots(all_cells, animal)


def main():
    process_data('mouse')
    process_data('rat')
    process_data('simulated')


if __name__ == '__main__':
    main()
