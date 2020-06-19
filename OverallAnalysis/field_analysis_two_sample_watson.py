import glob
import matplotlib.pylab as plt
import numpy as np
import os
import OverallAnalysis.folder_path_settings
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr


server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()
analysis_path = OverallAnalysis.folder_path_settings.get_local_path() + '/watson_two_test_fields/'


def calculate_watson_results_for_shuffled_data(server_path, spike_sorter, df_path='/DataFrames'):
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + df_path + '/shuffled_fields.pkl'
        data_frame_out = recording_folder + spike_sorter + df_path + '/shuffled_fields_watson.pkl'
        if os.path.exists(data_frame_path):
            if os.path.exists(data_frame_out):
                continue
            print('I found a field data frame. ' + recording_folder)
            field_data = pd.read_pickle(data_frame_path)
            if 'shuffled_hd_distribution' in field_data:
                watson_stat_all_fields = []
                for field in range((len(field_data))):
                    hd_session = field_data.hd_in_field_session.iloc[field]
                    shuffled_hd = field_data.shuffled_hd_distribution.iloc[field]
                    number_of_spikes_in_field = field_data.iloc[field].number_of_spikes_in_field
                    number_of_shuffles = int(len(shuffled_hd) / number_of_spikes_in_field)
                    watson_stat_shuffle = []
                    for shuffle in range(number_of_shuffles):
                        individual_shuffle = shuffled_hd[shuffle*number_of_spikes_in_field:(shuffle+1) * number_of_spikes_in_field]
                        individual_shuffle = np.around(individual_shuffle, decimals=2)
                        watson_stat = run_two_sample_watson_test(individual_shuffle, hd_session)
                        watson_stat = round(watson_stat, 2)
                        watson_stat_shuffle.append(watson_stat)
                    watson_stat_all_fields.append(watson_stat_shuffle)
                field_data['watson_stat_shuffled'] = watson_stat_all_fields
                field_data.to_pickle(data_frame_out)


# load field data from server - must include hd in fields
def load_data_frame_field_data(output_path, server_path, spike_sorter='/MountainSort', df_path='/DataFrames'):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data
    else:
        field_data_combined = pd.DataFrame()
        for recording_folder in glob.glob(server_path + '*'):
            os.path.isdir(recording_folder)
            data_frame_path = recording_folder + spike_sorter + df_path + '/shuffled_fields_watson.pkl'
            if os.path.exists(data_frame_path):
                print('I found a field data frame.')
                field_data = pd.read_pickle(data_frame_path)
                if 'shuffled_hd_distribution' in field_data:
                    field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id',
                                                        'number_of_spikes_in_field',
                                                        'hd_in_field_spikes', 'hd_hist_spikes',
                                                        'time_spent_in_field',
                                                        'hd_in_field_session', 'hd_hist_session',
                                                        'hd_histogram_real_data', 'time_spent_in_bins',
                                                        'field_histograms_hz', 'grid_score', 'grid_spacing',
                                                        'hd_score', 'watson_stat_shuffled']].copy()

                    field_data_combined = field_data_combined.append(field_data_to_combine)
                    # print(field_data_combined.head())
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


# run 2 sample watson test and put it in df
def run_two_sample_watson_test(hd_cluster, hd_session):
    circular = importr("circular")
    watson_two_test = circular.watson_two_test
    hd_cluster = ro.FloatVector(hd_cluster)
    hd_session = ro.FloatVector(hd_session)
    stat = watson_two_test(hd_cluster, hd_session)
    return stat[0][0]  # this is the part of the return r object that is the stat


# call R to tun two sample watson test on HD from firing field when the cell fired vs HD when the mouse was in the field
def compare_hd_when_the_cell_fired_to_heading(field_data):
    two_watson_stats = []
    for index, field in field_data.iterrows():
        print('analyzing ' + field.unique_id)
        hd_cluster = field.hd_in_field_spikes
        hd_session = field.hd_in_field_session
        two_watson_stat = run_two_sample_watson_test(hd_cluster, hd_session)
        two_watson_stats.append(two_watson_stat)
    field_data['watson_two_stat'] = two_watson_stats
    return field_data


def plot_histogram_of_watson_stat(field_data, type='all', animal='mouse', xlim=False):
    if type == 'grid':
        grid_cells = field_data['cell type'] == 'grid'
        watson_stats_accepted_fields = field_data.watson_two_stat[field_data.accepted_field & grid_cells]
        watson_shuffled = field_data.watson_stat_shuffled[field_data.accepted_field & grid_cells]
    elif type == 'nc':
        not_classified = field_data['cell type'] == 'na'
        watson_stats_accepted_fields = field_data.watson_two_stat[field_data.accepted_field & not_classified]
        watson_shuffled = field_data.watson_stat_shuffled[field_data.accepted_field & not_classified]
    else:
        watson_stats_accepted_fields = field_data.watson_two_stat[field_data.accepted_field]
        watson_shuffled = field_data.watson_stat_shuffled[field_data.accepted_field]

    fig, ax = plt.subplots()
    if xlim is True:
        plt.xlim(0, 0.5)
        tag = 'zoom'
    else:
        tag = ''
    plt.hist(watson_stats_accepted_fields, bins=20, color='navy', alpha=0.7, normed=True)
    plt.hist(watson_shuffled.values.flatten()[0], bins=20, color='grey', alpha=0.7, normed=True)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    print('Number of ' + type + ' fields in ' + animal + ': ' + str(len(watson_stats_accepted_fields)))
    print('p < 0.01 for ' + str((watson_stats_accepted_fields > 0.268).sum()))

    # plt.axvline(x=0.385, linewidth=1, color='red')  # p < 0.001 threshold
    plt.axvline(x=0.268, linewidth=3, color='red')  # p < 0.01 based on r docs for watson two test
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Watson test statistic', size=30)
    ax.set_ylabel('Frequency', size=30)
    plt.savefig(analysis_path + 'two_sample_watson_stats_hist_' + type + '_' + animal + tag + '.png', bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    plt.yticks([0, 1])
    values, base = np.histogram(watson_stats_accepted_fields, bins=40)
    cumulative = np.cumsum(values / len(watson_stats_accepted_fields))
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)

    values, base = np.histogram(watson_shuffled.values.flatten()[0], bins=40)
    cumulative = np.cumsum(values / len(watson_shuffled.values.flatten()[0]))
    plt.plot(base[:-1], cumulative, c='gray', linewidth=5)

    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xscale('log')
    plt.axvline(x=0.268, linewidth=3, color='red')  # p < 0.01 based on r docs for watson two test
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Watson test statistic', size=30)
    ax.set_ylabel('Cumulative probability', size=30)
    plt.savefig(analysis_path + 'two_sample_watson_stats_hist_' + type + '_' + animal + tag + 'cumulative.png', bbox_inches="tight")


def analyze_data(animal):
    if animal == 'mouse':
        server_path = server_path_mouse
        false_positive_file_name = 'list_of_accepted_fields.xlsx'
        data_frame_name = 'all_mice_fields_watson_test.pkl'
        spike_sorter = '/MountainSort'
        df_path = '/DataFrames'

    elif animal == 'rat':
        server_path = server_path_rat
        false_positive_file_name = 'included_fields_detector2_sargolini.xlsx'
        data_frame_name = 'all_rats_fields_watson_test.pkl'
        spike_sorter = ''
        df_path = '/DataFrames'
    else:
        server_path = server_path_simulated
        data_frame_name = 'all_simulated_fields_watson_test.pkl'
        spike_sorter = ''
        df_path = ''

    calculate_watson_results_for_shuffled_data(server_path, spike_sorter, df_path=df_path)

    field_data = load_data_frame_field_data(analysis_path + data_frame_name, server_path, spike_sorter=spike_sorter, df_path=df_path)   # for two-sample watson analysis
    if animal == 'mouse':
        accepted_fields = pd.read_excel(analysis_path + false_positive_file_name)
        field_data = tag_accepted_fields_mouse(field_data, accepted_fields)
    elif animal == 'rat':
        accepted_fields = pd.read_excel(analysis_path + false_positive_file_name)
        field_data = tag_accepted_fields_rat(field_data, accepted_fields)
    else:
        field_data['accepted_field'] = True
    field_data = add_cell_types_to_data_frame(field_data)
    field_data = compare_hd_when_the_cell_fired_to_heading(field_data)
    plot_histogram_of_watson_stat(field_data, animal=animal)
    plot_histogram_of_watson_stat(field_data, type='grid', animal=animal)
    plot_histogram_of_watson_stat(field_data, type='nc', animal=animal)
    plot_histogram_of_watson_stat(field_data, animal=animal, xlim=True)
    plot_histogram_of_watson_stat(field_data, type='grid', animal=animal, xlim=True)
    plot_histogram_of_watson_stat(field_data, type='nc', animal=animal, xlim=True)


def main():
    # analyze_data('simulated')  # todo downsample
    analyze_data('mouse')
    analyze_data('rat')


if __name__ == '__main__':
    main()