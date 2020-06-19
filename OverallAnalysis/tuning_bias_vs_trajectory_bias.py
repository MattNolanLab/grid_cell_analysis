'''
I will quantify how similar the trajectory hd distribution is to a uniform distribution (1 sample watson test)
 and then correlate the results of this to the number of significant bins from the distributive shuffled analysis
'''


import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis_all_animals
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import scipy.stats
# utils = importr('utils')
# utils.install_packages('circular')
from scipy.stats import linregress


analysis_path = OverallAnalysis.folder_path_settings.get_local_path() + '/tuning_bias_vs_trajectory_bias/'
local_path_to_shuffled_field_data_mice = analysis_path + 'shuffled_field_data_all_mice.pkl'
local_path_to_shuffled_field_data_rats = analysis_path + 'shuffled_field_data_all_rats.pkl'


# run 2 sample watson test and put it in df
def run_one_sample_watson_test(hd_session):
    circular = importr("circular")
    watson_test = circular.watson_test
    hd_session = ro.FloatVector(hd_session)
    stat = watson_test(hd_session)
    return stat[0][0]  # this is the part of the return r object that is the stat


def compare_trajectory_hd_to_uniform_dist(fields):
    hd = fields.hd_in_field_session
    stats_values = []
    for field in hd:
        stat = run_one_sample_watson_test(field)
        stats_values.append(stat)
    fields['watson_stat'] = stats_values
    return fields


def plot_results(grid_fields, animal):
    number_of_significantly_directional_bins = grid_fields.number_of_different_bins_bh
    watson_stats = grid_fields.watson_stat
    plt.figure()
    plt.scatter(watson_stats, number_of_significantly_directional_bins)
    plt.xlabel('Bias in trajectory', fontsize=18)
    plt.ylabel('Number of directional bins', fontsize=18)
    plt.savefig(analysis_path + 'number_of_significantly_directional_bins_vs_watson_stats' + animal + '.png')


def add_percentiles(fields):
    percentiles_correction = []
    for index, field in fields.iterrows():
        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled_corrected_p, field.number_of_different_bins_bh)
        percentiles_correction.append(percentile)
    fields['directional_percentile'] = percentiles_correction
    return fields


def check_if_they_correlate(fields):
    print('compare tuning and trajectory bias:')
    trajectory_bias = fields.watson_stat
    tuning = fields.number_of_different_bins_bh
    slope, intercept, r_value, p_value, std_err = linregress(trajectory_bias, tuning)
    print("slope: %f    intercept: %f  p_value %f" % (slope, intercept, p_value))


def process_data(animal):
    print(animal)
    if animal == 'mouse':
        local_path_to_field_data = local_path_to_shuffled_field_data_mice
        accepted_fields = pd.read_excel(analysis_path + 'list_of_accepted_fields.xlsx')
        shuffled_field_data = pd.read_pickle(local_path_to_field_data)
        shuffled_field_data = OverallAnalysis.shuffle_field_analysis_all_animals.tag_accepted_fields_mouse(shuffled_field_data, accepted_fields)

    else:
        local_path_to_field_data = local_path_to_shuffled_field_data_rats
        accepted_fields = pd.read_excel(analysis_path + 'included_fields_detector2_sargolini.xlsx')
        shuffled_field_data = pd.read_pickle(local_path_to_field_data)
        shuffled_field_data = OverallAnalysis.shuffle_field_analysis_all_animals.tag_accepted_fields_rat(shuffled_field_data, accepted_fields)

    grid = shuffled_field_data.grid_score >= 0.4
    hd = shuffled_field_data.hd_score >= 0.5
    grid_cells = np.logical_and(grid, np.logical_not(hd))
    accepted_field = shuffled_field_data.accepted_field == True
    grid_fields = shuffled_field_data[grid_cells & accepted_field]
    grid_fields = compare_trajectory_hd_to_uniform_dist(grid_fields)
    grid_fields = add_percentiles(grid_fields)

    plot_results(grid_fields, animal)
    check_if_they_correlate(grid_fields)


def main():
    process_data('mouse')
    process_data('rat')



if __name__ == '__main__':
    main()


