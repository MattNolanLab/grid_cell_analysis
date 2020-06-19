from astropy.stats import rayleightest
import os
import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import subprocess
import sys

import PostSorting.open_field_firing_maps


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out


def get_hd_histogram(angles, window_size=23):
    angles = angles[~np.isnan(angles)]
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    binned_hd, _, _ = plt.hist(angles, theta)
    smooth_hd = get_rolling_sum(binned_hd, window=window_size)
    return smooth_hd


# max firing rate at the angle where the firing rate is highest
def get_max_firing_rate(spatial_firing):
    max_firing_rates = []
    preferred_directions = []
    for index, cluster in spatial_firing.iterrows():
        hd_hist = cluster.hd_spike_histogram
        max_firing_rate = np.max(hd_hist.flatten())
        max_firing_rates.append(max_firing_rate)

        preferred_direction = np.where(hd_hist == max_firing_rate)
        preferred_directions.append(preferred_direction[0])

    spatial_firing['max_firing_rate_hd'] = np.array(max_firing_rates) / 1000  # Hz
    spatial_firing['preferred_HD'] = preferred_directions
    return spatial_firing


def get_hd_score_for_cluster(hd_hist):
    angles = np.linspace(-179, 180, 360)
    angles_rad = angles*np.pi/180
    dy = np.sin(angles_rad)
    dx = np.cos(angles_rad)

    totx = sum(dx * hd_hist)/sum(hd_hist)
    toty = sum(dy * hd_hist)/sum(hd_hist)
    r = np.sqrt(totx*totx + toty*toty)
    return r


'''
This test is used to identify a non-uniform distribution, i.e. it is designed for detecting an unimodal deviation from 
uniformity. More precisely, it assumes the following hypotheses: - H0 (null hypothesis): The population is distributed 
uniformly around the circle. - H1 (alternative hypothesis): The population is not distributed uniformly around the 
circle. Small p-values suggest to reject the null hypothesis.

This is an alternative to using the population mean vector as a head-directions score.

https://docs.astropy.org/en/stable/_modules/astropy/stats/circstats.html#rayleightest
'''


def get_rayleigh_score_for_cluster(hd_hist: np.ndarray) -> float:
    bins_in_histogram = len(hd_hist)
    values = np.radians(np.arange(0, 360, int(360 / bins_in_histogram)))
    rayleigh_p = rayleightest(values, weights=hd_hist)
    return rayleigh_p


def add_rayleigh_score_for_all_clusters(spatial_firing: pd.DataFrame) -> pd.DataFrame:
    print('I will do the Rayleigh test to check if head-direction tuning is uniform.')
    rayleigh_ps = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        hd_hist = spatial_firing.hd_spike_histogram[cluster].copy()
        p = get_rayleigh_score_for_cluster(hd_hist)
        rayleigh_ps.append(p)
    spatial_firing['rayleigh_score'] = np.array(rayleigh_ps)
    return spatial_firing


def calculate_hd_score(spatial_firing):
    hd_scores = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        hd_hist = spatial_firing.hd_spike_histogram[cluster].copy()
        r = get_hd_score_for_cluster(hd_hist)
        hd_scores.append(r)
    spatial_firing['hd_score'] = np.array(hd_scores)
    return spatial_firing


# save hd
def save_hd_for_r(hd_session, hd_cluster, cluster, prm):
    fields_path = prm.get_filepath() + '/Firing_fields/'
    save_path = fields_path + str(int(cluster + 1)) + '_whole_field/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    np.savetxt(save_path + 'session.csv', hd_session, delimiter=',')
    np.savetxt(save_path + 'cluster.csv', hd_cluster, delimiter=',')


def write_shell_script_to_call_r_analysis(prm, cluster):
    firing_field_path = prm.get_filepath() + '/Firing_fields/' + str(int(cluster + 1)) + '_whole_field/'
    python_script_path = os.path.dirname(sys.argv[0])
    script_path = prm.get_filepath() + '/Firing_fields' + '/run_r.sh'
    batch_writer = open(script_path, 'w', newline='\n')
    batch_writer.write('#!/bin/bash\n')
    batch_writer.write('echo "-----------------------------------------------------------------------------------"\n')
    batch_writer.write('echo "This is a shell script that will call R to analyze firing fields."\n')
    batch_writer.write('Rscript ' + python_script_path + '/PostSorting/process_fields.r ' + firing_field_path)
    batch_writer.close()


# calculate statistics for hd in fields
def analyze_hd_r(prm, cluster):
    fields_path = prm.get_filepath() + '/Firing_fields/'
    path = fields_path
    write_shell_script_to_call_r_analysis(prm, cluster)
    os.chmod(path + '/run_r.sh', 484)
    subprocess.call(path + '/run_r.sh', shell=True)


def put_stat_results_in_spatial_df(spatial_firing, prm):
    df_stats = pd.DataFrame([])
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        fields_path = prm.get_filepath() + '/Firing_fields/'
        circular_statistics_path = fields_path + str(int(cluster + 1)) + '_whole_field/circular_out.csv'
        if os.path.isfile(circular_statistics_path) is True:
            path_to_hd_stats = circular_statistics_path
            hd_stats_cluster_df = pd.read_csv(path_to_hd_stats)
            df_stats = df_stats.append(hd_stats_cluster_df)
    if 'Watson_two_sample' in df_stats:
        spatial_firing['watson_test_hd'] = df_stats.Watson_two_sample.values
        spatial_firing['kuiper_cluster'] = df_stats.Kuiper_Cluster.values
        spatial_firing['kuiper_session'] = df_stats.Kuiper_Session.values
        spatial_firing['watson_cluster'] = df_stats.Watson_Cluster.values
        spatial_firing['watson_session'] = df_stats.Watson_Session.values
    return spatial_firing


def process_hd_data(spatial_firing, spatial_data, prm):
    print('I will process head-direction data now.')
    angles_whole_session = (np.array(spatial_data.hd) + 180) * np.pi / 180
    hd_histogram = get_hd_histogram(angles_whole_session)
    hd_histogram /= prm.get_sampling_rate()

    hd_spike_histograms = []
    for index, cluster in spatial_firing.iterrows():
        # cluster = spatial_firing.cluster_id.values[index] - 1
        try:
            angles_spike = (cluster.hd + 180) * np.pi / 180
        except:
            angles_spike = (np.array(cluster.hd) + 180) * np.pi / 180

        if prm.get_is_stable() is False:
            print('The watson test is not going to run. If you need this data, you can run it on the dataframes later.')
            # save_hd_for_r(angles_whole_session, angles_spike, index, prm)
            # analyze_hd_r(prm, index)

        hd_spike_histogram = get_hd_histogram(angles_spike)
        hd_spike_histogram = hd_spike_histogram / hd_histogram
        hd_spike_histograms.append(hd_spike_histogram)

    # spatial_firing = put_stat_results_in_spatial_df(spatial_firing, prm)
    spatial_firing['hd_spike_histogram'] = hd_spike_histograms
    spatial_firing = get_max_firing_rate(spatial_firing)
    spatial_firing = calculate_hd_score(spatial_firing)
    spatial_firing = add_rayleigh_score_for_all_clusters(spatial_firing)
    return hd_histogram, spatial_firing


# get HD data for a specific bin of the rate map
def get_indices_for_bin(bin_in_field, spatial_data, prm):
    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size(prm)
    bin_x = bin_in_field[0]
    bin_x_left_pixels = bin_x * bin_size_pixels
    bin_x_right_pixels = (bin_x+1) * bin_size_pixels
    bin_y = bin_in_field[1]
    bin_y_bottom_pixels = bin_y * bin_size_pixels
    bin_y_top_pixels = (bin_y+1) * bin_size_pixels

    left_x_border = spatial_data.x > bin_x_left_pixels
    right_x_border = spatial_data.x < bin_x_right_pixels
    bottom_y_border = spatial_data.y > bin_y_bottom_pixels
    top_y_border = spatial_data.y < bin_y_top_pixels

    inside_bin = spatial_data[left_x_border & right_x_border & bottom_y_border & top_y_border]
    return inside_bin


# get head-direction data from bins of field
def get_hd_in_field_spikes(rate_map_indices, spatial_data, prm):
    hd_in_field = []
    event_times_in_field = []
    for bin_in_field in rate_map_indices:
        inside_bin = get_indices_for_bin(bin_in_field, spatial_data, prm)
        hd = inside_bin.hd.values
        hd_in_field.extend(hd)
        event_times = inside_bin.firing_times.values
        event_times_in_field.extend(event_times)
    return hd_in_field, event_times_in_field


# get head-direction data from bins of field
def get_hd_in_field(rate_map_indices, spatial_data, prm):
    hd_in_field = []
    event_times_in_field = []
    for bin_in_field in rate_map_indices:
        inside_bin = get_indices_for_bin(bin_in_field, spatial_data, prm)
        hd = inside_bin.hd.values
        hd_in_field.extend(hd)
        event_times = inside_bin.synced_time.values
        event_times_in_field.extend(event_times)
    return hd_in_field, event_times_in_field


# return array of HD in subfield when cell fired for cluster
def get_hd_in_firing_rate_bins_for_cluster(spatial_firing, rate_map_indices, cluster, prm):
    cluster_id = np.arange(len(spatial_firing.firing_times[cluster]))
    spatial_firing_cluster = pd.DataFrame(cluster_id)
    if type(spatial_firing.position_x_pixels[cluster]) is np.ndarray:
        spatial_firing_cluster['x'] = spatial_firing.position_x_pixels[cluster]
        spatial_firing_cluster['y'] = spatial_firing.position_y_pixels[cluster]
        spatial_firing_cluster['hd'] = spatial_firing.hd[cluster]
    elif type(spatial_firing.position_x_pixels[cluster]) is list:
        spatial_firing_cluster['x'] = spatial_firing.position_x_pixels[cluster]
        spatial_firing_cluster['y'] = spatial_firing.position_y_pixels[cluster]
        spatial_firing_cluster['hd'] = spatial_firing.hd[cluster]
    else:
        spatial_firing_cluster['x'] = spatial_firing.position_x_pixels[cluster].values
        spatial_firing_cluster['y'] = spatial_firing.position_y_pixels[cluster].values
        spatial_firing_cluster['hd'] = spatial_firing.hd[cluster].values

    spatial_firing_cluster['firing_times'] = spatial_firing.firing_times[cluster]
    hd_in_field, spike_times = get_hd_in_field_spikes(rate_map_indices, spatial_firing_cluster, prm)
    hd_in_field = (np.array(hd_in_field) + 180) * np.pi / 180
    return hd_in_field, spike_times


# return array of HD angles in subfield when from the whole session
def get_hd_in_firing_rate_bins_for_session(spatial_data, rate_map_indices, prm):
    spatial_data_field = pd.DataFrame()
    spatial_data_field['x'] = spatial_data.position_x_pixels
    spatial_data_field['y'] = spatial_data.position_y_pixels
    spatial_data_field['hd'] = spatial_data.hd
    spatial_data_field['synced_time'] = spatial_data.synced_time
    hd_in_field, times = get_hd_in_field(rate_map_indices, spatial_data_field, prm)
    hd_in_field = (np.array(hd_in_field) + 180) * np.pi / 180
    return hd_in_field, times


def main():
    pass


if __name__ == '__main__':
    main()
