import os
import numpy as np
import pandas as pd
import subprocess
import PostSorting.open_field_head_direction

import matplotlib.pylab as plt


# return indices of neighbors of bin considering borders
def find_neighbors(bin_to_test, max_x, max_y):
    x = bin_to_test[0]
    y = bin_to_test[1]

    neighbors = [[x, y+1], [x, y-1], [x+1, y], [x-1, y]]

    if x == max_x:
        neighbors = [[x, y+1], [x, y-1], [x-1, y]]
    if y == max_y:
        neighbors = [[x, y-1], [x+1, y], [x-1, y]]
    if x == max_x and y == max_y:
        neighbors = [[x, y-1], [x-1, y]]
    if x == 0:
        neighbors = [[x, y+1], [x, y-1], [x+1, y]]
    if y == 0:
        neighbors = [[x, y+1], [x+1, y], [x-1, y]]
    if x == 0 and y == 0:
        neighbors = [[x, y+1], [x+1, y]]

    if x == max_x and y == 0:
        neighbors = [[x, y+1], [x-1, y]]

    if y == max_y and x == 0:
        neighbors = [[x, y-1], [x+1, y]]

    return neighbors


# return the masked rate map and change the neighbor's indices to 1 if they are above threshold
def find_neighborhood(masked_rate_map, rate_map, firing_rate_of_max, threshold=35):
    changed = False
    threshold = firing_rate_of_max * threshold / 100

    firing_field_bins = np.array(np.where(masked_rate_map > 0))
    firing_field_bins = firing_field_bins.T

    for bin_to_test in firing_field_bins:
        masked_rate_map[bin_to_test[0], bin_to_test[1]] = 2
        neighbors = find_neighbors(bin_to_test, max_x=(masked_rate_map.shape[0]-1), max_y=(masked_rate_map.shape[1]-1))
        for neighbor in neighbors:
            if masked_rate_map[neighbor[0], neighbor[1]] == 2:
                continue

            firing_rate = rate_map[neighbor[0], neighbor[1]]
            if firing_rate >= threshold:
                masked_rate_map[neighbor[0], neighbor[1]] = 1
                changed = True

    return masked_rate_map, changed


# check if the detected field is big enough to be a firing field
def test_if_field_is_big_enough(field_indices):
    number_of_pixels = len(field_indices)
    if number_of_pixels > 45:
        return True
    return False


# this is to avoid identifying the whole rate map as a field
def test_if_field_is_small_enough(field_indices, rate_map):
    number_of_pixels_in_field = len(field_indices)
    number_of_pixels_on_map = len(rate_map.flatten())
    if number_of_pixels_in_field > number_of_pixels_on_map / 2:
        return False
    else:
        return True


def get_field_edge_values(field_indices):
    x_min = np.array(field_indices)[:, 0].min()
    x_max = np.array(field_indices)[:, 0].max()
    y_min = np.array(field_indices)[:, 1].min()
    y_max = np.array(field_indices)[:, 1].max()
    return x_min, x_max, y_min, y_max


def test_if_field_is_not_too_spread_out(field_indices, rate_map):
    x_min, x_max, y_min, y_max = get_field_edge_values(field_indices)
    if (x_max - x_min) >= len(rate_map) / 2:
        return False
    if (y_max - y_min) >= len(rate_map) / 2:
        return False
    return True


def ensure_the_field_does_not_have_a_hole_in_the_middle(field_indices):
    x_min, x_max, y_min, y_max = get_field_edge_values(field_indices)
    middle_x = x_max - int((x_max - x_min) / 2)
    middle_y = y_max - int((y_max - y_min) / 2)
    if [middle_x, middle_y] not in field_indices.tolist():
        return False
    return True


# test if the firing rate of the detected local maximum is higher than average + std firing
def test_if_highest_bin_is_high_enough(rate_map, highest_rate_bin):
    flat_rate_map = rate_map.flatten()
    rate_map_without_removed_fields = np.take(flat_rate_map, np.where(flat_rate_map >= 0))
    average_rate = np.mean(rate_map_without_removed_fields)
    std_rate = np.std(rate_map)

    firing_rate_of_highest_bin = rate_map[highest_rate_bin[0], highest_rate_bin[1]]
    if firing_rate_of_highest_bin < 0.1:
        return False

    if firing_rate_of_highest_bin > average_rate + std_rate:
        return True
    else:
        return False


# find indices for an individual firing field
def find_current_maxima_indices(rate_map, threshold=35):
    highest_rate_bin = np.unravel_index(rate_map.argmax(), rate_map.shape)
    found_new = test_if_highest_bin_is_high_enough(rate_map, highest_rate_bin)
    max_fr = rate_map[highest_rate_bin]
    if found_new is False:
        return None, found_new, None
    # plt.imshow(rate_map)
    # plt.scatter(highest_rate_bin[1], highest_rate_bin[0], marker='o', s=500, color='yellow')
    masked_rate_map = np.full((rate_map.shape[0], rate_map.shape[1]), 0)
    masked_rate_map[highest_rate_bin] = 1
    changed = True
    while changed:
        masked_rate_map, changed = find_neighborhood(masked_rate_map, rate_map, rate_map[highest_rate_bin], threshold=threshold)

    field_indices = np.array(np.where(masked_rate_map > 0)).T
    found_new = test_if_field_is_big_enough(field_indices)
    if found_new is False:
        return None, found_new, None
    found_new = test_if_field_is_small_enough(field_indices, rate_map)
    if found_new is False:
        return None, found_new, None
    found_new = test_if_field_is_not_too_spread_out(field_indices, rate_map)
    if found_new is False:
        return None, found_new, None
    found_new = ensure_the_field_does_not_have_a_hole_in_the_middle(field_indices)
    if found_new is False:
        return None, found_new, None
    return field_indices, found_new, max_fr


# mark indices of firing fields that are already found (so we don't find them again)
def remove_indices_from_rate_map(rate_map, indices):
    for index in indices:
        rate_map[index[0], index[1]] = -10
    return rate_map


# find firing fields and maximum firing rates for each field for a cluster
def get_firing_field_data(spatial_firing, cluster, threshold=35):
    firing_fields_cluster = []
    max_firing_rates_cluster = []
    rate_map = spatial_firing.firing_maps[cluster].copy()
    found_new = True
    while found_new:
        field_indices, found_new, max_firing_rate = find_current_maxima_indices(rate_map, threshold=threshold)
        if found_new:
            firing_fields_cluster.append(field_indices)
            max_firing_rates_cluster.append(max_firing_rate)
            rate_map = remove_indices_from_rate_map(rate_map, field_indices)
    return firing_fields_cluster, max_firing_rates_cluster


def analyze_fields_in_cluster(spatial_firing, cluster, firing_fields=None, max_firing_rates=None, threshold=35):
    if firing_fields is None:
        firing_fields = []
    if max_firing_rates is None:
        max_firing_rates = []
    firing_fields_cluster, max_firing_rates_cluster = get_firing_field_data(spatial_firing, cluster, threshold=threshold)
    firing_fields.append(firing_fields_cluster)
    max_firing_rates.append(max_firing_rates_cluster)
    return firing_fields, max_firing_rates


# find firing fields and add them to spatial firing data frame
def analyze_firing_fields(spatial_firing, spatial_data, prm):
    print('I will identify individual firing fields if possible.')
    firing_fields = []
    max_firing_rates = []

    if prm.get_first_half_only() or prm.get_second_half_only():
        spatial_firing_whole_session = pd.read_pickle(prm.get_local_recording_folder_path() + '/DataFrames/spatial_firing.pkl')
        spatial_firing['firing_fields'] = spatial_firing_whole_session.firing_fields
        spatial_firing['field_max_firing_rate'] = spatial_firing_whole_session.max_firing_rate
        spatial_firing = analyze_hd_in_firing_fields(spatial_firing, spatial_data, prm)
        return spatial_firing

    for cluster in range(len(spatial_firing)):
        cluster_id = spatial_firing.cluster_id.values[cluster] - 1
        firing_fields, max_firing_rates = analyze_fields_in_cluster(spatial_firing, cluster_id, firing_fields, max_firing_rates)

    spatial_firing['firing_fields'] = firing_fields
    spatial_firing['field_max_firing_rate'] = max_firing_rates
    spatial_firing = analyze_hd_in_firing_fields(spatial_firing, spatial_data, prm)
    return spatial_firing


# save hd that corresponds to fields
def save_hd_in_fields(hd_session, hd_cluster, cluster, field_id, prm):
    fields_path = prm.get_filepath() + '/Firing_fields/'
    save_path = fields_path + str(int(cluster + 1)) + '/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    np.savetxt(save_path + 'field_' + str(int(field_id + 1)) + '_session.csv', hd_session, delimiter=',')
    np.savetxt(save_path + 'field_' + str(int(field_id + 1)) + '_cluster.csv', hd_cluster, delimiter=',')


def write_shell_script_to_call_r_analysis(prm, cluster):
    firing_field_path = prm.get_filepath() + '/Firing_fields/' + str(int(cluster + 1)) + '/'
    script_path = prm.get_filepath() + '/Firing_fields' + '/run_r.sh'
    batch_writer = open(script_path, 'w', newline='\n')
    batch_writer.write('#!/bin/bash\n')
    batch_writer.write('echo "-----------------------------------------------------------------------------------"\n')
    batch_writer.write('echo "This is a shell script that will call R to analyze firing fields."\n')
    batch_writer.write('Rscript /home/nolanlab/PycharmProjects/in_vivo_ephys_openephys/PostSorting/process_fields.r ' + firing_field_path)
    batch_writer.close()


# calculate statistics for hd in fields
def analyze_fields_r(prm, cluster):
    fields_path = prm.get_filepath() + '/Firing_fields/'
    path = fields_path
    write_shell_script_to_call_r_analysis(prm, cluster)
    os.chmod(path + '/run_r.sh', 484)
    subprocess.call(path + '/run_r.sh', shell=True)


def analyze_hd_in_field(spatial_data, field, prm, spatial_firing, cluster, field_id):
    hd_in_field_session, times_in_field = PostSorting.open_field_head_direction.get_hd_in_firing_rate_bins_for_session(spatial_data, field, prm)
    hd_in_field_cluster, spike_times_in_field = PostSorting.open_field_head_direction.get_hd_in_firing_rate_bins_for_cluster(spatial_firing, field, cluster, prm)
    save_hd_in_fields(hd_in_field_session, hd_in_field_cluster, cluster, field_id, prm)
    hd_hist_session = PostSorting.open_field_head_direction.get_hd_histogram(hd_in_field_session)
    hd_hist_session /= prm.get_sampling_rate()
    hd_hist_cluster = PostSorting.open_field_head_direction.get_hd_histogram(hd_in_field_cluster)
    max_firing_rate_cluster = np.max(hd_hist_cluster.flatten())
    hd_score_cluster = PostSorting.open_field_head_direction.get_hd_score_for_cluster(hd_hist_cluster)
    preferred_direction = np.where(hd_hist_cluster == max_firing_rate_cluster)
    return hd_hist_session, hd_hist_cluster, max_firing_rate_cluster, hd_score_cluster, preferred_direction, hd_in_field_cluster, hd_in_field_session, spike_times_in_field, times_in_field


def analyze_hd_in_firing_fields(spatial_firing, spatial_data, prm):
    print('I will analyze head-direction in the detected firing fields.')
    hd_session_all = []
    hd_cluster_all = []
    max_firing_rates_all = []
    preferred_hd_all = []
    hd_score_all = []
    number_of_spikes_in_field_all = []
    number_of_samples_in_field_all = []
    spike_times_in_field_all = []
    times_in_session_all = []

    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
        firing_fields_cluster = spatial_firing.firing_fields[cluster]
        hd_session = []
        hd_cluster = []
        max_firing_rate = []
        preferred_hd = []
        hd_score = []
        number_of_spikes_in_fields = []
        number_of_samples_in_fields = []
        spike_times_in_fields = []
        times_in_field_sessions = []
        if number_of_firing_fields > 0:

            for field_id, field in enumerate(firing_fields_cluster):
                hd_hist_session, hd_hist_cluster, max_firing_rate_cluster, hd_score_cluster, preferred_direction, hd_in_field_cluster, hd_in_field_session, spike_times_in_field, times_in_field = analyze_hd_in_field(spatial_data, field, prm, spatial_firing, cluster, field_id)
                hd_session.append(list(hd_hist_session))
                hd_cluster.append(list(hd_hist_cluster))
                max_firing_rate.append(max_firing_rate_cluster/1000)
                preferred_hd.append(preferred_direction[0])
                hd_score.append(hd_score_cluster)
                number_of_spikes_in_fields.append(len(hd_in_field_cluster))
                number_of_samples_in_fields.append(len(hd_in_field_session))
                spike_times_in_fields.append(spike_times_in_field)
                times_in_field_sessions.append(times_in_field)

            # analyze_fields_r(prm, cluster)
        else:
            hd_session.append([None])
            hd_cluster.append([None])
            max_firing_rate.append(None)
            preferred_hd.append(None)
            hd_score.append(None)
            number_of_spikes_in_fields.append([None])
            number_of_samples_in_fields.append([None])
            spike_times_in_fields.append([None])
            times_in_field_sessions.append([None])

        hd_session_all.append(hd_session)
        hd_cluster_all.append(hd_cluster)
        max_firing_rates_all.append(max_firing_rate)
        preferred_hd_all.append(preferred_hd)
        hd_score_all.append(hd_score)
        number_of_spikes_in_field_all.append(number_of_spikes_in_fields)
        number_of_samples_in_field_all.append(number_of_samples_in_fields)
        spike_times_in_field_all.append(spike_times_in_fields)
        times_in_session_all.append(times_in_field_sessions)

    spatial_firing['firing_fields_hd_session'] = hd_session_all
    spatial_firing['firing_fields_hd_cluster'] = hd_cluster_all
    spatial_firing['field_hd_max_rate'] = max_firing_rates_all
    spatial_firing['field_preferred_hd'] = preferred_hd_all
    spatial_firing['field_hd_score'] = hd_score_all
    spatial_firing['number_of_spikes_in_fields'] = number_of_spikes_in_field_all
    spatial_firing['time_spent_in_fields_sampling_points'] = number_of_samples_in_field_all
    spatial_firing['spike_times_in_fields'] = spike_times_in_field_all
    spatial_firing['times_in_session_fields'] = times_in_session_all
    return spatial_firing
