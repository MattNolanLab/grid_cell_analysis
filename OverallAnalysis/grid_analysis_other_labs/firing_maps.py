from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pylab as plt
import pandas as pd
from numba import jit
import numpy as np
import math
import time


def get_dwell(spatial_data, prm):
    min_dwell_distance_cm = 5  # from point to determine min dwell time
    dt_position_ms = spatial_data.time_seconds.diff().mean() * 1000  # sampling interval in position data
    min_dwell_time_ms = 3 * dt_position_ms  # this is about 100 ms
    min_dwell_time = round(min_dwell_time_ms / dt_position_ms)
    return min_dwell_time, min_dwell_distance_cm


def get_bin_size(prm):
    bin_size_cm = 2.5
    return bin_size_cm


def get_number_of_bins(spatial_data, prm):
    bin_size = get_bin_size(prm)
    length_of_arena_x = spatial_data.position_x[~np.isnan(spatial_data.position_x)].max()
    length_of_arena_y = spatial_data.position_y[~np.isnan(spatial_data.position_y)].max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size)
    number_of_bins_y = math.ceil(length_of_arena_y / bin_size)
    return number_of_bins_x, number_of_bins_y


@jit
def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny


def calculate_firing_rate_for_cluster_parallel(cluster, smooth, firing_data_spatial, positions_x, positions_y, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms):
    print('Started another cluster')
    print(cluster)
    cluster_index = firing_data_spatial.cluster_id.values[cluster] - 1
    cluster_firings = pd.DataFrame({'position_x': firing_data_spatial.position_x[cluster_index], 'position_y': firing_data_spatial.position_y[cluster_index]})
    spike_positions_x = cluster_firings.position_x.values
    spike_positions_y = cluster_firings.position_y.values
    firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y))
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2) + np.power(py - spike_positions_y, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2) + np.power((py - positions_y), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                firing_rate_map[x, y] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))

            else:
                firing_rate_map[x, y] = 0
    #firing_rate_map = np.rot90(firing_rate_map)
    return firing_rate_map


def get_spike_heatmap_parallel(spatial_data, firing_data_spatial, prm):
    print('I will calculate firing rate maps now.')
    dt_position_ms = spatial_data.time_seconds.diff().mean()*1000
    min_dwell, min_dwell_distance_pixels = get_dwell(spatial_data, prm)
    smooth = 5  # / 100 * prm.get_pixel_ratio()
    bin_size_pixels = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)
    num_cores = multiprocessing.cpu_count()
    clusters = range(len(firing_data_spatial))
    time_start = time.time()
    firing_rate_maps = Parallel(n_jobs=num_cores, max_nbytes=None)(delayed(calculate_firing_rate_for_cluster_parallel)(cluster, smooth, firing_data_spatial, spatial_data.position_x.values, spatial_data.position_y.values, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms) for cluster in clusters)
    time_end = time.time()
    print('Making the rate maps took:')
    time_diff = time_end - time_start
    print(time_diff)
    firing_data_spatial['firing_maps'] = firing_rate_maps

    return firing_data_spatial


def get_position_heatmap(spatial_data, prm):
    min_dwell, min_dwell_distance_cm = get_dwell(spatial_data, prm)
    min_dwell_distance_cm = 5
    bin_size_cm = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)

    position_heat_map = np.zeros((number_of_bins_x, number_of_bins_y))

    # find value for each bin for heatmap
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_cm + (bin_size_cm / 2)
            py = y * bin_size_cm + (bin_size_cm / 2)

            occupancy_distances = np.sqrt(np.power((px - spatial_data.position_x.values), 2) + np.power((py - spatial_data.position_y.values), 2))
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_cm)[0])

            if bin_occupancy >= min_dwell:
                position_heat_map[x, y] = bin_occupancy
            else:
                position_heat_map[x, y] = None
    return position_heat_map


# this is the firing rate in the bin with the highest rate
def find_maximum_firing_rate(spatial_firing):
    max_firing_rates = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_rate_map = spatial_firing.firing_maps[cluster]
        max_firing_rate = np.max(firing_rate_map.flatten())
        max_firing_rates.append(max_firing_rate)
    spatial_firing['max_firing_rate'] = max_firing_rates
    return spatial_firing


def make_firing_field_maps(spatial_data, firing_data_spatial, prm):
    position_heat_map = get_position_heatmap(spatial_data, prm)
    firing_data_spatial = get_spike_heatmap_parallel(spatial_data, firing_data_spatial, prm)
    #position_heat_map = np.rot90(position_heat_map)  # to rotate map to be like matlab plots
    firing_data_spatial = find_maximum_firing_rate(firing_data_spatial)
    return position_heat_map, firing_data_spatial