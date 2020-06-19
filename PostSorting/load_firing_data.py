import mdaio
import numpy as np
import os
from pathlib import Path
import pandas as pd
import PreClustering.dead_channels
import data_frame_utility


def get_firing_info(file_path, prm):
    firing_times_path = file_path + '/Electrophysiology' + prm.get_sorter_name() + '/firings.mda'
    units_list = None
    firing_info = None
    if os.path.exists(firing_times_path):
        firing_info = mdaio.readmda(firing_times_path)
        units_list = np.unique(firing_info[2])
    else:
        print('I could not find the MountainSort output [firing.mda] file. I will check if the data was sorted earlier.')
        spatial_firing_path = file_path + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.exists(spatial_firing_path):
            spatial_firing = pd.read_pickle(spatial_firing_path)
            os.mknod(file_path + '/sorted_data_exists.txt')
            return units_list, firing_info, spatial_firing
        else:
            print('There are no sorting results available for this recording.')
    return units_list, firing_info, False


# if the recording has dead channels, detected channels need to be shifted to get read channel ids
def correct_detected_ch_for_dead_channels(dead_channels, primary_channels):
    for dead_channel in dead_channels:
        indices_to_add_to = np.where(primary_channels >= dead_channel)
        primary_channels[indices_to_add_to] += 1
    return primary_channels


def correct_for_dead_channels(primary_channels, prm):
    PreClustering.dead_channels.get_dead_channel_ids(prm)
    dead_channels = prm.get_dead_channels()
    if len(dead_channels) != 0:
        dead_channels = list(map(int, dead_channels[0]))
        primary_channels = correct_detected_ch_for_dead_channels(dead_channels, primary_channels)
    return primary_channels


def process_firing_times(recording_to_process, session_type, prm):
    session_id = recording_to_process.split('/')[-1]
    units_list, firing_info, spatial_firing = get_firing_info(recording_to_process, prm)
    if isinstance(spatial_firing, pd.DataFrame):
        firing_data = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'firing_times', 'firing_times_opto', 'isolation', 'noise_overlap', 'peak_snr', 'mean_firing_rate', 'random_snippets', 'position_x', 'position_y', 'hd', 'position_x_pixels', 'position_y_pixels', 'speed']].copy()
        return firing_data
    cluster_ids = firing_info[2]
    firing_times = firing_info[1]
    primary_channel = firing_info[0]
    primary_channel = correct_for_dead_channels(primary_channel, prm)
    if session_type == 'openfield' and prm.get_opto_tagging_start_index() is not None:
        firing_data = data_frame_utility.df_empty(['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'firing_times', 'firing_times_opto'], dtypes=[str, np.uint8, np.uint8, np.uint8, np.uint64, np.uint64])
        for cluster in units_list:
            cluster_firings_all = firing_times[cluster_ids == cluster]
            cluster_firings = np.take(cluster_firings_all, np.where(cluster_firings_all < prm.get_opto_tagging_start_index())[0])
            cluster_firings_opto = np.take(cluster_firings_all, np.where(cluster_firings_all >= prm.get_opto_tagging_start_index())[0])
            channel_detected = primary_channel[cluster_ids == cluster][0]
            tetrode = int((channel_detected-1)/4 + 1)
            ch = int((channel_detected - 1) % 4 + 1)
            firing_data = firing_data.append({
                "session_id": session_id,
                "cluster_id":  int(cluster),
                "tetrode": tetrode,
                "primary_channel": ch,
                "firing_times": cluster_firings,
                "firing_times_opto": cluster_firings_opto
            }, ignore_index=True)
    else:
        firing_data = data_frame_utility.df_empty(['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'firing_times', 'trial_number', 'trial_type'], dtypes=[str, np.uint8, np.uint8, np.uint8, np.uint64, np.uint8, np.uint16])
        for cluster in units_list:
            cluster_firings = firing_times[cluster_ids == cluster]
            channel_detected = primary_channel[cluster_ids == cluster][0]
            tetrode = int((channel_detected-1)/4 + 1)
            ch = int((channel_detected - 1) % 4 + 1)
            firing_data = firing_data.append({
                "session_id": session_id,
                "cluster_id":  int(cluster),
                "tetrode": tetrode,
                "primary_channel": ch,
                "firing_times": cluster_firings
            }, ignore_index=True)
    return firing_data


def create_firing_data_frame(recording_to_process, session_type, prm):
    spike_data = None
    spike_data = process_firing_times(recording_to_process, session_type, prm)
    return spike_data

