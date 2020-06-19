import file_utility
import glob
from mat4py import loadmat
import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd
import sys
import traceback

import OverallAnalysis.grid_analysis_other_labs.firing_maps
import PostSorting.make_plots
import PostSorting.open_field_make_plots
import PostSorting.open_field_firing_fields
import PostSorting.open_field_head_direction
import PostSorting.open_field_grid_cells
import PostSorting.open_field_spatial_data
import PostSorting.parameters
prm = PostSorting.parameters.Parameters()


# this is necessary, because several datasets are missing tracking information from the second LED
def check_if_all_columns_exist(matlab_data):
    if len(matlab_data['post']) == 0:
        print('The position data is missing the timestamp values.')
        return False
    if len(matlab_data['posx']) == 0:
        print('The position data is missing the x1 coordinates.')
        return False
    if len(matlab_data['posx2']) == 0:
        print('The position data is missing the x2 coordinates.')
        return False
    if len(matlab_data['posy']) == 0:
        print('The position data is missing the y1 coordinates.')
        return False
    if len(matlab_data['posy2']) == 0:
        print('The position data is missing the y2 coordinates.')
        return False
    return True


def get_position_data_frame(matlab_data):
    all_columns_exist = check_if_all_columns_exist(matlab_data)
    if all_columns_exist:
        position_data = pd.DataFrame()
        position_data['time_seconds'] = matlab_data['post']
        position_data.time_seconds = position_data.time_seconds.sum()
        position_data['x_left_cleaned'] = matlab_data['posx']
        position_data.x_left_cleaned = position_data.x_left_cleaned.sum()
        position_data['x_right_cleaned'] = matlab_data['posx2']
        position_data.x_right_cleaned = position_data.x_right_cleaned.sum()
        position_data['y_left_cleaned'] = matlab_data['posy']
        position_data.y_left_cleaned = position_data.y_left_cleaned.sum()
        position_data['y_right_cleaned'] = matlab_data['posy2']
        position_data.y_right_cleaned = position_data.y_right_cleaned.sum()

        position_data = PostSorting.open_field_spatial_data.calculate_position(position_data)  # get central position and interpolate missing data
        position_data = PostSorting.open_field_spatial_data.calculate_head_direction(position_data)  # use coord from the two beads to get hd and interpolate
        position_data = PostSorting.open_field_spatial_data.shift_to_start_from_zero_at_bottom_left(position_data)
        position_data = PostSorting.open_field_spatial_data.calculate_central_speed(position_data)
        position_data['position_x_pixels'] = position_data.position_x.values  # this is here so I don't have to change the pipeline too much
        position_data['position_y_pixels'] = position_data.position_y.values
        position_data['hd'] = position_data['hd'].values
        position_data['synced_time'] = position_data.time_seconds
        position_of_mouse = position_data[['time_seconds', 'synced_time', 'position_x', 'position_y', 'position_x_pixels', 'position_y_pixels', 'hd', 'speed']].copy()
        # plt.plot(position_data.position_x, position_data.position_y) # this is to plot the trajectory.
        return position_of_mouse
    else:
        return False


# calculate the sampling rate of the position data (camera) based on the intervals in the array
def calculate_position_sampling_rate(position_data):
    times = position_data.time_seconds
    interval = times[1] - times[0]
    sampling_rate = 1 / interval
    return sampling_rate


# search for all cells in the session where the position data was found correctly
def get_firing_data(folder_to_search_in, session_id, firing_data):
    firing_times_all_cells = []
    session_ids_all = []
    cell_names_all = []
    cluster_id_all = []
    number_of_spikes_all = []
    cell_counter = 1
    for name in glob.glob(folder_to_search_in + '/*' + session_id + '*'):
        if os.path.exists(name) and os.path.isdir(name) is False:
                if 'EEG' not in name and 'EGF' not in name and 'POS' not in name and 'md5' not in name:
                    cell_id = name.split('\\')[-1].split('_')[-1].split('.')[0]
                    print('I found this cell:' + name)
                    firing_times = pd.DataFrame()
                    firing_times['times'] = loadmat(name)['cellTS']
                    firing_times['times'] = firing_times['times'].sum()
                    firing_times_all_cells.append(firing_times.times.values)
                    cell_names_all.append(cell_id)
                    session_ids_all.append(session_id)
                    cluster_id_all.append(cell_counter)
                    number_of_spikes_all.append(len(firing_times.times))
                    cell_counter += 1
    firing_data['session_id'] = session_ids_all
    firing_data['cell_id'] = cell_names_all
    firing_data['cluster_id'] = cluster_id_all
    firing_data['firing_times'] = firing_times_all_cells
    firing_data['number_of_spikes'] = number_of_spikes_all
    return firing_data


# get corresponding position data for firing events
def get_spatial_data_for_firing_events(firing_data, position_data, sampling_rate_position_data):
    spike_position_x_all = []
    spike_position_y_all = []
    spike_hd_all = []
    spike_speed_all = []
    mean_firing_rate_all = []
    total_length_of_session_seconds = position_data.time_seconds.max()
    for index, cell in firing_data.iterrows():
        firing_times = cell.firing_times.round(2)  # turn this into position indices based on sampling rate
        corresponding_indices_in_position_data = np.round(firing_times / (1 / sampling_rate_position_data))
        spike_x = position_data.position_x[corresponding_indices_in_position_data]
        spike_y = position_data.position_y[corresponding_indices_in_position_data]
        spike_hd = position_data.hd[corresponding_indices_in_position_data]
        spike_speed = position_data.speed[corresponding_indices_in_position_data]
        spike_position_x_all.append(spike_x)
        spike_position_y_all.append(spike_y)
        spike_hd_all.append(spike_hd)
        spike_speed_all.append(spike_speed)
        mean_firing_rate_all.append(cell.number_of_spikes / total_length_of_session_seconds)
    firing_data['position_x'] = np.array(spike_position_x_all)
    firing_data['position_y'] = np.array(spike_position_y_all)
    firing_data['position_x_pixels'] = np.array(spike_position_x_all)
    firing_data['position_y_pixels'] = np.array(spike_position_y_all)
    firing_data['hd'] = np.array(spike_hd_all)
    firing_data['speed'] = np.array(spike_speed_all)
    firing_data['mean_firing_rate'] = np.array(mean_firing_rate_all)
    return firing_data


# load firing data and get corresponding spatial data
def fill_firing_data_frame(position_data, firing_data, name, folder_to_search_in, session_id):
    sampling_rate_of_position_data = calculate_position_sampling_rate(position_data)
    # example file name: 10073-17010302_POS.mat - ratID-sessionID_POS.mat
    print('Session ID = ' + session_id)
    firing_data_session = pd.DataFrame()
    firing_data_session = get_firing_data(folder_to_search_in, session_id, firing_data_session)
    firing_data = firing_data.append(firing_data_session)
    # get corresponding position and HD data for spike data frame
    firing_data = get_spatial_data_for_firing_events(firing_data, position_data, sampling_rate_of_position_data)
    return firing_data


# make folder for output and set parameter object to point at it
def create_folder_structure(file_path, session_id, prm):
    main_folder = file_path.split('\\')[:-1][0]
    main_recording_session_folder = main_folder + '/' + session_id
    prm.set_file_path(main_recording_session_folder)
    prm.set_output_path(main_recording_session_folder)
    if os.path.isdir(main_recording_session_folder) is False:
        os.makedirs(main_recording_session_folder)
        print('I made this folder: ' + main_recording_session_folder)


def get_rate_maps(position_data, firing_data):
    position_heat_map, spatial_firing = OverallAnalysis.grid_analysis_other_labs.firing_maps.make_firing_field_maps(position_data, firing_data, prm)
    return position_heat_map, spatial_firing


def save_data_frames(spatial_firing, spatial_data):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing.pkl')
    spatial_data.to_pickle(prm.get_output_path() + '/DataFrames/position.pkl')


def make_plots(position_data, spatial_firing, position_heat_map, hd_histogram, prm):
    # PostSorting.make_plots.plot_spike_histogram(spatial_firing, prm)
    PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, prm)
    # PostSorting.make_plots.plot_autocorrelograms(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)
    PostSorting.open_field_make_plots.plot_firing_rate_maps(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, prm)
    # PostSorting.open_field_make_plots.plot_spikes_on_firing_fields(spatial_firing, prm)
    PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)


def process_data(folder_to_search_in):
    prm.set_sampling_rate(48000)  # this is according to Sargolini et al. (2006)
    prm.set_pixel_ratio(100)  # this is because the data is already in cm so there's no need to convert
    prm.set_sorter_name('Manual')
    # prm.set_is_stable(True)  # todo: this needs to be removed - R analysis won't run for now
    for name in glob.glob(folder_to_search_in + '/*.mat'):
        if os.path.exists(name):
            if 'POS' in name:
                print('I found this:' + name)
                position_data_matlab = loadmat(name)
                position_data = get_position_data_frame(position_data_matlab)
                session_id = name.split('\\')[-1].split('_')[0]
                if position_data is not False:
                    try:
                        firing_data = pd.DataFrame()
                        create_folder_structure(name, session_id, prm)
                        firing_data = fill_firing_data_frame(position_data, firing_data, name, folder_to_search_in, session_id)
                        hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(firing_data, position_data, prm)
                        position_heat_map, spatial_firing = get_rate_maps(position_data, firing_data)
                        spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
                        spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing, position_data, prm)
                        save_data_frames(spatial_firing, position_data)
                        make_plots(position_data, spatial_firing, position_heat_map, hd_histogram, prm)
                    except Exception as ex:
                        print('I failed to analyze this cell.')
                        print(ex)
                        exc_type, exc_value, exc_traceback = sys.exc_info()
                        traceback.print_tb(exc_traceback)

    print('Processing finished.')


def main():
    process_data('/Sargolini/all_data')
   

if __name__ == '__main__':
    main()