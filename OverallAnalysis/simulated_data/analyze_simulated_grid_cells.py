import glob
import numpy as np
import os
import pandas as pd
import PostSorting.make_plots
import PostSorting.open_field_make_plots
import PostSorting.open_field_firing_fields
import PostSorting.open_field_head_direction
import PostSorting.open_field_grid_cells
import PostSorting.open_field_spatial_data
import PostSorting.parameters
import PostSorting.open_field_spatial_data
import OverallAnalysis.grid_analysis_other_labs.firing_maps

import matplotlib.pylab as plt



prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(100)  # data is in cm already
prm.set_sampling_rate(1000)


# load data frames and reorganize to be similar to real data to make it easier to rerun analyses
def organize_data(analysis_path):
    spatial_data_path = analysis_path + 'v_spatial_data'
    spatial_data = pd.read_pickle(spatial_data_path)
    position_data = pd.DataFrame()
    position_data['synced_time'] = spatial_data.synced_time.iloc[0]
    position_data['time_seconds'] = spatial_data.synced_time.iloc[0]
    position_data['position_x'] = spatial_data.position_x.iloc[0]
    position_data['position_y'] = spatial_data.position_y.iloc[0]
    position_data['position_x_pixels'] = spatial_data.position_x.iloc[0]
    position_data['position_y_pixels'] = spatial_data.position_y.iloc[0]
    position_data['hd'] = spatial_data.hd.iloc[0] - 180
    for name in glob.glob(analysis_path + '*'):
        if os.path.exists(name) and os.path.isdir(name) is False and name != spatial_data_path:
            if not os.path.isdir(name + '_simulated'):
                cell = pd.read_pickle(name)
                id_count = 1
                cell['session_id'] = 'simulated'
                cell['cluster_id'] = id_count
                cell['animal'] = 'simulated'
                os.mkdir(name + '_simulated')
                position_data.to_pickle(name + '_simulated/position.pkl')
                cell.to_pickle(name + '_simulated/spatial_firing.pkl')
                id_count += 1


def get_rate_maps(position_data, firing_data):
    position_heat_map, spatial_firing = OverallAnalysis.grid_analysis_other_labs.firing_maps.make_firing_field_maps(position_data, firing_data, prm)
    return position_heat_map, spatial_firing


def make_plots(position_data, spatial_firing, position_heat_map, hd_histogram, prm):
    # PostSorting.make_plots.plot_spike_histogram(spatial_firing, prm)
    # PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, prm)
    # PostSorting.make_plots.plot_autocorrelograms(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)
    PostSorting.open_field_make_plots.plot_firing_rate_maps(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing, prm)
    try:
        PostSorting.open_field_make_plots.plot_hd(spatial_firing, position_data, prm)
    except:
        print('I did not manage to plot 2d hd scatter.')
    PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, prm)
    # PostSorting.open_field_make_plots.plot_spikes_on_firing_fields(spatial_firing, prm)
    try:
        PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)
    except:
        print('I did not manage to make combined plots.')


def process_data(analysis_path):
    organize_data(analysis_path)
    for name in glob.glob(analysis_path + '*'):
        if os.path.isdir(name):
            if os.path.exists(name + '/spatial_firing.pkl'):
                print(name)
                prm.set_file_path(name)
                prm.set_output_path(name)
                position = pd.read_pickle(name + '/position.pkl')
                # process position data - add hd etc
                spatial_firing = pd.read_pickle(name + '/spatial_firing.pkl')

                hd = [item for sublist in spatial_firing.hd[0] for item in sublist]
                spatial_firing['hd'] = [np.array(hd) - 180]
                #if len(spatial_firing.hd) == 1:
                 #   spatial_firing['hd'] = np.array(spatial_firing.hd)
                spatial_firing['position_x_pixels'] = spatial_firing.position_x
                spatial_firing['position_y_pixels'] = spatial_firing.position_y

                prm.set_sampling_rate(1000000)  # this is to make the histograms similar to the real data
                hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spatial_firing, position, prm)

                # if 'firing_maps' not in spatial_firing:
                position_heat_map, spatial_firing = get_rate_maps(position, spatial_firing)

                spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
                spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing, position, prm)
                spatial_firing.to_pickle(name + '/spatial_firing.pkl')
                make_plots(position, spatial_firing, position_heat_map, hd_histogram, prm)


def main():
    analysis_path = '/grid_fields/simulated_data/ventral_narrow/'
    process_data(analysis_path)
    analysis_path = 'grid_fields/simulated_data/control_narrow/'
    process_data(analysis_path)


if __name__ == '__main__':
    main()
