"""
This code shows a few of the analyses implemented in PostSorting on the data frames in ExampleData.
Output figures will be saved in ExampleOutput.
"""

path_where_you_want_to_save_the_output = ''

import os
import pandas as pd
import PostSorting.open_field_firing_maps
import PostSorting.open_field_head_direction
import PostSorting.make_plots
import PostSorting.open_field_make_plots
import PostSorting.parameters
import PostSorting.speed

# the prm object contains some recording parameters such as sampling rate
prm = PostSorting.parameters.Parameters()
# set path to output folder (for figures)
prm.set_output_path(path_where_you_want_to_save_the_output + '/ExampleOutput/')
prm.set_pixel_ratio(440)
prm.set_sampling_rate(30000)  # sampling rate of mouse ephys data


def spatial_firing_analysis():
    # Load data frame with spike sorted data
    spatial_firing = pd.read_pickle('ExampleData/spatial_firing.pkl')
    # Load data frame with the trajectory of the animal
    position = pd.read_pickle('ExampleData/position.pkl')

    # plot how well the animal explored the arena
    position_heat_map = PostSorting.open_field_firing_maps.get_position_heatmap(position, prm)
    PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)

    # plot spikes on the trajectory of the animal
    PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position, spatial_firing, prm)
    # plot firing rate of cell vs running speed
    PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position, prm)

    # calculate the speed score of the cell(s) in spatial_firing and add it to the data frame as a new column
    sspatial_firing = PostSorting.speed.calculate_speed_score(position, spatial_firing, 250,
                                                                 prm.get_sampling_rate())

    # make another plot to look at speed dependence
    PostSorting.make_plots.plot_speed_vs_firing_rate(position, spatial_firing, prm.get_sampling_rate(), 250, prm)

    # plot firing rate maps
    PostSorting.open_field_make_plots.plot_firing_rate_maps(spatial_firing, prm)

    # if the rate map is not in the data frame, run the rate map analyses
    if not 'firing_maps' in spatial_firing:
        position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(position,
                                                                                                  spatial_firing,
                                                                                                  prm)
    # also rerun the grid cell analysis
    spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)

    # plot the autocorrelograms of the firing rate maps
    PostSorting.open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing, prm)

    # get the head direction histogram from the trajectory and rerun hd analyses
    hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spatial_firing,
                                                                                         position, prm)
    # plot traditional polar head direction plots
    PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)

    # plot head direction in individual firing fields
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position, prm)


def main():
    if not os.path.isdir(prm.get_output_path()):  # check if output folder exists
        os.mkdir(prm.get_output_path())  # make it if it doesn't exist
    spatial_firing_analysis()


if __name__ == '__main__':
    main()
