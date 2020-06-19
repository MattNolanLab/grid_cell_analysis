import matplotlib.pylab as plt
import numpy as np
from scipy.stats.stats import pearsonr
import scipy.signal
import PostSorting.open_field_firing_maps
import plot_utility
import PostSorting.open_field_grid_cells


def plot_two_rate_maps_with_spatial_score(rate_map_1, rate_map_2, corr_score, excluded_bins, path):
    print('Plot rate maps.')
    plt.cla()
    fig, axs = plt.subplots(2)
    fig.suptitle('Spatial corr: ' + str(round(corr_score, 2)) + '\n % of excluded bins: ' + str(round(excluded_bins, 2)) + ' %')
    map_1 = axs[0].imshow(rate_map_1, cmap='jet', interpolation='nearest')
    fig.colorbar(map_1)
    map_2 = axs[1].imshow(rate_map_2, cmap='jet', interpolation='nearest')
    fig.colorbar(map_2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()


def make_trajectory_heat_maps(whole_trajectory, trajectory_1, trajectory_2, number_of_bins_x, number_of_bins_y, prm):
    min_dwell, min_dwell_distance_cm = PostSorting.open_field_firing_maps.get_dwell(whole_trajectory, prm)
    bin_size_cm = PostSorting.open_field_firing_maps.get_bin_size(prm)
    position_heat_map_first = PostSorting.open_field_firing_maps.get_position_heatmap_fixed_bins(trajectory_1, number_of_bins_x, number_of_bins_y, bin_size_cm, min_dwell_distance_cm, min_dwell)
    position_heat_map_second = PostSorting.open_field_firing_maps.get_position_heatmap_fixed_bins(trajectory_2, number_of_bins_x, number_of_bins_y, bin_size_cm, min_dwell_distance_cm, min_dwell)
    print('Made trajectory heatmaps for both halves.')
    return position_heat_map_first, position_heat_map_second


def make_same_sized_rate_maps(trajectory_1, trajectory_2, spatial_firing_1, spatial_firing_2, prm):
    spatial_firing_1.set_index([spatial_firing_1.cluster_id - 1], inplace=True)
    spatial_firing_2.set_index([spatial_firing_2.cluster_id - 1], inplace=True)
    whole_trajectory = trajectory_1.append(trajectory_2)

    number_of_bins_x, number_of_bins_y = PostSorting.open_field_firing_maps.get_number_of_bins(whole_trajectory, prm)
    dt_position_ms = whole_trajectory.synced_time.diff().mean() * 1000
    smooth = 5 / 100 * prm.get_pixel_ratio()
    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size(prm)
    min_dwell, min_dwell_distance_pixels = PostSorting.open_field_firing_maps.get_dwell(whole_trajectory, prm)
    cluster = 0
    rate_map_1 = PostSorting.open_field_firing_maps.calculate_firing_rate_for_cluster_parallel(cluster, smooth, spatial_firing_1,
                                                                                  trajectory_1.position_x_pixels,
                                                                                  trajectory_1.position_y_pixels,
                                                                                  number_of_bins_x, number_of_bins_y,
                                                                                  bin_size_pixels, min_dwell,
                                                                                  min_dwell_distance_pixels,
                                                                                  dt_position_ms)
    rate_map_2 = PostSorting.open_field_firing_maps.calculate_firing_rate_for_cluster_parallel(cluster, smooth, spatial_firing_2,
                                                                                  trajectory_2.position_x_pixels,
                                                                                  trajectory_2.position_y_pixels,
                                                                                  number_of_bins_x, number_of_bins_y,
                                                                                  bin_size_pixels, min_dwell,
                                                                                  min_dwell_distance_pixels,
                                                                                  dt_position_ms)

    position_heatmap_1, position_heatmap_2 = make_trajectory_heat_maps(whole_trajectory, trajectory_1, trajectory_2, number_of_bins_x, number_of_bins_y, prm)

    return rate_map_1, rate_map_2, position_heatmap_1, position_heatmap_2


def correlate_ratemaps(rate_map_first, rate_map_second, position_heatmap_1, position_heatmap_2):
    print('Correlate rate maps.')
    rate_map_first_flat = rate_map_first.flatten()
    rate_map_second_flat = rate_map_second.flatten()
    position_heatmap_1_flat = position_heatmap_1.flatten()
    position_heatmap_2_flat = position_heatmap_2.flatten()

    mask_for_nans_in_first = ~np.isnan(position_heatmap_1_flat)
    mask_for_nans_in_second = ~np.isnan(position_heatmap_2_flat)
    combined_mask = mask_for_nans_in_first & mask_for_nans_in_second

    rate_map_first_filtered = rate_map_first_flat[combined_mask]
    rate_map_second_filtered = rate_map_second_flat[combined_mask]

    pearson_r, p = pearsonr(rate_map_first_filtered, rate_map_second_filtered)
    percentage_of_excluded_bins = (len(rate_map_first_flat) - len(rate_map_first_filtered)) / len(rate_map_first_flat) * 100
    return pearson_r, percentage_of_excluded_bins


def calculate_spatial_correlation_between_rate_maps(first, second, position_first, position_second, prm):
    """
    This function accepts two sets of data (so for example first and second halves of the recording), makes a rate map
    for both halves in a way that the rate maps correspond, and correlates these rate maps to obtain a spatial correlation
    score. It will also return what percentage of rate map bins had to be excluded from the analysis due to no
    trajectory sampling.

    first : spatial firing data frame containing data for rate map # 1
    second: spatial firing data frame with data for rate map # 2

    position_first: position df for first rate map
    position_second: position df for second rate map
    """

    rate_map_first, rate_map_second, position_heatmap_1, position_heatmap_2 = make_same_sized_rate_maps(position_first, position_second, first, second, prm)
    # possibly need to remove nans here and maybe count how many there are and return that number as well
    pearson_r, percentage_of_excluded_bins = correlate_ratemaps(rate_map_first, rate_map_second, position_heatmap_1, position_heatmap_2)
    return pearson_r, percentage_of_excluded_bins, rate_map_first, rate_map_second

#todo tidy these
def plot_rate_map_comparison(grid_data, rate_map_1, rate_map_2, iterator, path):
    # corr = scipy.signal.correlate2d(rate_map_2, rate_map_2)
    corr = PostSorting.open_field_grid_cells.get_rate_map_autocorrelogram(rate_map_2)
    firing_rate_map_fig = plt.figure()
    firing_rate_map_fig.set_size_inches(5, 5, forward=True)
    ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax = plot_utility.style_open_field_plot(ax)
    rate_map_img = ax.imshow(corr, cmap='jet', interpolation='nearest')
    # firing_rate_map_fig.colorbar(rate_map_img)
    ax.axvline(x=(corr.shape[1]) / 2, color='black', linewidth=4)
    ax.axhline(y=(corr.shape[0]) / 2, color='black', linewidth=4)
    plt.savefig(
        path + grid_data.iloc[iterator].session_id + str(grid_data.iloc[iterator].session_id) + 'auto_corr2.png')

    plt.cla()


    corr = PostSorting.open_field_grid_cells.get_rate_map_autocorrelogram(rate_map_1)
    firing_rate_map_fig = plt.figure()
    firing_rate_map_fig.set_size_inches(5, 5, forward=True)
    ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax = plot_utility.style_open_field_plot(ax)
    rate_map_img = ax.imshow(corr, cmap='jet', interpolation='nearest')
    # firing_rate_map_fig.colorbar(rate_map_img)
    ax.axvline(x=(corr.shape[1]) / 2, color='black', linewidth=4)
    ax.axhline(y=(corr.shape[0]) / 2, color='black', linewidth=4)
    plt.savefig(
        path + grid_data.iloc[iterator].session_id + str(grid_data.iloc[iterator].session_id) + 'auto_corr1.png')

    plt.cla()

    corr = PostSorting.open_field_grid_cells.get_rate_map_crosscorrelogram(rate_map_1, rate_map_2)
    firing_rate_map_fig = plt.figure()
    firing_rate_map_fig.set_size_inches(5, 5, forward=True)
    ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax = plot_utility.style_open_field_plot(ax)
    rate_map_img = ax.imshow(corr, cmap='jet', interpolation='nearest')
    # firing_rate_map_fig.colorbar(rate_map_img)
    ax.axvline(x=(corr.shape[1]) / 2, color='black', linewidth=4)
    ax.axhline(y=(corr.shape[0]) / 2, color='black', linewidth=4)
    plt.savefig(
        path + grid_data.iloc[iterator].session_id + str(grid_data.iloc[iterator].session_id) + 'cross_corr.png')

    plt.cla()
    firing_rate_map_fig = plt.figure()
    firing_rate_map_fig.set_size_inches(5, 5, forward=True)
    ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax = plot_utility.style_open_field_plot(ax)
    rate_map_img = ax.imshow(rate_map_2 - rate_map_1, cmap='jet', interpolation='nearest')
    firing_rate_map_fig.colorbar(rate_map_img)
    plt.savefig(
        path + grid_data.iloc[iterator].session_id + str(grid_data.iloc[iterator].session_id) + 'subtract.png')
    plt.cla()

    plt.cla()
    firing_rate_map_fig = plt.figure()
    firing_rate_map_fig.set_size_inches(5, 5, forward=True)
    ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax = plot_utility.style_open_field_plot(ax)
    rate_map_img = ax.imshow(rate_map_1 - rate_map_2, cmap='jet', interpolation='nearest')
    firing_rate_map_fig.colorbar(rate_map_img)
    plt.savefig(
        path + grid_data.iloc[iterator].session_id + str(grid_data.iloc[iterator].session_id) + 'subtract2.png')
    plt.cla()


def main():
    trajectory_1 = []
    trajectory_2 = []
    spatial_firing_1 = []
    spatial_firing_2 = []
    make_same_sized_rate_maps(trajectory_1, trajectory_2, spatial_firing_1, spatial_firing_2)


if __name__ == '__main__':
    main()