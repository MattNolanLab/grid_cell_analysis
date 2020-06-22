import cmocean
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import os
import plot_utility
import math
import numpy as np
import PostSorting.parameters
import PostSorting.open_field_head_direction

import pandas as pd
import PostSorting.open_field_firing_fields


def plot_position(position_data):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=5)
    plt.close()


def plot_spikes_on_trajectory(position_data, spike_data, prm):
    print('I will make scatter plots of spikes on the trajectory of the animal.')
    save_path = prm.get_output_path() + '/Figures/firing_scatters'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_id in range(len(spike_data)):
        cluster_id = spike_data.cluster_id.values[cluster_id] - 1
        spikes_on_track = plt.figure()
        spikes_on_track.set_size_inches(5, 5, forward=True)
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1, alpha=0.7)
        ax.scatter(spike_data.position_x[cluster_id], spike_data.position_y[cluster_id], color='red', marker='o', s=10, zorder=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            right=False,
            left=False,
            labelleft=False,
            labelbottom=False)  # labels along the bottom edge are off
        ax.set_aspect('equal')
        plt.title('Spikes on trajectory', y=1.08, fontsize=24)
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_id] + '_' + str(cluster_id + 1) + '_spikes_on_trajectory.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.savefig(save_path + '/' + spike_data.session_id[cluster_id] + '_' + str(cluster_id + 1) + '_spikes_on_trajectory.pdf', bbox_inches='tight')
        plt.close()


def plot_coverage(position_heat_map, prm):
    print('I will plot a heat map of the position of the animal to show coverage.')
    save_path = prm.get_output_path() + '/Figures/session'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    coverage = plt.figure()
    coverage.set_size_inches(5, 5, forward=True)
    ax = coverage.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax = plot_utility.style_open_field_plot(ax)
    position_heat_map = np.rot90(position_heat_map)
    coverage_fig = ax.imshow(position_heat_map, cmap=cmocean.cm.thermal, interpolation='nearest')
    coverage.colorbar(coverage_fig)
    plt.title('Coverage', y=1.08, fontsize=24)
    plt.savefig(save_path + '/heatmap.png', dpi=300)
    # plt.savefig(save_path + '/heatmap.pdf')
    plt.close()


def plot_firing_rate_maps(spatial_firing, prm):
    print('I will make rate map plots.')
    save_path = prm.get_output_path() + '/Figures/rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_rate_map_original = spatial_firing.firing_maps[cluster]
        firing_rate_map = np.rot90(firing_rate_map_original)
        firing_rate_map_fig = plt.figure()
        firing_rate_map_fig.set_size_inches(5, 5, forward=True)
        ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax = plot_utility.style_open_field_plot(ax)
        rate_map_img = ax.imshow(firing_rate_map, cmap='jet', interpolation='nearest')
        firing_rate_map_fig.colorbar(rate_map_img)
        plt.title('Firing rate map \n max fr: ' + str(round(spatial_firing.max_firing_rate[cluster], 2)) + ' Hz', y=1.08, fontsize=24)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_rate_map_' + str(cluster + 1) + '.png', dpi=300)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_rate_map_' + str(cluster + 1) + '.pdf')
        plt.close()


def plot_hd(spatial_firing, position_data, prm):
    print('I will plot HD on open field maps as a scatter plot for each cluster.')
    save_path = prm.get_output_path() + '/Figures/head_direction_plots_2d'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        x_positions = spatial_firing.position_x[cluster]
        y_positions = spatial_firing.position_y[cluster]
        hd = spatial_firing.hd[cluster]
        hd_map_fig = plt.figure()
        hd_map_fig.set_size_inches(5, 5, forward=True)
        ax = hd_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax = plot_utility.style_open_field_plot(ax)
        ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1,
                alpha=0.2)
        hd_plot = ax.scatter(x_positions, y_positions, s=20, c=hd, vmin=-180, vmax=180, marker='o', cmap=cmocean.cm.phase)
        plt.colorbar(hd_plot, fraction=0.046, pad=0.04)
        plt.title('Head direction at spikes', y=1.08, fontsize=24)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_polar_head_direction_histogram(hd_hist, spatial_firing, prm):
    print('I will make the polar HD plots now.')
    save_path = prm.get_output_path() + '/Figures/head_direction_plots_polar'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        hd_polar_fig = plt.figure()
        hd_polar_fig.set_size_inches(5, 5, forward=True)
        ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        hd_hist_cluster = spatial_firing.hd_spike_histogram[cluster]
        theta = np.linspace(0, 2*np.pi, 361)  # x axis
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        ax.plot(theta[:-1], hd_hist_cluster, color='red', linewidth=2)
        ax.plot(theta[:-1], hd_hist*(max(hd_hist_cluster)/max(hd_hist)), color='black', linewidth=2)
        plt.tight_layout()
        #  + '\nKuiper p: ' + str(spatial_firing.hd_p[cluster])
        plt.title('Head direction \n max fr: ' + str(round(spatial_firing.max_firing_rate_hd[cluster], 2)) + ' Hz' + ', hd score: ' + str(round(spatial_firing.hd_score[cluster], 2)) + '\n', y=1.08, fontsize=24)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_polar_' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_polar_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
        plt.close()


# plot polar hd histograms without needing the whole df as an input
def plot_polar_hd_hist(hist_1, hist_2, cluster, save_path, color1='lime', color2='navy', title=''):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    ax.plot(theta[:-1], hist_1, color=color1, linewidth=2)
    ax.plot(theta[:-1], hist_2, color=color2, linewidth=2)
    plt.title(title)
    # ax.plot(theta[:-1], hist_2 * (max(hist_1) / max(hist_2)), color='navy', linewidth=2)
    plt.tight_layout()
    plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
    # plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
    plt.close()


# plot polar hd histograms without needing the whole df as an input
def plot_single_polar_hd_hist(hist_1, cluster, save_path, color1='lime', title=''):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(0, np.nanmax(hist_1) * 1.4)
    plt.xticks([math.radians(0), math.radians(90), math.radians(180), math.radians(270)])
    ax.plot(theta[:-1], hist_1, color=color1, linewidth=6)
    plt.title(title)
    # ax.plot(theta[:-1], hist_2 * (max(hist_1) / max(hist_2)), color='navy', linewidth=2)
    plt.tight_layout()
    plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
    # plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
    plt.close()


def plot_rate_map_autocorrelogram(spatial_firing, prm):
    print('I will make the rate map autocorrelogram grid plots now.')
    save_path = prm.get_output_path() + '/Figures/rate_map_autocorrelogram'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        rate_map_autocorr_fig = plt.figure()
        rate_map_autocorr_fig.set_size_inches(5, 5, forward=True)
        ax = rate_map_autocorr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        rate_map_autocorr = spatial_firing.rate_map_autocorrelogram[cluster]
        if rate_map_autocorr.size:
            ax = plt.subplot(1, 1, 1)
            ax = plot_utility.style_open_field_plot(ax)
            autocorr_img = ax.imshow(rate_map_autocorr, cmap='jet', interpolation='nearest')
            rate_map_autocorr_fig.colorbar(autocorr_img)
            plt.tight_layout()
            plt.title('Autocorrelogram \n grid score: ' + str(round(spatial_firing.grid_score[cluster], 2)), fontsize=24)
            plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_rate_map_autocorrelogram_' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
            # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_rate_map_autocorrelogram_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
        plt.close()


def mark_firing_field_with_scatter(field, plot, colors, field_id, rate_map):
    y_max = rate_map.shape[0] -1
    for bin in field:
        plot.scatter(bin[0], y_max - bin[1], color=colors[field_id], marker='o', s=25)
    return plot


# generate more random colors if necessary
def generate_colors(number_of_firing_fields):
    colors = [[0, 1, 0], [1, 0.6, 0.3], [0, 1, 1], [1, 0, 1], [0.7, 0.3, 1], [0.6, 0.5, 0.4], [0.6, 0, 0]]  # green, orange, cyan, pink, purple, grey, dark red
    if number_of_firing_fields > len(colors):
        for i in range(number_of_firing_fields):
            colors.append(plot_utility.generate_new_color(colors, pastel_factor=0.9))
    return colors


def save_field_polar_plot(save_path, hd_hist_session, hd_hist_cluster, cluster, spatial_firing, colors, field_id, name):
    field_polar = plt.figure()
    field_polar.set_size_inches(5, 5, forward=True)
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    hd_plot_field = field_polar.add_subplot(1, 1, 1, polar=True)
    hd_plot_field = plot_utility.style_polar_plot(hd_plot_field)

    hd_plot_field.plot(theta[:-1], hd_hist_session*(max(hd_hist_cluster)/max(hd_hist_session)), color='black', linewidth=2, alpha=0.9)
    hd_plot_field.plot(theta[:-1], hd_hist_cluster, color=colors[field_id], linewidth=2)
    plt.tight_layout()
    if 'field_max_firing_rate' in spatial_firing:
        field_max_firing_rate = str(round(spatial_firing.field_max_firing_rate[cluster][field_id], 2))
    else:
        field_max_firing_rate = '?'

    plt.title(str(spatial_firing.number_of_spikes_in_fields[cluster][field_id]) + ' spikes'
              + ' in ' + str(round(spatial_firing.time_spent_in_fields_sampling_points[cluster][field_id]/30, 2))
              +' seconds\n max fr: ' + field_max_firing_rate + 'Hz \n',
              y=1.08, fontsize=24)

    plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_cluster_' + str(cluster + 1) + name + str(field_id + 1) + '.png', dpi=300, bbox_inches="tight")
    # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_cluster_' + str(cluster + 1) + name + str(field_id + 1) + '.pdf', bbox_inches="tight")
    plt.close()


def plot_hd_for_firing_fields(spatial_firing, spatial_data, prm):
    print('I will make the polar HD plots for individual firing fields now.')
    save_path = prm.get_output_path() + '/Figures/firing_field_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        if 'firing_fields' in spatial_firing:
            number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
            firing_rate_map = spatial_firing.firing_maps[cluster]
            if number_of_firing_fields > 0:
                plt.clf()
                of_figure = plt.figure()
                plt.title('HD in detected fields', fontsize=24)
                of_figure.set_size_inches(5, 5, forward=True)
                of_plot = of_figure.add_subplot(1, 1, 1)
                of_plot.axis('off')
                firing_rate_map_90 = np.rot90(firing_rate_map)
                of_plot.imshow(firing_rate_map_90)

                firing_fields_cluster = spatial_firing.firing_fields[cluster]
                colors = generate_colors(number_of_firing_fields)

                for field_id, field in enumerate(firing_fields_cluster):
                    of_plot = mark_firing_field_with_scatter(field, of_plot, colors, field_id, firing_rate_map_90)
                    hd_hist_session = spatial_firing.firing_fields_hd_session[cluster][field_id]
                    hd_hist_session = np.array(hd_hist_session) / prm.get_sampling_rate()
                    hd_hist_cluster = np.array(spatial_firing.firing_fields_hd_cluster[cluster][field_id])
                    hd_hist_cluster_normalized = np.divide(hd_hist_cluster, hd_hist_session, out=np.zeros_like(hd_hist_cluster), where=hd_hist_session != 0)

                    save_field_polar_plot(save_path, hd_hist_session, hd_hist_cluster_normalized, cluster, spatial_firing, colors, field_id, '_firing_field_')
                    # save_field_polar_plot(save_path, hd_hist_session, hd_hist_cluster, cluster, spatial_firing, colors, field_id, '_firing_field_raw')

                plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_firing_fields_rate_map' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
                # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_firing_fields_rate_map' + str(cluster + 1) + '.pdf', bbox_inches="tight")
                plt.close()


def plot_spikes_not_in_fields(spatial_firing, cluster, spatial_firing_cluster, of_plot):
    all_spikes_in_fields = np.hstack(np.array(spatial_firing.spike_times_in_fields[cluster]))
    mask_for_spikes_not_in_fields = ~np.in1d(spatial_firing.firing_times[cluster], all_spikes_in_fields)
    try:
        spike_times_not_in_fields = spatial_firing.firing_times[cluster][mask_for_spikes_not_in_fields]
    except:
        spike_times_not_in_fields = np.array(spatial_firing.firing_times[cluster])[mask_for_spikes_not_in_fields]
    not_in_fields_df = spatial_firing_cluster.loc[spatial_firing_cluster['firing_times'].isin(spike_times_not_in_fields)]
    of_plot.scatter(not_in_fields_df['x'].values, not_in_fields_df['y'].values, color='black', marker='o', s=6)


def make_df_for_cluster(spatial_firing, cluster):
    cluster_id = np.arange(len(spatial_firing.firing_times[cluster]))
    spatial_firing_cluster = pd.DataFrame(cluster_id)
    spatial_firing_cluster['x'] = spatial_firing.position_x_pixels[cluster]
    spatial_firing_cluster['y'] = spatial_firing.position_y_pixels[cluster]
    spatial_firing_cluster['hd'] = spatial_firing.hd[cluster]
    spatial_firing_cluster['firing_times'] = spatial_firing.firing_times[cluster]
    return spatial_firing_cluster

'''
Plot spikes on rate map colour coded to the [grid] field they belong to. This is only done for cells where fields
were detected.

'''


def plot_spikes_on_firing_fields(spatial_firing, prm):
    print('I will plot detected spikes colour coded in fields.')
    save_path = prm.get_output_path() + '/Figures/firing_fields_coloured_spikes'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        if 'firing_fields' in spatial_firing:
            number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
            if number_of_firing_fields > 0:
                plt.clf()
                of_figure = plt.figure()
                plt.title('spikes in fields')
                of_figure.set_size_inches(5, 5, forward=True)
                of_plot = of_figure.add_subplot(1, 1, 1)
                of_plot.axis('off')
                firing_fields_cluster = spatial_firing.firing_fields[cluster]
                colors = generate_colors(number_of_firing_fields)
                spatial_firing_cluster = make_df_for_cluster(spatial_firing, cluster)

                for field_id, field in enumerate(firing_fields_cluster):
                    spike_times_field = spatial_firing.spike_times_in_fields[cluster][field_id]
                    field_df = spatial_firing_cluster.loc[spatial_firing_cluster['firing_times'].isin(spike_times_field)]
                    of_plot.scatter(field_df['x'].values, field_df['y'].values, color=colors[field_id], marker='o', s=10)
                plot_spikes_not_in_fields(spatial_firing, cluster, spatial_firing_cluster, of_plot)

                plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_firing_fields_coloured_spikes' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
                # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_firing_fields_coloured_spikes' + str(cluster + 1) + '.pdf', bbox_inches="tight")
                plt.close()


def make_combined_figure(prm, spatial_firing):
    print('I will make the combined images now.')
    save_path = prm.get_output_path() + '/Figures/combined'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = prm.get_output_path() + '/Figures/'
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        coverage_path = figures_path + 'session/heatmap.png'
        spike_scatter_path = figures_path + 'firing_scatters/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spikes_on_trajectory.png'
        rate_map_path = figures_path + 'rate_maps/' + spatial_firing.session_id[cluster] + '_rate_map_' + str(cluster + 1) + '.png'
        head_direction_polar_path = figures_path + 'head_direction_plots_polar/' + spatial_firing.session_id[cluster] + '_hd_polar_' + str(cluster + 1) + '.png'
        head_direction_map_path = figures_path + 'head_direction_plots_2d/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.png'
        firing_fields_rate_map_path = figures_path + 'firing_field_plots/' + spatial_firing.session_id[cluster] + '_firing_fields_rate_map' + str(cluster + 1) + '.png'
        spike_histogram_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_histogram.png'
        speed_histogram_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_speed_histogram.png'
        firing_field_path = figures_path + 'firing_field_plots/' + spatial_firing.session_id[cluster] + '_cluster_' + str(cluster + 1) + '_firing_field_'
        autocorrelograms = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelograms.png'
        waveforms_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms.png'
        rate_map_autocorrelogram_path = figures_path + 'rate_map_autocorrelogram/' + spatial_firing.session_id[cluster] + '_rate_map_autocorrelogram_' + str(cluster + 1) + '.png'
        speed_vs_firing_rate_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_speed_vs_firing_rate.png'

        number_of_firing_fields = 0
        if 'firing_fields' in spatial_firing:
            number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
        number_of_rows = math.ceil((number_of_firing_fields + 1)/6) + 2

        grid = plt.GridSpec(number_of_rows, 5, wspace=0.025, hspace=0.05)
        if os.path.exists(waveforms_path):
            waveforms = mpimg.imread(waveforms_path)
            waveforms_plot = plt.subplot(grid[0, 0])
            waveforms_plot.axis('off')
            waveforms_plot.imshow(waveforms)
        if os.path.exists(spike_histogram_path):
            spike_hist = mpimg.imread(spike_histogram_path)
            spike_hist_plot = plt.subplot(grid[0, 2])
            spike_hist_plot.axis('off')
            spike_hist_plot.imshow(spike_hist)
        if os.path.exists(autocorrelograms):
            autocorrelogram_10 = mpimg.imread(autocorrelograms)
            autocorrelogram_10_plot = plt.subplot(grid[0, 1])
            autocorrelogram_10_plot.axis('off')
            autocorrelogram_10_plot.imshow(autocorrelogram_10)
        if os.path.exists(speed_vs_firing_rate_path):
            speed_vs_rate = mpimg.imread(speed_vs_firing_rate_path)
            speed_vs_rate_plot = plt.subplot(grid[0, 3])
            speed_vs_rate_plot.axis('off')
            speed_vs_rate_plot.imshow(speed_vs_rate)
        if os.path.exists(coverage_path):
            coverage = mpimg.imread(coverage_path)
            coverage_plot = plt.subplot(grid[0, 4])
            coverage_plot.axis('off')
            coverage_plot.imshow(coverage)
        if os.path.exists(spike_scatter_path):
            spike_scatter = mpimg.imread(spike_scatter_path)
            spike_scatter_plot = plt.subplot(grid[1, 0])
            spike_scatter_plot.axis('off')
            spike_scatter_plot.imshow(spike_scatter)
        if os.path.exists(rate_map_path):
            rate_map = mpimg.imread(rate_map_path)
            rate_map_plot = plt.subplot(grid[1, 1])
            rate_map_plot.axis('off')
            rate_map_plot.imshow(rate_map)
        if os.path.exists(rate_map_autocorrelogram_path):
            rate_map_autocorr = mpimg.imread(rate_map_autocorrelogram_path)
            rate_map_autocorr_plot = plt.subplot(grid[1, 2])
            rate_map_autocorr_plot.axis('off')
            rate_map_autocorr_plot.imshow(rate_map_autocorr)
        if os.path.exists(head_direction_polar_path):
            polar_hd = mpimg.imread(head_direction_polar_path)
            polar_hd_plot = plt.subplot(grid[1, 3])
            polar_hd_plot.axis('off')
            polar_hd_plot.imshow(polar_hd)
        if os.path.exists(head_direction_map_path):
            hd_map = mpimg.imread(head_direction_map_path)
            hd_map_plot = plt.subplot(grid[1, 4])
            hd_map_plot.axis('off')
            hd_map_plot.imshow(hd_map)
        if os.path.exists(firing_fields_rate_map_path):
            firing_fields = mpimg.imread(firing_fields_rate_map_path)
            firing_fields_plot = plt.subplot(grid[2, 0])
            firing_fields_plot.axis('off')
            firing_fields_plot.imshow(firing_fields)
        for field in range(number_of_firing_fields):
            path = firing_field_path + str(field + 1) + '.png'
            firing_field_polar = mpimg.imread(path)
            row = math.floor((field+1)/5) + 2
            col = (field+1) % 5
            firing_fields_polar_plot = plt.subplot(grid[row, col])
            firing_fields_polar_plot.axis('off')
            firing_fields_polar_plot.imshow(firing_field_polar)

        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '.png', dpi=1000)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '.pdf')
        plt.close()


def make_combined_field_analysis_figures(prm, spatial_firing):
    print('I will make the combined images for field analysis results now.')
    save_path = prm.get_output_path() + '/Figures/field_analysis'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = prm.get_output_path() + '/Figures/'
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        spike_scatter_path = figures_path + 'firing_scatters/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spikes_on_trajectory.png'
        rate_map_path = figures_path + 'rate_maps/' + spatial_firing.session_id[cluster] + '_rate_map_' + str(cluster + 1) + '.png'
        head_direction_polar_path = figures_path + 'head_direction_plots_polar/' + spatial_firing.session_id[cluster] + '_hd_polar_' + str(cluster + 1) + '.png'
        head_direction_map_path = figures_path + 'head_direction_plots_2d/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.png'
        firing_fields_rate_map_path = figures_path + 'firing_field_plots/' + spatial_firing.session_id[cluster] + '_firing_fields_rate_map' + str(cluster + 1) + '.png'
        firing_fields_coloured_spikes_path = figures_path + 'firing_fields_coloured_spikes/' + spatial_firing.session_id[cluster] + '_firing_fields_coloured_spikes' + str(cluster + 1) + '.png'
        firing_field_path = figures_path + 'firing_field_plots/' + spatial_firing.session_id[cluster] + '_cluster_' + str(cluster + 1) + '_firing_field_'
        firing_field_path_first = prm.get_output_path() + '/first_half/Figures/firing_field_plots/' + spatial_firing.session_id[cluster] + '_cluster_' + str(cluster + 1) + '_firing_field_'
        firing_field_path_second = prm.get_output_path() + '/second_half/Figures/firing_field_plots/' + spatial_firing.session_id[cluster] + '_cluster_' + str(cluster + 1) + '_firing_field_'

        number_of_firing_fields = 0
        if 'firing_fields' in spatial_firing:
            number_of_firing_fields = len(spatial_firing.firing_fields[cluster])
            if number_of_firing_fields == 0:
                continue
        number_of_rows = 4
        number_of_columns = 5
        if number_of_firing_fields > 5:
            number_of_columns = number_of_firing_fields
        grid = plt.GridSpec(number_of_rows, number_of_columns, wspace=0.2, hspace=0.2)
        rounded_r = [ '%.4f' % elem for elem in spatial_firing.field_corr_r[cluster]]
        rounded_p = [ '%.4f' % elem for elem in spatial_firing.field_corr_p[cluster]]
        plt.suptitle("r: " + str(rounded_r) + '\np: ' + str(rounded_p))
        if os.path.exists(spike_scatter_path):
            spike_scatter = mpimg.imread(spike_scatter_path)
            spike_scatter_plot = plt.subplot(grid[0, 0])
            spike_scatter_plot.axis('off')
            spike_scatter_plot.imshow(spike_scatter)
        if os.path.exists(head_direction_polar_path):
            polar_hd = mpimg.imread(head_direction_polar_path)
            polar_hd_plot = plt.subplot(grid[0, 1])
            polar_hd_plot.axis('off')
            polar_hd_plot.imshow(polar_hd)
        if os.path.exists(head_direction_map_path):
            hd_map = mpimg.imread(head_direction_map_path)
            hd_map_plot = plt.subplot(grid[0, 2])
            hd_map_plot.axis('off')
            hd_map_plot.imshow(hd_map)
        if os.path.exists(firing_fields_rate_map_path):
            firing_fields = mpimg.imread(firing_fields_rate_map_path)
            firing_fields_plot = plt.subplot(grid[0, 3])
            firing_fields_plot.axis('off')
            firing_fields_plot.imshow(firing_fields)
        if os.path.exists(firing_fields_coloured_spikes_path):
            firing_fields = mpimg.imread(firing_fields_coloured_spikes_path)
            firing_fields_plot = plt.subplot(grid[0, 4])
            firing_fields_plot.axis('off')
            firing_fields_plot.imshow(firing_fields)

        for field in range(number_of_firing_fields):
            path = firing_field_path + str(field + 1) + '.png'
            firing_field_polar = mpimg.imread(path)
            row = 1
            col = field
            firing_fields_polar_plot = plt.subplot(grid[row, col])
            firing_fields_polar_plot.axis('off')
            firing_fields_polar_plot.imshow(firing_field_polar)

        for field in range(number_of_firing_fields):
            path = firing_field_path_first + str(field + 1) + '.png'
            firing_field_polar = mpimg.imread(path)
            row = 2
            col = field
            firing_fields_polar_plot = plt.subplot(grid[row, col])
            firing_fields_polar_plot.axis('off')
            firing_fields_polar_plot.imshow(firing_field_polar)

        for field in range(number_of_firing_fields):
            path = firing_field_path_second + str(field + 1) + '.png'
            firing_field_polar = mpimg.imread(path)
            row = 3
            col = field
            firing_fields_polar_plot = plt.subplot(grid[row, col])
            firing_fields_polar_plot.axis('off')
            firing_fields_polar_plot.imshow(firing_field_polar)

        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '.png', dpi=1000)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '.pdf', dpi=1000)
        plt.close()

