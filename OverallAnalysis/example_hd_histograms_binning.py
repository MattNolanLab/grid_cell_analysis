import matplotlib.pylab as plt
import numpy as np
import OverallAnalysis.folder_path_settings
import pandas as pd
import plot_utility
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots


local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/example_hd_histograms/'
server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()
server_path_simulated = OverallAnalysis.folder_path_settings.get_server_path_simulated()


def plot_polar_head_direction_histogram(spike_hist, hd_hist, id, save_path):
    print('I will make the polar HD plots now.')

    hd_polar_fig = plt.figure()
    # hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    hd_hist_cluster = spike_hist
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    ax.plot(theta[:-1], hd_hist_cluster, color='red', linewidth=2)
    ax.plot(theta[:-1], hd_hist*(max(hd_hist_cluster)/max(hd_hist)), color='black', linewidth=2)
    # plt.tight_layout()
    max_firing_rate = np.max(hd_hist_cluster.flatten())
    plt.title(str(round(max_firing_rate, 2)) + 'Hz', y=1.08)
    #  + '\nKuiper p: ' + str(spatial_firing.hd_p[cluster])
    # plt.title('max fr: ' + str(round(spatial_firing.max_firing_rate_hd[cluster], 2)) + ' Hz' + ', preferred HD: ' + str(round(spatial_firing.preferred_HD[cluster][0], 0)) + ', hd score: ' + str(round(spatial_firing.hd_score[cluster], 2)), y=1.08, fontsize=12)
    plt.savefig(save_path + '/' + id + '_hd_polar_' + '.png', dpi=300)
    plt.close()


def plot_example_hd_histograms():
    position = pd.read_pickle(local_path + 'position.pkl')
    hd_pos = np.array(position.hd)
    hd_pos = (hd_pos + 180) * np.pi / 180

    spatial_firing = pd.read_pickle(local_path + 'spatial_firing.pkl')
    hd = np.array(spatial_firing.hd.iloc[0])
    hd = (hd + 180) * np.pi / 180
    hd_spike_histogram_23 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=23)
    hd_spike_histogram_10 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=10)
    hd_spike_histogram_20 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=20)
    hd_spike_histogram_30 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=30)
    hd_spike_histogram_40 = PostSorting.open_field_head_direction.get_hd_histogram(hd, window_size=40)

    hd_spike_histogram_23_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=23) / 30000  # 30000 is the ephys sampling rate for the mouse data
    hd_spike_histogram_10_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=10) / 30000
    hd_spike_histogram_20_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=20) / 30000
    hd_spike_histogram_30_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=30) / 30000
    hd_spike_histogram_40_pos = PostSorting.open_field_head_direction.get_hd_histogram(hd_pos, window_size=40) / 30000

    hd_spike_histogram_10_norm = hd_spike_histogram_10 / hd_spike_histogram_10_pos / 1000
    hd_spike_histogram_20_norm = hd_spike_histogram_20 / hd_spike_histogram_20_pos / 1000
    hd_spike_histogram_30_norm = hd_spike_histogram_30 / hd_spike_histogram_30_pos / 1000
    hd_spike_histogram_40_norm = hd_spike_histogram_40 / hd_spike_histogram_40_pos / 1000

    print('max rate:')
    max_firing_rate = np.max(hd_spike_histogram_10_norm.flatten())
    print(max_firing_rate)
    max_firing_rate = np.max(hd_spike_histogram_20_norm.flatten())
    print(max_firing_rate)
    max_firing_rate = np.max(hd_spike_histogram_30_norm.flatten())
    print(max_firing_rate)
    max_firing_rate = np.max(hd_spike_histogram_40_norm.flatten())
    print(max_firing_rate)

    plot_polar_head_direction_histogram(hd_spike_histogram_10_norm, hd_spike_histogram_10_pos, str(10), local_path)
    plot_polar_head_direction_histogram(hd_spike_histogram_20_norm, hd_spike_histogram_20_pos, str(20), local_path)
    plot_polar_head_direction_histogram(hd_spike_histogram_30_norm, hd_spike_histogram_30_pos, str(30), local_path)
    plot_polar_head_direction_histogram(hd_spike_histogram_40_norm, hd_spike_histogram_40_pos, str(40), local_path)


def main():
    plot_example_hd_histograms()


if __name__ == '__main__':
    main()
