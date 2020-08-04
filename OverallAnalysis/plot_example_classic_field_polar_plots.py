import math
import numpy as np
import pandas as pd
import plot_utility
import matplotlib.pylab as plt
import OverallAnalysis.folder_path_settings
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots


local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/plot_hd_tuning_vs_shuffled_fields/'
output_path = local_path + '/example_fields_classic/'


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
    # plt.xticks([math.radians(0), math.radians(90), math.radians(180), math.radians(270)])
    ax.plot(theta[:-1], hist_1, color=color1, linewidth=12)
    plt.title(title)
    # ax.plot(theta[:-1], hist_2 * (max(hist_1) / max(hist_2)), color='navy', linewidth=2)
    plt.tight_layout()
    plt.savefig(save_path + '_hd_polar_' + cluster + '.png', dpi=300, bbox_inches="tight")
    # plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
    plt.close()


def make_example_plots_mouse():
    mouse_df_path = analysis_path + 'shuffled_field_data_all_mice.pkl'
    mouse_df = pd.read_pickle(mouse_df_path)
    session_id = 'M12_2018-04-10_14-22-14_of'
    example_session = mouse_df.session_id == session_id
    example_cell = mouse_df[example_session]
    colors = PostSorting.open_field_make_plots.generate_colors(len(example_cell))
    for index, field in example_cell.iterrows():
        hd_session = field.hd_in_field_session
        hd_session_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_session)
        hd_spikes = field.hd_in_field_spikes
        hd_spikes_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_spikes)
        hist = hd_spikes_hist / hd_session_hist
        plot_single_polar_hd_hist(hist, 'mouse_' + str(field.field_id), output_path, color1=colors[field.field_id], title='')


def make_example_plots_rat():
    rat_df_path = analysis_path + 'shuffled_field_data_all_rats.pkl'
    rat_df = pd.read_pickle(rat_df_path)
    session_id = '11207-06070501+02'
    example_session = rat_df.session_id == session_id
    example_cell = rat_df[example_session]
    example_cluster = example_cell.cluster_id == 2
    example_cell = example_cell[example_cluster]
    colors = PostSorting.open_field_make_plots.generate_colors(len(example_cell))
    for index, field in example_cell.iterrows():
        hd_session = field.hd_in_field_session
        hd_session_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_session)
        hd_spikes = field.hd_in_field_spikes
        hd_spikes_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_spikes)
        hist = hd_spikes_hist / hd_session_hist
        plot_single_polar_hd_hist(hist, 'rat_' + str(field.field_id), output_path, color1=colors[field.field_id], title='')


def main():
    """
    Make example classic hd plots for example fields (Figure 4a)
    """
    make_example_plots_rat()
    make_example_plots_mouse()


if __name__ == '__main__':
    main()
