import glob
import os
import pandas as pd
import PostSorting.open_field_firing_field_detection
import PostSorting.open_field_firing_fields
import PostSorting.open_field_make_plots

import matplotlib.pylab as plt


# load data
def load_data_frame(path):
    spike_data_frame_path = path + '/spatial_firing.pkl'
    spike_data = pd.read_pickle(spike_data_frame_path)
    return spike_data


def plot_fields_on_rate_map(save_path, rate_map, fields, name):
    number_of_firing_fields = len(fields)
    colors = PostSorting.open_field_make_plots.generate_colors(number_of_firing_fields)
    of_figure = plt.figure()
    of_plot = of_figure.add_subplot(1, 1, 1)
    of_plot.axis('off')
    of_plot.imshow(rate_map)
    for field_id, field in enumerate(fields[0]):
        of_plot = PostSorting.open_field_make_plots.mark_firing_field_with_scatter(field, of_plot, colors, field_id)
    plt.savefig(save_path + '/detection_results_' + name + '.png')
    plt.close()


def call_detector_original(spike_data, cluster, save_path):
    firing_fields, max_firing_rates = PostSorting.open_field_firing_fields.analyze_fields_in_cluster(spike_data, cluster, threshold=20)
    plot_fields_on_rate_map(save_path, spike_data.firing_maps[cluster], firing_fields, '01Hz_20')


def call_detector_modified_params(spike_data, cluster, save_path):
    firing_fields, max_firing_rates = PostSorting.open_field_firing_fields.analyze_fields_in_cluster(spike_data, cluster, threshold=30)
    plot_fields_on_rate_map(save_path, spike_data.firing_maps[cluster], firing_fields, '01Hz_30')

    firing_fields, max_firing_rates = PostSorting.open_field_firing_fields.analyze_fields_in_cluster(spike_data, cluster, threshold=35)
    plot_fields_on_rate_map(save_path, spike_data.firing_maps[cluster], firing_fields, '01Hz_35')

    firing_fields, max_firing_rates = PostSorting.open_field_firing_fields.analyze_fields_in_cluster(spike_data, cluster, threshold=40)
    plot_fields_on_rate_map(save_path, spike_data.firing_maps[cluster], firing_fields, '01Hz_40')

    firing_fields, max_firing_rates = PostSorting.open_field_firing_fields.analyze_fields_in_cluster(spike_data, cluster, threshold=50)
    plot_fields_on_rate_map(save_path, spike_data.firing_maps[cluster], firing_fields, '01Hz_50')


def call_detector_gauss(spike_data, cluster, save_path):
    PostSorting.open_field_firing_field_detection.detect_firing_fields(spike_data, cluster, save_path)


def get_clusters_to_analyse(path):
    cluster_ids_path = path + '/cluster.txt'
    cluster_ids = []
    with open(cluster_ids_path) as cluster_id_file:
        for line in cluster_id_file:
            cluster_ids.append(int(line) - 1)
    return cluster_ids


def compare_field_detection_methods():
    folder_path = 'C:/Users/s1466507/Documents/Ephys/field_detection_test'
    for name in glob.glob(folder_path + '/*'):
        if os.path.isdir(name):
            spike_data = load_data_frame(name)
            cluster_ids = get_clusters_to_analyse(name)
            for cluster in range(len(cluster_ids)):
                call_detector_gauss(spike_data, cluster_ids[cluster], name)
                call_detector_original(spike_data, cluster_ids[cluster], name)
                call_detector_modified_params(spike_data, cluster_ids[cluster], name)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('I will compare different field detection methods on a test dataset.')

    compare_field_detection_methods()


if __name__ == '__main__':
    main()










