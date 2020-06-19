import os


def get_list_of_false_positives(false_positives_path):
    if os.path.isfile(false_positives_path) is True:
        if os.stat(false_positives_path).st_size == 0:
            os.remove(false_positives_path)
    false_positive_reader = open(false_positives_path, 'r')
    false_positives = false_positive_reader.readlines()
    false_positive_clusters = list([x.strip() for x in false_positives])
    false_positive_clusters_stripped = (str.strip, false_positive_clusters)
    return false_positive_clusters_stripped[1]


def tag_false_positives(spike_df, false_positives_path):
    false_positives_list = get_list_of_false_positives(false_positives_path)
    spike_df['false_positive'] = spike_df['fig_name_id'].isin(false_positives_list)
    return spike_df


def add_figure_name_id(spike_df):
    # todo change order in data and put -  current format M9-10/04/2018-Tetrode-1-Cluster-4
    figure_name_ids = spike_df['animal'] + '-' + spike_df['day'].apply(str) + '-Tetrode-' + spike_df['tetrode'].apply(str) + '-Cluster-' + spike_df['cluster'].apply(str)
    spike_df['fig_name_id'] = figure_name_ids
    return spike_df


def get_accepted_clusters(spike_data_frame, false_positives_path):
    spike_data_frame = add_figure_name_id(spike_data_frame)
    spike_data_frame = tag_false_positives(spike_data_frame, false_positives_path)
    not_false_positive = spike_data_frame['false_positive'] == 0
    good_cluster = spike_data_frame['goodcluster'] == 1
    accepted_clusters = spike_data_frame[good_cluster & not_false_positive]
    return accepted_clusters


def get_false_positives(spike_data_frame, false_positives_path):
    spike_data_frame = add_figure_name_id(spike_data_frame)
    spike_data_frame = tag_false_positives(spike_data_frame, false_positives_path)
    false_positive = spike_data_frame['false_positive'] == 1
    good_cluster = spike_data_frame['goodcluster'] == 1
    false_positives = spike_data_frame[good_cluster & false_positive]
    return false_positives
