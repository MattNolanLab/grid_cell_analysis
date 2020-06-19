import pandas as pd
import OverallAnalysis.false_positives
import PostSorting.open_field_make_plots
import PostSorting.open_field_head_direction

save_output_path = '/watson_two_test_cells/'
local_path = 'watson_two_test_cells/all_mice_df.pkl'
false_positives_path = '/watson_two_test_cells/false_positives_all.txt'


def load_data_frame(path):
    # this is the output of load_df.py which read all dfs from a folder and saved selected columns into a combined df
    df = pd.read_pickle(path)
    return df


def add_combined_id_to_df(df_all_mice):
    animal_ids = [session_id.split('_')[0] for session_id in df_all_mice.session_id.values]
    dates = [session_id.split('_')[1] for session_id in df_all_mice.session_id.values]
    tetrode = df_all_mice.tetrode.values
    cluster = df_all_mice.cluster_id.values

    combined_ids = []
    for cell in range(len(df_all_mice)):
        id = animal_ids[cell] + '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    df_all_mice['false_positive_id'] = combined_ids
    return df_all_mice


def plot_hd_for_example_cells(name, cluster):
    path = save_output_path + name
    spatial_firing = pd.read_pickle(path + '/spatial_firing.pkl')
    position = pd.read_pickle(path + '/position.pkl')
    hd_hist_cluster = spatial_firing.hd_spike_histogram[cluster]
    hd_position = position.hd
    hd_hist_position = PostSorting.open_field_head_direction.get_hd_histogram(hd_position)
    hd_hist_position = hd_hist_position * max(hd_hist_cluster)/max(hd_hist_position)

    PostSorting.open_field_make_plots.plot_polar_hd_hist(hd_hist_cluster, hd_hist_cluster, cluster, path + '/' + str(cluster), color1='red', color2='red')
    PostSorting.open_field_make_plots.plot_polar_hd_hist(spatial_firing.iloc[0].hd_hist_first_half, spatial_firing.iloc[0].hd_hist_second_half, 1, save_output_path + 'first_vs_second_half' + name)


def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    df_all_mice = load_data_frame(local_path)
    list_of_false_positives = OverallAnalysis.false_positives.get_list_of_false_positives(false_positives_path)
    df_all_mice = add_combined_id_to_df(df_all_mice)
    df_all_mice['false_positive'] = df_all_mice['false_positive_id'].isin(list_of_false_positives)

    good_cluster = df_all_mice.false_positive == False

    plot_hd_for_example_cells('M13_2018-05-14_09-37-33_of', 6)


if __name__ == '__main__':
    main()
