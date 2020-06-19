import numpy as np
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots
import PostSorting.open_field_firing_maps
import PostSorting.post_process_sorted_data
from scipy.stats.stats import pearsonr
import pandas as pd


def get_data_from_data_frame_for_cluster(spike_data, cluster, indices):
    spike_data_cluster = pd.DataFrame()
    spike_data_cluster['firing_times'] = spike_data.firing_times[cluster][indices].copy()
    spike_data_cluster['position_x'] = np.array(spike_data.position_x[cluster])[indices].copy()
    spike_data_cluster['position_y'] = np.array(spike_data.position_y[cluster])[indices].copy()
    spike_data_cluster['position_x_pixels'] = np.array(spike_data.position_x_pixels[cluster])[indices].copy()
    spike_data_cluster['position_y_pixels'] = np.array(spike_data.position_y_pixels[cluster])[indices].copy()
    spike_data_cluster['hd'] = np.array(spike_data.hd[cluster])[indices].copy()
    return spike_data_cluster


def get_data_from_data_frames_fields(spike_data, spike_data_cluster, synced_spatial_data, cluster, end_of_first_half_seconds, end_of_first_half_ephys_sampling_points, half='first'):
    number_of_firing_fields = len(spike_data.firing_fields[cluster])
    number_of_spikes_in_fields = []
    number_of_samples_in_fields = []
    hd_in_fields_cluster = []
    hd_in_field_sessions = []
    if number_of_firing_fields > 0:
        firing_field_spike_times = spike_data.spike_times_in_fields[cluster]
        firing_field_times_session = spike_data.times_in_session_fields[cluster]
        for field_id, field in enumerate(firing_field_spike_times):
            if half == 'first':
                firing_times_field = np.take(field, np.where(field < end_of_first_half_ephys_sampling_points))
                mask_firing_times_in_field = np.in1d(spike_data_cluster.firing_times, firing_times_field)

            else:
                firing_times_field = np.take(field, np.where(field >= end_of_first_half_ephys_sampling_points))
                mask_firing_times_in_field = np.in1d(spike_data_cluster.firing_times, firing_times_field)
            number_of_spikes_field = mask_firing_times_in_field.sum()
            hd_field_cluster = spike_data_cluster.hd[mask_firing_times_in_field]
            hd_field_cluster = (np.array(hd_field_cluster) + 180) * np.pi / 180
            hd_fields_cluster_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_field_cluster)
            number_of_spikes_in_fields.append(number_of_spikes_field)
            hd_in_fields_cluster.append(hd_fields_cluster_hist)

        for field_id, field in enumerate(firing_field_times_session):
            if half == 'first':
                times_field = np.take(field, np.where(field < end_of_first_half_seconds))
                mask_times_in_field = np.in1d(synced_spatial_data.synced_time, times_field)
            else:
                times_field = np.take(field, np.where(field >= end_of_first_half_seconds))
                mask_times_in_field = np.in1d(synced_spatial_data.synced_time, times_field)
            amount_of_time_spent_in_field = mask_times_in_field.sum()
            hd_field = synced_spatial_data.hd[mask_times_in_field]
            hd_field = (np.array(hd_field) + 180) * np.pi / 180
            number_of_samples_in_fields.append(amount_of_time_spent_in_field)
            hd_field_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_field)
            hd_in_field_sessions.append(hd_field_hist)

    else:
        number_of_spikes_in_fields.append(None)
        number_of_samples_in_fields.append(None)
        hd_in_fields_cluster.append([None])
        hd_in_field_sessions.append([None])

    spike_data.at[cluster, 'number_of_spikes_in_fields'] = number_of_spikes_in_fields
    spike_data.at[cluster, 'time_spent_in_fields_sampling_points'] = number_of_samples_in_fields
    spike_data.at[cluster, 'firing_fields_hd_cluster'] = hd_in_fields_cluster
    spike_data.at[cluster, 'firing_fields_hd_session'] = hd_in_field_sessions

    return spike_data


def get_half_of_the_data(prm, spike_data_in, synced_spatial_data_in, half='first_half'):
    spike_data = spike_data_in.copy()
    synced_spatial_data = synced_spatial_data_in.copy()
    synced_spatial_data_half = None
    spike_data_half = None
    end_of_first_half_seconds = (synced_spatial_data.synced_time.max() - synced_spatial_data.synced_time.min()) / 2
    end_of_first_half_ephys_sampling_points = end_of_first_half_seconds * 30000

    if half == 'first_half':
        first_half_synced_data_indices = synced_spatial_data.synced_time < end_of_first_half_seconds
        synced_spatial_data_half = synced_spatial_data[first_half_synced_data_indices].copy()
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            firing_times_first_half = spike_data.firing_times[cluster] < end_of_first_half_ephys_sampling_points
            spike_data_cluster = get_data_from_data_frame_for_cluster(spike_data, cluster, firing_times_first_half)
            spike_data = get_data_from_data_frames_fields(spike_data, spike_data_cluster, synced_spatial_data, cluster, end_of_first_half_seconds, end_of_first_half_ephys_sampling_points, half='first')

        spike_data_half = spike_data[['cluster_id', 'session_id', 'firing_times', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'number_of_spikes_in_fields', 'time_spent_in_fields_sampling_points', 'firing_fields_hd_cluster', 'firing_fields_hd_session', 'firing_fields']].copy()
    if half == 'second_half':
        second_half_synced_data_indices = synced_spatial_data.synced_time >= end_of_first_half_seconds
        synced_spatial_data_half = synced_spatial_data[second_half_synced_data_indices]
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            firing_times_second_half = spike_data.firing_times[cluster] >= end_of_first_half_ephys_sampling_points
            spike_data_cluster = get_data_from_data_frame_for_cluster(spike_data, cluster, firing_times_second_half)
            spike_data = get_data_from_data_frames_fields(spike_data, spike_data_cluster, synced_spatial_data, cluster, end_of_first_half_seconds, end_of_first_half_ephys_sampling_points, half='second')
        spike_data_half = spike_data[['cluster_id', 'session_id', 'firing_times', 'position_x', 'position_x_pixels', 'position_y', 'position_y_pixels', 'hd', 'number_of_spikes_in_fields', 'time_spent_in_fields_sampling_points', 'firing_fields_hd_cluster', 'firing_fields_hd_session','firing_fields']].copy()
    return spike_data_half, synced_spatial_data_half


def get_half_of_the_data_cell(prm, spike_data_in, synced_spatial_data_in, half='first_half'):
    spike_data = spike_data_in.copy()
    synced_spatial_data = synced_spatial_data_in.copy()
    synced_spatial_data_half = None
    spike_data_half = None
    end_of_first_half_seconds = (synced_spatial_data.synced_time.max() - synced_spatial_data.synced_time.min()) / 2
    end_of_first_half_ephys_sampling_points = end_of_first_half_seconds * prm.get_sampling_rate()

    if half == 'first_half':
        first_half_synced_data_indices = synced_spatial_data.synced_time < end_of_first_half_seconds
        synced_spatial_data_half = synced_spatial_data[first_half_synced_data_indices].copy()
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            firing_times_first_half = spike_data.firing_times[cluster] < end_of_first_half_ephys_sampling_points
            spike_data_cluster = get_data_from_data_frame_for_cluster(spike_data, cluster, firing_times_first_half)

    if half == 'second_half':
        second_half_synced_data_indices = synced_spatial_data.synced_time >= end_of_first_half_seconds
        synced_spatial_data_half = synced_spatial_data[second_half_synced_data_indices]
        for cluster in range(len(spike_data)):
            cluster = spike_data.cluster_id.values[cluster] - 1
            firing_times_second_half = spike_data.firing_times[cluster] >= end_of_first_half_ephys_sampling_points
            spike_data_cluster = get_data_from_data_frame_for_cluster(spike_data, cluster, firing_times_second_half)
    return spike_data_cluster, synced_spatial_data_half



'''
slope : slope of the regression line
intercept : intercept of the regression line
r-value : correlation coefficient
p-value : two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero
stderr : Standard error of the estimate
'''


def correlate_hd_in_fields_in_two_halves(first_half, second_half, spike_data):
    print('I will now correlate the first and second halves of the recording [fields are analyzed].')
    pearson_rs = []
    ps = []
    for cluster in range(len(first_half)):
        pearson_rs_clu = []
        ps_clu = []
        cluster = first_half.cluster_id.values[cluster] - 1
        number_of_firing_fields = len(first_half.firing_fields[cluster])
        hd_in_fields_first = first_half.firing_fields_hd_cluster[cluster]
        if number_of_firing_fields > 0:
            for field_id, field in enumerate(hd_in_fields_first):
                hd_in_fields_first_session = first_half.firing_fields_hd_session[cluster][field_id]
                hd_in_fields_first_norm = np.divide(field, hd_in_fields_first_session, out=np.zeros_like(field), where=hd_in_fields_first_session != 0)
                hd_in_fields_second = second_half.firing_fields_hd_cluster[cluster][field_id]
                hd_in_fields_second_session = second_half.firing_fields_hd_session[cluster][field_id]
                hd_in_fields_second_norm = np.divide(hd_in_fields_second, hd_in_fields_second_session, out=np.zeros_like(hd_in_fields_second), where=hd_in_fields_second_session != 0)
                pearson_r, p = pearsonr(hd_in_fields_first_norm, hd_in_fields_second_norm)
                pearson_rs_clu.append(pearson_r)
                ps_clu.append(p)
        else:
            pearson_rs_clu.append([None])
            ps_clu.append([None])
        pearson_rs.append(pearson_rs_clu)
        ps.append(ps_clu)

    spike_data['field_corr_r'] = pearson_rs
    spike_data['field_corr_p'] = ps

    return spike_data


def get_hd_hists_for_data_frame(spike_data_frame, synced_spatial_data):
    print('I will now correlate the first and second halves of the recording [the whole session is analyzed].')
    hd_spike_histograms = []
    hd_session = synced_spatial_data.hd
    hd_session_rad = (np.array(hd_session) + 180) * np.pi / 180
    hd_session_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_session_rad)

    for cluster in range(len(spike_data_frame)):
        cluster = spike_data_frame.cluster_id.values[cluster] - 1
        hd_spike = spike_data_frame.hd[cluster]
        hd_spike_rad = (np.array(hd_spike) + 180) * np.pi / 180
        hd_spike_hist = PostSorting.open_field_head_direction.get_hd_histogram(hd_spike_rad)
        hd_spike_hist_normalized = hd_spike_hist / hd_session_hist
        hd_spike_histograms.append(hd_spike_hist_normalized)

    spike_data_frame['hd_spike_histogram'] = hd_spike_histograms
    return spike_data_frame


def correlate_hd_for_session(first_half, second_half, spike_data):
    pearson_rs = []
    ps = []
    hd_hists_first_half = []
    hd_hists_second_half = []
    for cluster in range(len(spike_data)):
        cluster = spike_data.cluster_id.values[cluster] - 1
        hd_first_half = first_half.hd_spike_histogram[cluster]
        hd_second_half = second_half.hd_spike_histogram[cluster]
        pearson_r, p = pearsonr(hd_first_half, hd_second_half)
        pearson_rs.append(pearson_r)
        hd_hists_first_half.append(hd_first_half)
        hd_hists_second_half.append(hd_second_half)
        ps.append(p)
    spike_data['hd_correlation_first_vs_second_half'] = pearson_rs
    spike_data['hd_correlation_first_vs_second_half_p'] = ps
    spike_data['hd_hist_first_half'] = hd_hists_first_half
    spike_data['hd_hist_second_half'] = hd_hists_second_half
    return spike_data


def analyse_first_and_second_halves(prm, synced_spatial_data, spike_data_in):
    print('---------------------------------------------------------------------------')
    print('I will get data from the first half of the recording.')
    prm.set_output_path(prm.get_filepath() + '/' + prm.get_sorter_name() + '/first_half')
    spike_data_first, synced_spatial_data_first = get_half_of_the_data(prm, spike_data_in, synced_spatial_data, half='first_half')
    position_heat_map, spike_data_first = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_first, prm)
    spike_data_first = get_hd_hists_for_data_frame(spike_data_first, synced_spatial_data_first)
    PostSorting.post_process_sorted_data.save_data_frames(spike_data_first, synced_spatial_data_first, bad_clusters=None)
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spike_data_first, synced_spatial_data_first, prm)
    print('---------------------------------------------------------------------------')
    print('I will get data from the second half of the recording.')
    spike_data_second, synced_spatial_data_second = get_half_of_the_data(prm, spike_data_in, synced_spatial_data, half='second_half')
    position_heat_map, spike_data_second = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_second, prm)
    spike_data_second = get_hd_hists_for_data_frame(spike_data_second, synced_spatial_data_second)
    prm.set_output_path(prm.get_filepath() + '/' + prm.get_sorter_name() + '/second_half')
    PostSorting.post_process_sorted_data.save_data_frames(spike_data_second, synced_spatial_data_second, bad_clusters=None)
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spike_data_second, synced_spatial_data_second, prm)

    spike_data = correlate_hd_in_fields_in_two_halves(spike_data_first, spike_data_second, spike_data_in)
    spike_data = correlate_hd_for_session(spike_data_first, spike_data_second, spike_data)
    prm.set_output_path(prm.get_filepath() + '/' + prm.get_sorter_name())
    PostSorting.open_field_make_plots.make_combined_field_analysis_figures(prm, spike_data)
    return spike_data
