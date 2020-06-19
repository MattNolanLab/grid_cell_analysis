# calculate number of spikes and mean firing rate for each cluster and add to spatial firing df
def add_temporal_firing_properties_to_df(spatial_firing, prm):
    total_number_of_spikes_per_cluster = []
    mean_firing_rates = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_times = spatial_firing.firing_times[cluster]
        total_number_of_spikes = len(firing_times)
        total_length_of_recordings = prm.get_total_length_sampling_points()  # this does not include opto
        mean_firing_rate = total_number_of_spikes / total_length_of_recordings

        total_number_of_spikes_per_cluster.append(total_number_of_spikes)
        mean_firing_rates.append(mean_firing_rate)

    spatial_firing['number_of_spikes'] = total_number_of_spikes_per_cluster
    spatial_firing['mean_firing_rate'] = mean_firing_rates

    return spatial_firing
