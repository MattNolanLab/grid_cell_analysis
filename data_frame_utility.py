import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction
import numpy as np


# source: https://stackoverflow.com/users/48956/user48956
def df_empty(columns, dtypes, index=None):
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def append_field_to_data_frame(field_df, session_id, cluster_id, field_id, indices_rate_map, spike_times, number_of_spikes_in_field, position_x_spikes, position_y_spikes, hd_in_field_spikes, hd_hist_spikes, times_session, time_spent_in_field, position_x_session, position_y_session, hd_in_field_session, hd_hist_session, hd_score, grid_score, grid_spacing, field_size):
    field_df = field_df.append({
        "session_id": session_id,
        "cluster_id":  cluster_id,
        "field_id": field_id,
        "indices_rate_map": indices_rate_map,
        "spike_times": spike_times,
        "number_of_spikes_in_field": number_of_spikes_in_field,
        "position_x_spikes": position_x_spikes,
        "position_y_spikes": position_y_spikes,
        "hd_in_field_spikes": hd_in_field_spikes,
        "hd_hist_spikes": hd_hist_spikes,
        "times_session": times_session,
        "time_spent_in_field": time_spent_in_field,
        "position_x_session": position_x_session,
        "position_y_session": position_y_session,
        "hd_in_field_session": hd_in_field_session,
        "hd_hist_session": hd_hist_session,
        "hd_score": hd_score,
        "grid_score": grid_score,
        "grid_spacing": grid_spacing,
        "field_size": field_size
    }, ignore_index=True)
    return field_df


def get_field_data_frame(spatial_firing, position_data):
    field_df = pd.DataFrame(columns=['session_id', 'cluster_id', 'field_id', 'indices_rate_map', 'spike_times', 'number_of_spikes_in_field', 'position_x_spikes', 'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes', 'times_session', 'time_spent_in_field', 'position_x_session', 'position_y_session', 'hd_in_field_session', 'hd_hist_session', 'hd_score', 'grid_score', 'grid_spacing', 'field_size'])
    for index, cluster in spatial_firing.iterrows():
        cluster_id = spatial_firing.cluster_id[index]
        session_id = spatial_firing.session_id[index]
        number_of_firing_fields = len(spatial_firing.firing_fields[index])
        if number_of_firing_fields > 0:
            firing_field_spike_times = spatial_firing.spike_times_in_fields[index]
            for field_id, field in enumerate(firing_field_spike_times):
                indices_rate_map = spatial_firing.firing_fields[index][field_id]
                mask_firing_times_in_field = np.in1d(spatial_firing.firing_times[index], field)
                spike_times = field
                number_of_spikes_in_field = len(field)
                position_x_spikes = np.array(spatial_firing.position_x_pixels[index])[mask_firing_times_in_field]
                position_y_spikes = np.array(spatial_firing.position_y_pixels[index])[mask_firing_times_in_field]
                hd_in_field_spikes = np.array(spatial_firing.hd[index])[mask_firing_times_in_field]
                hd_in_field_spikes = (np.array(hd_in_field_spikes) + 180) * np.pi / 180
                hd_hist_spikes = PostSorting.open_field_head_direction.get_hd_histogram(hd_in_field_spikes)

                times_session = spatial_firing.times_in_session_fields[index][field_id]
                time_spent_in_field = len(times_session)
                mask_times_in_field = np.in1d(position_data.synced_time, times_session)
                position_x_session = position_data.position_x_pixels.values[mask_times_in_field]
                position_y_session = position_data.position_y_pixels.values[mask_times_in_field]
                hd_in_field_session = position_data.hd.values[mask_times_in_field]
                hd_in_field_session = (np.array(hd_in_field_session) + 180) * np.pi / 180
                hd_hist_session = PostSorting.open_field_head_direction.get_hd_histogram(hd_in_field_session)
                hd_score = cluster.hd_score
                if 'grid_score' in spatial_firing:
                    grid_score = cluster.grid_score
                    grid_spacing = cluster.grid_spacing
                    field_size = cluster.field_size
                else:
                    grid_score = np.nan
                    grid_spacing = np.nan
                    field_size = np.nan

                field_df = append_field_to_data_frame(field_df, session_id, cluster_id, field_id, indices_rate_map, spike_times, number_of_spikes_in_field, position_x_spikes, position_y_spikes, hd_in_field_spikes, hd_hist_spikes, times_session, time_spent_in_field, position_x_session, position_y_session, hd_in_field_session, hd_hist_session, hd_score, grid_score, grid_spacing, field_size)
    return field_df


def main():
    spatial_firing = pd.read_pickle(OverallAnalysis.folder_path_settings.get_local_test_recording_path() + 'DataFrames/spatial_firing.pkl')
    position_data = pd.read_pickle(OverallAnalysis.folder_path_settings.get_local_test_recording_path() + 'DataFrames/position.pkl')
    get_field_data_frame(spatial_firing, position_data)


if __name__ == '__main__':
    main()