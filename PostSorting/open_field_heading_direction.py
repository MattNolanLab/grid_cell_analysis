import math_utility
import numpy as np
import pandas as pd
import PostSorting.open_field_spatial_firing
import data_frame_utility


def calculate_heading_direction(position_x, position_y, pad_first_value=True):
    """
    Calculate heading direction of animal based on the central position of the tracking markers.
    Method from:
    https://doi.org/10.1016/j.brainres.2014.10.053

    input : position_x and position_y of the animal (arrays)
            pad_first_value - if True, the first value will be repeated so the output array's shape is the
            same as the input
    output : heading direction of animal
    based on the vector from consecutive samples
    """

    delta_x = np.diff(position_x)
    delta_y = np.diff(position_y)
    _, heading_direction = math_utility.cart2pol(delta_x, delta_y)

    heading_direction_deg = np.degrees(heading_direction)
    if pad_first_value:
        heading_direction_deg = np.insert(heading_direction_deg, 0, heading_direction_deg[0])

    return heading_direction_deg + 180


def add_heading_direction_to_position_data_frame(position: pd.DataFrame) -> pd.DataFrame:
    """
    Add heading direction to position data frame as a new column.
    input : position data frame, this is the tacking data from the animal, must have position_x and position_y columns
    output : position data frame with 'heading_direction' added as a new column in degrees
    """
    x = position.position_x
    y = position.position_y
    heading_direction = calculate_heading_direction(x, y, pad_first_value=True)
    position['heading_direction'] = heading_direction
    return position


def add_heading_direction_to_spatial_firing_data_frame(spatial_firing: pd.DataFrame, position: pd.DataFrame, ephys_sampling_rate: int):
    """
    Calculate heading direction corresponding to firing events for data frame where each row corresponds to a cell
    :param spatial_firing: Data frame where each row corresponds to a neuron.
    :param position: Data frame with motion tracking information
    :param ephys_sampling_rate: Sampling rate of electrophysiology data (Hz)
    :return: spatial firing and position data frames with heading direction added as new columns
    """
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)

    headings = []
    spatial_firing = PostSorting.open_field_spatial_firing.calculate_corresponding_indices(spatial_firing, position, avg_sampling_rate_open_ephys=ephys_sampling_rate)
    for index, cluster in spatial_firing.iterrows():
        bonsai_indices_cluster_round = cluster.bonsai_indices.round(0)
        heading = list(position.heading_direction[bonsai_indices_cluster_round])
        headings.append(heading)
    spatial_firing['heading_direction'] = headings
    return spatial_firing, position


def add_heading_direction_to_spatial_firing_data_frame_one_cluster(cluster: pd.Series, position: pd.DataFrame, ephys_sampling_rate: int):
    """
    :param cluster: Series, data from one neuron (one row of spatial firing)
    :param position: Data frame with motion tracking data.
    :param ephys_sampling_rate: Sampling rate of electrophysiology data in Hz
    :return: spatial firing and position data frames with heading direction added as a new column
    """
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)

    headings = []
    spatial_firing = PostSorting.open_field_spatial_firing.calculate_corresponding_indices(cluster, position, avg_sampling_rate_open_ephys=ephys_sampling_rate)
    bonsai_indices_cluster_round = cluster.bonsai_indices.round(0)
    heading = list(position.heading_direction[bonsai_indices_cluster_round])
    headings.append(heading)
    spatial_firing['heading_direction'] = headings
    return spatial_firing, position


def calculate_corresponding_indices(spike_data: pd.DataFrame, spatial_data: pd.DataFrame, avg_sampling_rate_open_ephys: int) -> pd.DataFrame:
    """
    Find indices from movement data that correspond to firing events in spike_data
    :param spike_data: Data frame where each row corresponds to a cell
    :param spatial_data: Position tracking data
    :param avg_sampling_rate_open_ephys: Sampling rate of electrophysiology data (Hz)
    :return: spike data with the corresponding indices added as a new column
    """
    avg_sampling_rate_bonsai = float(1 / spatial_data['synced_time'].diff().mean())
    sampling_rate_rate = avg_sampling_rate_open_ephys / avg_sampling_rate_bonsai
    bonsai_indices_all = (np.array(spike_data.spike_times) / sampling_rate_rate)
    spike_data['bonsai_indices'] = bonsai_indices_all
    return spike_data


def calculate_corresponding_indices_trajectory(spike_data: pd.DataFrame, video_sampling : int) -> pd.DataFrame:
    """
    :param spike_data: Data frame  - has df.times_session which is the time stamps from the movement data in seconds
    :param video_sampling: sampling rate of camera tracking the animal
    :return: spike data with corresponding video tracking indices added
    """

    spike_data['bonsai_indices_trajectory'] = np.array(spike_data.times_session) * video_sampling
    return spike_data


def add_heading_during_spikes_to_field_df(field: pd.DataFrame, position: pd.DataFrame, ephys_sampling: int) -> pd.DataFrame:
    """
    :param field: Data frame where each row is data from a firing field of a cell.
    :param position: Motion tracking data from the animal
    :param ephys_sampling: Sampling rate of electrophysiology data (in Hz, =1 if given in seconds)
    :return: fields data frame with heading direction during firing events added as a new column
    """
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)
    fields = calculate_corresponding_indices(field, position, ephys_sampling)
    bonsai_indices_cluster_round = fields.bonsai_indices.round(0)
    heading = list(position.heading_direction[bonsai_indices_cluster_round])
    fields['heading_direction_in_field_spikes'] = heading
    return fields


def add_heading_from_trajectory_to_field_df(fields: pd.DataFrame, position: pd.DataFrame, video_sampling: int) -> pd.DataFrame:
    """
    :param fields: Data frame where each row is data from a firing field of a cell.
    :param position: Motion tracking data from the animal
    :return: fields data frame with heading direction from the trajectory added as a new column
    """
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)
    headings = []
    fields = calculate_corresponding_indices_trajectory(fields, video_sampling)
    bonsai_indices_cluster_round = fields.bonsai_indices_trajectory.round(0)
    heading = list(position.heading_direction[bonsai_indices_cluster_round])
    headings.append(heading)
    fields['heading_direction_in_field_trajectory'] = headings
    return fields


# add heading direction to field df (where each row is data from a firing field - see data_frame_utility
def add_heading_direction_to_fields_frame(fields: pd.DataFrame, position: pd.DataFrame, ephys_sampling: int):
    """
    :param fields: Data frame where each row is data from a firing field of a cell.
    :param position: Data frame containing the tracking data from the animal from the entire session with time stamps.
    :param ephys_sampling: Sampling rate of electrophysiology data (in Hz, =1 if given in seconds)
    :return: fields and position data frames with heading direction added as new columns. The field data frame will
    have a new column with head directions from the trajectory in the field and head directions when the cell fired
    in the field
    """
    if 'heading_direction' not in position:
        position = add_heading_direction_to_position_data_frame(position)
    fields = add_heading_during_spikes_to_field_df(fields, position, ephys_sampling)
    fields = add_heading_from_trajectory_to_field_df(fields, position)
    return fields, position


def main():
    """
    This is just here for testing.
    """

    path = '/DataFrames/'
    position_path = path + 'position.pkl'
    position = pd.read_pickle(position_path)
    spatial_firing_path = path + 'spatial_firing.pkl'
    spatial_firing = pd.read_pickle(spatial_firing_path)
    position = add_heading_direction_to_position_data_frame(position)
    # spatial_firing, position = add_heading_direction_to_spatial_firing_data_frame(spatial_firing, position)

    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position)
    field_df, position = add_heading_direction_to_fields_frame(field_df, position)


if __name__ == '__main__':
    main()