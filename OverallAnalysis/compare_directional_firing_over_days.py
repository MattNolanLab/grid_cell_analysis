import data_frame_utility
import os
import OverallAnalysis.folder_path_settings
import OverallAnalysis.shuffle_field_analysis
import pandas as pd
import PostSorting.parameters


local_path = OverallAnalysis.folder_path_settings.get_local_path()
analysis_path = local_path + '/compare_directional_firing_over_days/'

prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)


def get_shuffled_field_data(spatial_firing, position_data, shuffle_type='distributive', sampling_rate_video=50):
    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
    field_df = OverallAnalysis.shuffle_field_analysis.add_rate_map_values_to_field_df_session(spatial_firing, field_df)
    field_df = OverallAnalysis.shuffle_field_analysis.shuffle_field_data(field_df, analysis_path, number_of_bins=20,
                                  number_of_times_to_shuffle=1000, shuffle_type=shuffle_type)
    field_df = OverallAnalysis.shuffle_field_analysis.analyze_shuffled_data(field_df, analysis_path, sampling_rate_video,
                                     number_of_bins=20, shuffle_type=shuffle_type)
    return field_df


def process_data():
    # load shuffled field data
    if os.path.exists(analysis_path + 'DataFrames_1/fields.pkl'):
        shuffled_fields_1 = pd.read_pickle(analysis_path + 'DataFrames_1/fields.pkl')
    else:
        day1_firing = pd.read_pickle(analysis_path + 'DataFrames_1/spatial_firing.pkl')
        day1_position = pd.read_pickle(analysis_path + 'DataFrames_1/position.pkl')
        shuffled_fields_1 = get_shuffled_field_data(day1_firing, day1_position)
        shuffled_fields_1.to_pickle(analysis_path + 'DataFrames_1/fields.pkl')

    if os.path.exists(analysis_path + 'DataFrames_2/fields.pkl'):
        shuffled_fields_2 = pd.read_pickle(analysis_path + 'DataFrames_2/fields.pkl')
    else:
        day2_firing = pd.read_pickle(analysis_path + 'DataFrames_2/spatial_firing.pkl')
        day2_position = pd.read_pickle(analysis_path + 'DataFrames_2/position.pkl')
        # shuffle field analysis
        shuffled_fields_2 = get_shuffled_field_data(day2_firing, day2_position)
        shuffled_fields_2.to_pickle(analysis_path + 'DataFrames_2/fields.pkl')
    print('I shuffled data from both days.')

    day_1_field_1 = shuffled_fields_1[(shuffled_fields_1.cluster_id == 27) & (shuffled_fields_1.field_id == 0)]
    day_1_field_2 = shuffled_fields_1[(shuffled_fields_1.cluster_id == 27) & (shuffled_fields_1.field_id == 1)]

    day_2_field_1 = shuffled_fields_2[(shuffled_fields_2.cluster_id == 20) & (shuffled_fields_2.field_id == 0)]
    day_2_field_2 = shuffled_fields_2[(shuffled_fields_2.cluster_id == 20) & (shuffled_fields_2.field_id == 1)]

    print('day 1 field 2')
    print(day_1_field_2.number_of_different_bins_bh)
    print('day 2 field 2')
    print(day_2_field_2.number_of_different_bins_bh)

    print('day 1 field 1')
    print(day_1_field_1.number_of_different_bins_bh)
    print('day 2 field 1')
    print(day_2_field_1.number_of_different_bins_bh)


def main():
    process_data()


if __name__ == '__main__':
    main()
