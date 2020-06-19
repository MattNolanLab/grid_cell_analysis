import numpy as np
import os
import pandas as pd

import OverallAnalysis.folder_path_settings
import PostSorting.open_field_make_plots


local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/pattern_of_shapes/'


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_mouse(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    field_data['unique_id'] = unique_id
    unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    accepted_fields['unique_id'] = unique_id
    field_data['unique_cell_id'] = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


def plot_average_field_in_region(field_data, x1, x2, y1, y2, tag):
    sum_of_fields = np.zeros(360)
    sum_of_fields_norm = np.zeros(360)
    number_of_fields = 0
    for index, field in field_data.iterrows():
        classic_hd_hist = field.hd_hist_spikes / field.hd_hist_session
        field_indices = field.indices_rate_map
        x = (field_indices[:, 0] * 2.5).mean()  # convert to cm
        y = (field_indices[:, 1] * 2.5).mean()
        if (x >= x1) & (x < x2) & (y >= y1) & (y < y2):
            field_hist = np.nan_to_num(classic_hd_hist)
            sum_of_fields += np.nan_to_num(field_hist)
            number_of_fields += 1
            normalized_hist = field_hist / np.nanmax(field_hist)
            sum_of_fields_norm += normalized_hist

    avg = sum_of_fields / number_of_fields
    print('Number of fields in ' + tag)
    print(number_of_fields)
    if number_of_fields > 0:
        save_path = local_path + 'smooth_histograms/' + tag + 'not_normalized'
        PostSorting.open_field_make_plots.plot_single_polar_hd_hist(avg, 0,
                                                                    save_path, color1='red', title='')

        avg_norm = sum_of_fields_norm / number_of_fields
        save_path = local_path + 'smooth_histograms/' + tag + 'normalized'
        PostSorting.open_field_make_plots.plot_single_polar_hd_hist(avg_norm, 0,
                                                                    save_path, color1='red', title='')


def plot_all_fields(field_data):
    if not os.path.isdir(local_path + 'smooth_histograms/'):
        os.mkdir(local_path + 'smooth_histograms/')

    plot_average_field_in_region(field_data, x1=0, x2=33, y1=0, y2=33, tag='region_1')
    plot_average_field_in_region(field_data, x1=33, x2=66, y1=0, y2=33, tag='region_2')
    plot_average_field_in_region(field_data, x1=66, x2=100, y1=0, y2=33, tag='region_3')

    plot_average_field_in_region(field_data, x1=0, x2=33, y1=33, y2=66, tag='region_4')
    plot_average_field_in_region(field_data, x1=33, x2=66, y1=33, y2=66, tag='region_5')
    plot_average_field_in_region(field_data, x1=66, x2=100, y1=33, y2=66, tag='region_6')

    plot_average_field_in_region(field_data, x1=0, x2=33, y1=66, y2=101, tag='region_7')
    plot_average_field_in_region(field_data, x1=33, x2=66, y1=66, y2=101, tag='region_8')
    plot_average_field_in_region(field_data, x1=66, x2=100, y1=66, y2=101, tag='region_9')

    for index, field in field_data.iterrows():
        save_path = local_path + 'smooth_histograms/' + field.session_id + str(field.cluster_id) + str(field.field_id) + '_'
        field_indices = field.indices_rate_map

        d1 = (field_indices[:, 0] * 2.5).mean()  # convert to cm
        d2 = (field_indices[:, 1] * 2.5).mean()
        classic_hd_hist = field.hd_hist_spikes / field.hd_hist_session
        PostSorting.open_field_make_plots.plot_single_polar_hd_hist(classic_hd_hist, 0, save_path + str(round(d1,1)) + '_' + str(round(d2,1)), color1='navy', title='')

    print('I made smooth plots for all fields.')


def process_data():
    fields = pd.read_pickle(local_path + 'mice.pkl')
    accepted_fields = pd.read_excel(local_path + 'list_of_accepted_fields.xlsx')
    fields = tag_accepted_fields_mouse(fields, accepted_fields)
    accepted = fields.accepted_field == True
    hd = fields.hd_score >= 0.5
    grid = fields.grid_score >= 0.4
    grid_cells = np.logical_and(grid, np.logical_not(hd))
    plot_all_fields(fields[grid_cells & accepted])


def main():
    process_data()


if __name__ == '__main__':
    main()