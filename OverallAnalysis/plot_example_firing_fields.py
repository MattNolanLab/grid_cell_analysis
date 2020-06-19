import pandas as pd
import PostSorting.parameters
import PostSorting.open_field_make_plots


analysis_path = '/example_firing_fields/'


def plot_hd_in_fields_of_example_cell(prm):
    spatial_firing = pd.read_pickle(analysis_path + 'spatial_firing.pkl')
    spatial_data = pd.read_pickle(analysis_path + 'position.pkl')
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, spatial_data, prm)


def main():
    prm = PostSorting.parameters.Parameters()
    prm.set_output_path(analysis_path)
    prm.set_sampling_rate(30000)
    plot_hd_in_fields_of_example_cell(prm)


if __name__ == '__main__':
    main()
