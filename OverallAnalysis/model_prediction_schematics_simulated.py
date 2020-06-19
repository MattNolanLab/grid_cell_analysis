import numpy as np
import matplotlib.pylab as plt
import OverallAnalysis.folder_path_settings
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots


analysis_path = OverallAnalysis.folder_path_settings.get_local_path() + 'prediction_schematics/'


def get_smooth_hist_and_plot(distribution, name):
    # get smooth hd hist
    smooth_hist = PostSorting.open_field_head_direction.get_hd_histogram(distribution, window_size=23)
    PostSorting.open_field_make_plots.plot_single_polar_hd_hist(smooth_hist, 0, analysis_path + name, color1='navy', title='')


def plot_distributions():
    print('multimodal')
    multimodal1 = np.random.vonmises(0, 6, 1000000)
    multimodal4 = np.random.vonmises(1.2, 6, 1000000)
    multimodal2 = np.random.vonmises(2, 6, 1000000)
    multimodal5 = np.random.vonmises(2.9, 7, 1000000)
    multimodal3 = np.random.vonmises(4, 6, 1000000)
    multimodal6 = np.random.vonmises(5.1, 6, 1000000)

    multimodal = np.concatenate((multimodal1, multimodal2, multimodal3, multimodal4, multimodal5, multimodal6))
    multimodal += np.pi
    get_smooth_hist_and_plot(multimodal, 'multimodal')

    uniform = np.random.uniform(0, 2 * np.pi, 1000000)
    print('uniform')
    get_smooth_hist_and_plot(uniform, 'uniform')
    print('unimodal')
    unimodal = np.random.vonmises(0.5, 4, 1000000) + np.pi
    get_smooth_hist_and_plot(unimodal, 'unimodal')


def main():
    plot_distributions()


if __name__ == '__main__':
    main()