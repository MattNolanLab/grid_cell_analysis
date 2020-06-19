import matplotlib.pylab as plt


def some_examples(spike_data_frame):
    good_cluster = spike_data_frame['goodcluster'] == 1
    light_responsive = spike_data_frame['lightscoreP'] <= 0.05

    good_light_responsive = spike_data_frame[good_cluster & light_responsive]

    print('Number of responses per animal:')
    print(spike_data_frame[light_responsive].groupby('animal').day.nunique())

    print('Avg firing freq per animal:')
    print(spike_data_frame[good_cluster].groupby('animal').avgFR.agg(['mean', 'median', 'count', 'min', 'max']))
    firing_freq = spike_data_frame[good_cluster].groupby('animal').avgFR.agg(['mean', 'median', 'count', 'min', 'max'])
    print(firing_freq.head())

    print(spike_data_frame[good_cluster].groupby(['animal', 'day']).avgFR.agg(['mean', 'median', 'count', 'min', 'max']))

    print(spike_data_frame[good_cluster].groupby(['animal', 'day']).goodcluster.agg(['count']))
    good_clusters_per_day = spike_data_frame[good_cluster].groupby(['animal', 'day']).goodcluster.agg(['count'])


def describe_dataset(spike_data_frame):
    good_cluster = spike_data_frame['goodcluster'] == 1
    number_of_good_clusters = spike_data_frame[good_cluster].count()
    print('Number of good clusters is:')
    print(number_of_good_clusters.id)

    print('Number of good clusters per animal:')
    print(spike_data_frame.groupby('animal').goodcluster.sum())

    print('Number of days per animal:')
    print(spike_data_frame.groupby('animal').day.nunique())


    print('Number of grid cells:')
    grid_cell = spike_data_frame['gridscore'] >= 0.4
    print(spike_data_frame[grid_cell].groupby('animal').day.nunique())
    print(spike_data_frame[grid_cell].fig_name_id)

    print('Number of hd cells:')
    hd_cell = spike_data_frame['r_HD'] >= 0.5
    print(spike_data_frame[hd_cell].groupby('animal').day.nunique())
    print(spike_data_frame[hd_cell].fig_name_id)


    print('Number of light responsive cells (low frequency):')
    light_responsive = spike_data_frame['lightscoreP'] <= 0.05
    print('Number of responses per animal:')
    print(spike_data_frame[light_responsive].groupby('animal').day.nunique())
    print(spike_data_frame[light_responsive].fig_name_id)


    print('Number of light responsive cells (high_frequency 100Hz):')
    light_responsive_high_100 = spike_data_frame['lightscore_p3'] <= 0.05
    print(spike_data_frame[light_responsive_high_100].groupby('animal').day.nunique())
    print(spike_data_frame[light_responsive_high_100].fig_name_id)

    print('Number of light responsive cells (high_frequency 200Hz):')
    light_responsive_high_200 = spike_data_frame['lightscore_p4'] <= 0.05
    print(spike_data_frame[light_responsive_high_200].groupby('animal').day.nunique())
    print(spike_data_frame[light_responsive_high_200].fig_name_id)


    # spike_data_frame_l2 = spike_data_frame.loc[spike_data_frame['location'] == 2]
    # spike_data_frame_l3 = spike_data_frame.loc[spike_data_frame['location'] == 3]
    spike_data_frame_l5 = spike_data_frame.loc[spike_data_frame['location'] == 5]
    spike_data_frame_superficial = spike_data_frame.loc[spike_data_frame['location'].isin([2, 3])]

    print('Number of grid cells in superficial layers:')
    grid_cell = spike_data_frame_superficial['gridscore'] >= 0.4
    print(spike_data_frame_superficial[grid_cell].groupby('animal').day.nunique())
    print(spike_data_frame_superficial[grid_cell].fig_name_id)

    print('Number of hd cells in superficial layers:')
    hd_cell = spike_data_frame_superficial['r_HD'] >= 0.5
    print(spike_data_frame_superficial[hd_cell].groupby('animal').day.nunique())
    print(spike_data_frame_superficial[hd_cell].fig_name_id)

    print('Number of conjunctive cells in superficial layers:')
    print(spike_data_frame_superficial[hd_cell & grid_cell].groupby('animal').day.nunique())
    print(spike_data_frame_superficial[hd_cell & grid_cell].fig_name_id)

    print('Number of grid cells in deep layers:')
    grid_cell = spike_data_frame_l5['gridscore'] >= 0.4
    print(spike_data_frame_l5[grid_cell].groupby('animal').day.nunique())
    print(spike_data_frame_l5[grid_cell].fig_name_id)

    print('Number of hd cells in deep layers:')
    hd_cell = spike_data_frame_l5['r_HD'] >= 0.5
    print(spike_data_frame_l5[hd_cell].groupby('animal').day.nunique())
    print(spike_data_frame_l5[hd_cell].fig_name_id)

    print('Number of conjunctive cells in deep layers:')
    print(spike_data_frame_l5[hd_cell & grid_cell].groupby('animal').day.nunique())
    print(spike_data_frame_l5[hd_cell & grid_cell].fig_name_id)

    print('***************************************************************')
    print('Average grid score in the deep layers is:')
    print(spike_data_frame_l5.gridscore.mean())
    print(spike_data_frame_l5.gridscore.std())
    print('Average grid score in the superficial layers is:')
    print(spike_data_frame_superficial.gridscore.mean())
    print(spike_data_frame_superficial.gridscore.std())

    print('Average spatial information score in the deep layers is:')
    print(spike_data_frame_l5.skaggs.mean())
    print(spike_data_frame_l5.skaggs.std())

    print('Spatial information score in superficial layers:')
    print(spike_data_frame_superficial.skaggs.mean())
    print(spike_data_frame_superficial.skaggs.std())

    print('Average hd score in the deep layers is:')
    print(spike_data_frame_l5.r_HD.mean())
    print(spike_data_frame_l5.r_HD.std())

    print('Average hd score in the superficial layers is:')
    print(spike_data_frame_superficial.r_HD.mean())
    print(spike_data_frame_superficial.r_HD.std())

    print('Average firing rate in the deep layers is:')
    print(spike_data_frame_l5.avgFR.mean())
    print(spike_data_frame_l5.avgFR.std())

    print('Average firing rate in the superficial layers is:')
    print(spike_data_frame_superficial.avgFR.mean())
    print(spike_data_frame_superficial.avgFR.std())

    print('Number of cells in superficial layers:')
    print(len(spike_data_frame_superficial))

    print('Number of cells in deep layers:')
    print(len(spike_data_frame_l5))

    print('Average firing rate in the deep layers in excitatory cells is:')
    excitatory_deep = spike_data_frame_l5['avgFR'] <= 10
    print(spike_data_frame_l5.avgFR[excitatory_deep].mean())
    print(spike_data_frame_l5.avgFR[excitatory_deep].std())

    print('Average firing rate in the superficial layers in excitatory cells is:')
    excitatory_superficial = spike_data_frame_superficial['avgFR'] <= 10
    print(spike_data_frame_superficial[excitatory_superficial].avgFR.mean())
    print(spike_data_frame_superficial[excitatory_superficial].avgFR.std())

    print('Number of excitatory cells in superficial layers:')
    print(len(spike_data_frame_superficial[excitatory_superficial]))

    print('Number of excitatory cells in deep layers:')
    print(len(spike_data_frame_l5[excitatory_deep]))

    print('Average firing rate in the deep layers in inhibitory cells is:')
    inhibitory_deep = spike_data_frame_l5['avgFR'] > 10
    print(spike_data_frame_l5.avgFR[inhibitory_deep].mean())
    print(spike_data_frame_l5.avgFR[inhibitory_deep].std())

    print('Average firing rate in the superficial layers in inhibitory cells is:')
    inhibitory_superficial = spike_data_frame_superficial['avgFR'] > 10
    print(spike_data_frame_superficial[inhibitory_superficial].avgFR.mean())
    print(spike_data_frame_superficial[inhibitory_superficial].avgFR.std())

    print('Number of excitatory cells in superficial layers:')
    print(len(spike_data_frame_superficial[inhibitory_superficial]))

    print('Number of excitatory cells in deep layers:')
    print(len(spike_data_frame_l5[inhibitory_deep]))


def plot_good_cells_per_day(spike_data_frame):
    for name, group in spike_data_frame.groupby('animal'):
        by_day = group.groupby('day').goodcluster.agg('sum')
        plt.xlabel('Days', fontsize=14)
        plt.ylabel('Number of good clusters', fontsize=14)
        by_day.plot(xlim=(-2, 16), ylim=(0, 20), linewidth=6)
        plt.savefig('C:/Users/s1466507/Documents/Ephys/overall_figures/good_cells_per_day/good_cells_per_day_' + name + '.png')


def describe_all_recordings(spike_data_frame):
    print('Number of recording days per animal:')
    print(spike_data_frame.groupby('animal').id.nunique())