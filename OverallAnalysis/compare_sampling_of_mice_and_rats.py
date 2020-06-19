import matplotlib.pylab as plt
import OverallAnalysis.folder_path_settings
import pandas as pd
import PostSorting.parameters
prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)

from scipy import stats

local_path = OverallAnalysis.folder_path_settings.get_local_path() + '/compare_sampling/'


def load_data():
    mouse_data = pd.read_csv(local_path + "mouse_.csv")
    rat_data = pd.read_csv(local_path + "rat_.csv")
    return mouse_data, rat_data


def compare_sampling(mouse, rat):
    print(mouse.head())
    print(rat.head())
    print('mouse')
    print('avg time spent in field')
    print((mouse.time_spent_in_field / 30).mean())
    print((mouse.time_spent_in_field / 30).std())
    print('number of spikes')
    print(mouse.number_of_spikes_in_field.mean())
    print(mouse.number_of_spikes_in_field.std())
    print('number of fields: ' + str(len(mouse)))

    print('rat')
    print('avg time spent in field')
    print((rat.time_spent_in_field / 50).mean())
    print((rat.time_spent_in_field / 50).std())
    print('number of spikes')
    print(rat.number_of_spikes_in_field.mean())
    print(rat.number_of_spikes_in_field.std())
    print('number of fields: ' + str(len(rat)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time spent in field')
    ax.set_ylabel('Number of spikes in field')
    plt.scatter(mouse.time_spent_in_field / 30, mouse.number_of_spikes_in_field, color='navy', alpha=0.6)
    plt.scatter(rat.time_spent_in_field / 50, rat.number_of_spikes_in_field, color='lime', alpha=0.6)
    plt.savefig(local_path + 'sampling_in_mice_vs_rats.png')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Time spent in field (s)')
    ax.set_ylabel('Number of fields')
    plt.hist(mouse.time_spent_in_field / 30, color='navy', alpha=0.6)
    plt.hist(rat.time_spent_in_field / 50, color='lime', alpha=0.6)
    plt.savefig(local_path + 'time_spent_in_field.png')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Number_of_spikes')
    ax.set_ylabel('Number of fields')
    plt.hist(mouse.number_of_spikes_in_field, color='navy', alpha=0.6)
    plt.hist(rat.number_of_spikes_in_field, color='lime', alpha=0.6)
    plt.savefig(local_path + 'number_of_spikes_in_field.png')
    plt.close()

    print('time spent in field comparison:')
    d, p = stats.ks_2samp(mouse.time_spent_in_field, rat.time_spent_in_field)
    print(d)
    print(p)

    print('number of spikes comparison:')
    d, p = stats.ks_2samp(mouse.number_of_spikes_in_field, rat.number_of_spikes_in_field)
    print(d)
    print(p)


def main():
    mouse, rat = load_data()
    compare_sampling(mouse, rat)


if __name__ == '__main__':
    main()
