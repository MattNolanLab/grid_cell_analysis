import OpenEphys
import numpy as np
import matplotlib.pylab as plt


def delete_noise(file_path, name, waveforms, timestamps):
    to_delete = np.array([])
    for wave in range(0, waveforms.shape[0]):
        if np.ndarray.max(abs(waveforms[wave, :, :])) > 0.0025:
            to_delete = np.append(to_delete, wave)

    # print('these are deleted')
    # print(to_delete)
    # print(waveforms[to_delete[0], :, 0])

    for spk in range(0, to_delete.shape[0]):
        plt.plot(waveforms[to_delete[spk], :, 0])

    plt.savefig(file_path + name + '_deleted_waves.png')

    waveforms = np.delete(waveforms, to_delete, axis=0)
    timestamps = np.delete(timestamps, to_delete)

    return waveforms, timestamps


def get_data_spike(folder_path, file_path, name):
    data = OpenEphys.load(file_path) # returns a dict with data, timestamps, etc.
    timestamps = data['timestamps']
    waveforms = data['spikes']

    # print('{} waveforms were found in the spike file'.format(waveforms.shape[0]))

    waveforms, timestamps = delete_noise(folder_path, name, waveforms, timestamps)

    return waveforms, timestamps


def get_data_continuous(prm, file_path):
    data = OpenEphys.load(file_path)
    signal = data['data']
    signal = np.asanyarray(signal)
    return signal


def get_events(prm, file_path):
    events = OpenEphys.load(file_path)
    return events