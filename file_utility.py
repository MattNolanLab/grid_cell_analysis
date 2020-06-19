import glob
import os


def find_the_file(file_path, pattern, type):
    name = None
    file_found = True
    file_name = None

    file_counter = 0
    for name in glob.glob(file_path + pattern):
        file_counter += 1
        pass

    if file_counter > 1:
        print('There are more than one ' + type + ' files in this folder. This may not be okay.')

    if name is not None:
        file_name = name.rsplit('\\', 1)[1]
    else:
        print('The '+ type + ' file(such as ' + pattern + ' )is not here, or it has an unusual name.')

        file_found = False

    return file_name, file_found


def init_data_file_names(prm, beginning, end):
    prm.set_continuous_file_name(beginning)
    prm.set_continuous_file_name_end(end)


def set_continuous_data_path(prm):
    file_path = prm.get_filepath()
    continuous_file_name_1 = '105_CH'
    continuous_file_name_end_1 = '_0'
    continuous_file_name_2 = '100_CH'
    continuous_file_name_end_2 = ''

    recording_path = file_path + continuous_file_name_1 + str(1) + continuous_file_name_end_1 + '.continuous'
    if os.path.isfile(recording_path) is True:
        init_data_file_names(prm, continuous_file_name_1, continuous_file_name_end_1)

    recording_path = file_path + continuous_file_name_2 + str(1) + continuous_file_name_end_2 + '.continuous'
    if os.path.isfile(recording_path) is True:
        init_data_file_names(prm, continuous_file_name_2, continuous_file_name_end_2)


def set_dead_channel_path(prm):
    file_path = prm.get_filepath()
    dead_ch_path = file_path + "/dead_channels.txt"
    prm.set_dead_channel_path(dead_ch_path)


def create_behaviour_folder_structure(prm):
    movement_path = prm.get_filepath() + 'Behaviour'
    prm.set_behaviour_path(movement_path)

    data_path = movement_path + '/Data'
    analysis_path = movement_path + '/Analysis'

    prm.set_behaviour_data_path(data_path)
    prm.set_behaviour_analysis_path(analysis_path)

    if os.path.exists(movement_path) is False:
        print('Behavioural data will be saved in {}.'.format(movement_path))
        os.makedirs(movement_path)
        os.makedirs(data_path)
        os.makedirs(analysis_path)


# main path is the folder that contains 'recordings' and 'sorting_files'
def get_main_path(prm):
    file_path = prm.get_filepath()
    main_path = file_path.rsplit('/', 3)[-4]
    return main_path


def get_raw_mda_path_all_channels(prm):
    raw_mda_path = prm.get_filepath() + 'Electrophysiology/' + prm.get_spike_sorter() + '/raw.mda'
    return raw_mda_path


def get_raw_mda_path_separate_tetrodes(prm):
    raw_mda_path = '/data/raw.mda'
    return raw_mda_path


def folders_for_separate_tetrodes(prm):
    ephys_path = prm.get_filepath() + 'Electrophysiology'

    spike_path = ephys_path + '/Spike_sorting'
    data_path = ephys_path + '/Data'
    sorting_t1_path_continuous = spike_path + '/t1'
    sorting_t2_path_continuous = spike_path + '/t2'
    sorting_t3_path_continuous = spike_path + '/t3'
    sorting_t4_path_continuous = spike_path + '/t4'

    mountain_data_folder_t1 = spike_path + '/t1/data'
    mountain_data_folder_t2 = spike_path + '/t2/data'
    mountain_data_folder_t3 = spike_path + '/t3/data'
    mountain_data_folder_t4 = spike_path + '/t4/data'

    if os.path.exists(ephys_path) is False:
        os.makedirs(ephys_path)
        os.makedirs(spike_path)
        os.makedirs(data_path)

    if os.path.exists(sorting_t1_path_continuous) is False:
        os.makedirs(sorting_t1_path_continuous)
        os.makedirs(sorting_t2_path_continuous)
        os.makedirs(sorting_t3_path_continuous)
        os.makedirs(sorting_t4_path_continuous)

        os.makedirs(mountain_data_folder_t1)
        os.makedirs(mountain_data_folder_t2)
        os.makedirs(mountain_data_folder_t3)
        os.makedirs(mountain_data_folder_t4)


def create_ephys_folder_structure(prm):
    ephys_path = prm.get_filepath() + 'Electrophysiology'
    prm.set_ephys_path(ephys_path)
    data_path = ephys_path + '/' + prm.get_spike_sorter()

    if os.path.exists(ephys_path) is False:
        os.makedirs(ephys_path)
    if os.path.exists(data_path) is False:
        os.makedirs(data_path)


def create_folder_structure(prm):
    create_behaviour_folder_structure(prm)
    create_ephys_folder_structure(prm)


