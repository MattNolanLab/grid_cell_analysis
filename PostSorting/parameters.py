class Parameters:

    is_ubuntu = True
    is_windows = False
    is_stable = False
    is_interleaved_opto = False
    delete_two_min = False
    first_half_only = False
    second_half_only = False
    pixel_ratio = None
    opto_channel = ''
    sync_channel = ''
    sampling_rate = 0
    opto_tagging_start_index = None
    sampling_rate_rate = 0
    local_recording_folder_path = ''
    file_path = []
    output_path = []
    ms_tmp_path = []
    total_length_sampling_points = 0
    dead_channels = []
    sorter_name = []

    # vr parameters
    first_trial_channel = ''  # vr
    second_trial_channel = ''  # vr
    movement_channel = ''  # vr
    stop_threshold = 10.7  # vr
    track_length = 200  # vr
    cue_conditioned_goal = False

    def __init__(self):
        return

    def get_is_stable(self):
            return Parameters.is_stable

    def set_is_stable(self, is_stbl):
        Parameters.is_stable = is_stbl

    def get_sorter_name(self):
            return Parameters.sorter_name

    def set_sorter_name(self, name):
        Parameters.sorter_name = name

    def get_first_half_only(self):
        return Parameters.first_half_only

    def set_first_half_only(self, is_first):
        Parameters.first_half_only = is_first

    def get_second_half_only(self):
        return Parameters.second_half_only

    def set_second_half_only(self, is_second):
        Parameters.second_half_only = is_second

    def get_is_ubuntu(self):
        return Parameters.is_ubuntu

    def set_is_ubuntu(self, is_ub):
        Parameters.is_ubuntu = is_ub

    def get_is_windows(self):
        return Parameters.is_windows

    def set_is_windows(self, is_win):
        Parameters.is_windows = is_win

    def get_pixel_ratio(self):
        return Parameters.pixel_ratio

    def set_pixel_ratio(self, pr):
        Parameters.pixel_ratio = pr

    def get_opto_channel(self):
        return Parameters.opto_channel

    def set_opto_channel(self, opto_ch):
        Parameters.opto_channel = opto_ch

    def get_sync_channel(self):
        return Parameters.sync_channel

    def set_sync_channel(self, sync_ch):
        Parameters.sync_channel = sync_ch

    def get_sampling_rate(self):
        return Parameters.sampling_rate

    def set_sampling_rate(self, sr):
        Parameters.sampling_rate = sr

    def get_opto_tagging_start_index(self):
        return Parameters.opto_tagging_start_index

    def set_opto_tagging_start_index(self, opto_start):
        Parameters.opto_tagging_start_index = opto_start

    def get_sampling_rate_rate(self):
        return Parameters.sampling_rate_rate

    def set_sampling_rate_rate(self, sr):
        Parameters.sampling_rate_rate = sr

    def get_local_recording_folder_path(self):
        return Parameters.local_recording_folder_path

    def set_local_recording_folder_path(self, path):
        Parameters.local_recording_folder_path = path

    def get_filepath(self):
        return Parameters.file_path

    def set_file_path(self, path):
        Parameters.file_path = path

    def get_output_path(self):
        return Parameters.output_path

    def set_output_path(self, path):
        Parameters.output_path = path

    def get_ms_tmp_path(self):
        return Parameters.ms_tmp_path

    def set_ms_tmp_path(self, path):
        Parameters.ms_tmp_path = path

    def get_total_length_sampling_points(self):
        return Parameters.total_length_sampling_points

    def set_total_length_sampling_points(self, length):
        Parameters.total_length_sampling_points = length

    def get_dead_channels(self):
        return Parameters.dead_channels

    def set_dead_channels(d_ch = [], *args):
        dead_ch = []
        for dead_chan in args:
            dead_ch.append(dead_chan)

        Parameters.dead_channels = dead_ch

    def get_dead_channel_path(self):
        return Parameters.dead_channel_path

    def set_dead_channel_path(self, dead_ch):
        Parameters.dead_channel_path = dead_ch




