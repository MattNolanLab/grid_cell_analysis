class OverallParameters:

    isolation = 0
    noise_overlap = 0
    snr = 0

    path_to_data = ''
    save_output_path = ''
    false_positives_path_all = ''
    false_positives_path_separate = ''

    def __init__(self):
        return

    def get_isolation(self):
        return OverallParameters.isolation

    def set_isolation(self, isolation_th):
        OverallParameters.isolation = isolation_th

    def get_noise_overlap(self):
            return OverallParameters.noise_overlap

    def set_noise_overlap(self, noise_overlap_th):
        OverallParameters.noise_overlap = noise_overlap_th

    def get_snr(self):
            return OverallParameters.snr

    def set_snr(self, signal_to_noise_ratio):
        OverallParameters.snr = signal_to_noise_ratio

    def get_path_to_data(self):
            return OverallParameters.path_to_data

    def set_path_to_data(self, path):
        OverallParameters.path_to_data = path

    def get_save_output_path(self):
            return OverallParameters.save_output_path

    def set_save_output_path(self, path):
        OverallParameters.save_output_path = path

    def get_false_positives_path_all(self):
            return OverallParameters.false_positives_path_all

    def set_false_positives_path_all(self, path):
        OverallParameters.false_positives_path_all = path

    def get_false_positives_path_separate(self):
            return OverallParameters.false_positives_path_separate

    def set_false_positives_path_separate(self, path):
        OverallParameters.false_positives_path_separate = path