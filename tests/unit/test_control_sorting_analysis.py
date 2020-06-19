import control_sorting_analysis
import pytest


class TestGetSessionType:

    def test_openfield_type(self, tmp_path):
        parameters = '''openfield
        JohnWick/Open_field_opto_tagging_p038/M5_2018-03-06_15-34-44_of
        '''
        with open(tmp_path / 'parameters.txt', 'w') as f:
            f.write(parameters)

        is_vr, is_open_field = control_sorting_analysis.get_session_type(str(tmp_path))

        assert is_vr == False
        assert is_open_field == True

    def test_vr_type(self, tmp_path):
        parameters = '''vr
        JohnWick/Open_field_opto_tagging_p038/M5_2018-03-06_15-34-44_of
        '''
        with open(tmp_path / 'parameters.txt', 'w') as f:
            f.write(parameters)

        is_vr, is_open_field = control_sorting_analysis.get_session_type(str(tmp_path))

        assert is_vr == True
        assert is_open_field == False

    def test_invalid_type(self, tmp_path):
            parameters = '''openvr
            JohnWick/Open_field_opto_tagging_p038/M5_2018-03-06_15-34-44_of
            '''
            with open(tmp_path / 'parameters.txt', 'w') as f:
                f.write(parameters)

            is_vr, is_open_field = control_sorting_analysis.get_session_type(str(tmp_path))

            assert is_vr == False
            assert is_open_field == False

    def test_file_is_dir(self, tmp_path):
        with pytest.raises(Exception):
            is_vr, is_open_field = control_sorting_analysis.get_session_type(str(tmp_path))
