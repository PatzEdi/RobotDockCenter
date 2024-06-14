# Script that extracts data from the .RDCconfig file. This script will be used throuhgout the project.

import json

class ConfigParser:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        return config

    def get_selected_template(self):
        return self.config['RDCSettings'][0]['mainTemplate']

    def get_main_template_data(self):
        return self.config[self.get_selected_template()][0]

    def get_data_set_len(self):
        return self.get_main_template_data()['data_set_len']

    def get_distance_cline_range(self):
        return self.get_main_template_data()['distance_cline_range']

    def get_distance_rline_range(self):
        return self.get_main_template_data()['distance_rline_range']

    def get_rotation_range(self):
        return self.get_main_template_data()['rotation_range']