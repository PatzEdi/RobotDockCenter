# Script that extracts data from the .RDCconfig file. This script will be used throuhgout the project.

import json


def load_config():
    with open(".RDCconfig", 'r') as file:
        config = json.load(file)
    return config

def get_selected_template():
    return load_config()['RDCSettings'][0]['mainTemplate']

def get_main_template_data():
    return load_config()[get_selected_template()][0]

def get_model_num():
    return get_main_template_data()['model_num']

def get_data_set_len_row():
    return get_main_template_data()['data_set_len_row']

def get_data_set_len_col():
    return get_main_template_data()['data_set_len_col']

def get_distance_cline_range():
    return get_main_template_data()['distance_cline_range']

def get_distance_rline_range():
    return get_main_template_data()['distance_rline_range']

def get_rotation_range():
    return get_main_template_data()['rotation_range']