# Файл с основными настройками приложения

import sys
import pathlib

# append the path of the project's root directory
root_path = str(pathlib.Path(__file__).parent.parent)
sys.path.append(root_path)

##################################################################################################
# -------------------------Database-------------------------
##################################################################################################
db_dataset_columns = ['avg_area_income', 'avg_area_house_age', 'avg_area_number_of_rooms',
                      'avg_area_number_of_bedrooms', 'area_population', 'price', 'address']

db_separated_tables_columns_distribution = {
    'area_id_to_area_info': {'area_id', 'address', 'population', 'income'},
    'area_id_to_avg_house_info': {'area_id', 'avg_house_age', 'avg_number_of_rooms',
                                  'avg_number_of_bedrooms', 'avg_price'}
}

db_separated_tables_all_columns = ['area_id', 'address', 'population', 'income',
                                   'avg_house_age', 'avg_number_of_rooms',
                                   'avg_number_of_bedrooms', 'avg_price']

db_descriptive_characteristics = ['address']
db_numeric_characteristics = ['area_id', 'population', 'income',
                              'avg_house_age', 'avg_number_of_rooms',
                              'avg_number_of_bedrooms', 'avg_price', 'area_id']

db_name = root_path + '/Data/' + 'dataset.db'
##################################################################################################
