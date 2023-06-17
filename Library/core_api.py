import sqlite3
import sys
import pathlib
import os

import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple

# append the path of the project's root directory
root_path = str(pathlib.Path(__file__).parent.parent)
sys.path.append(root_path)

from Scripts.settings import (db_name,
                              db_separated_tables_columns_distribution,
                              db_numeric_characteristics,
                              db_descriptive_characteristics)


class Selector:
    """
    Simple SQL select-request wrapper
    """

    def __init__(self, *args, restrictions: Dict[str, str]):
        """
        Restrictions is a map from string to string.
        Keys is a names of restriction's field, values of map is a compare sign and comparable value.
        For all numeric values you can use next comparison operators: '>', '<', '>=', '<=', '=', '!='.
        In case string value you must pass only comparable value (only equivalency checking),
        but it is forbidden to contain leading compare-operators by this string. Undefined behaviour as result!
        Example for table with columns <id>, <value_a>, <value_b>:
            Selector(
                expected_params=('value_a', 'value_b', ),
                restrictions={
                    'id': '>=2',
                    'value_a': 'hi'
                    }
                )
            This can take all records with id >= 2 and value_a equal to string "hi".

        :param args: not expected (for keyword-only arguments passing)
        :param expected_params: keys from table that must be got
        :param restrictions: restrictions on which records are selected
        """
        assert not args
        self.possible_prefixes = ('>', '<', '>=', '<=', '=', '!=')
        self.restrictions = restrictions

    def get_select_query_for_table(self, table_name: str, expected_params: tuple):
        """
        Make SQL SELECT query.
        The columns in expected_params must belong to table table_name.
        If you need to select columns from another table,
            you need to create two queries and join the results (by the key area_id)
        :param table_name:
        :param expected_params:
        :return:
        """

        query = f'SELECT {", ".join(expected_params)} FROM {table_name}'

        if self.restrictions:

            restrictions_belongs_to_select_table = \
                self._restrictions_is_belong_to_the_table(table_name, self.restrictions)

            if restrictions_belongs_to_select_table == "NO":

                table_names = list(db_separated_tables_columns_distribution.keys())
                assert len(table_names) == 2
                other_table = table_names[0] if table_names[0] != table_name else table_names[1]
                table_refinement = f' WHERE area_id ' \
                                   f'IN (SELECT area_id ' \
                                   f'FROM {other_table} ' \
                                   f'WHERE {self._make_restrictions_part(self.restrictions)})'
                query += table_refinement

            elif restrictions_belongs_to_select_table == "YES":

                table_refinement = ' WHERE ' + self._make_restrictions_part(self.restrictions)
                query += table_refinement

            elif restrictions_belongs_to_select_table == "NOT ONLY":

                table_refinement = self._make_cross_table_condition(table_name)
                query += table_refinement

            else:
                raise RuntimeError(f'Unexpected evaluation value {restrictions_belongs_to_select_table}')

        return query

    def get_update_query_for_table(self, table_name: str, values_to_update: Dict[str, str]) -> str:
        """
        Make SQL UPDATE query.
        If values_to_update contains field with string type, need to use "str:<value>".
        :param table_name: The name of the table in which you want to change the data.
                            Only changing data in one table is supported,
                            if you need to change something in each table,
                            you need to create and execute several queries.
        :param values_to_update: Column names and values to be written.
        :return: SQLite update query
        """
        required_string_prefix = 'str:'
        query = f'UPDATE {table_name}'

        # specify fields to change
        changed_fields = ' SET'
        for field, value in values_to_update.items():
            if value.startswith(required_string_prefix):
                changed_fields += f' {field}="{value}"'
            else:
                changed_fields += f' {field}={value}'
        query += changed_fields

        if self.restrictions:

            restrictions_belongs_to_updated_table = \
                self._restrictions_is_belong_to_the_table(table_name, self.restrictions)

            if restrictions_belongs_to_updated_table == "NO":

                table_names = list(db_separated_tables_columns_distribution.keys())
                assert len(table_names) == 2
                other_table = table_names[0] if table_names[0] != table_name else table_names[1]
                table_refinement = f' WHERE area_id ' \
                                   f'IN (SELECT area_id ' \
                                   f'FROM {other_table} ' \
                                   f'WHERE {self._make_restrictions_part(self.restrictions)})'
                query += table_refinement

            elif restrictions_belongs_to_updated_table == "YES":

                if self.restrictions:
                    query += ' WHERE ' + self._make_restrictions_part(self.restrictions)

            elif restrictions_belongs_to_updated_table == "NOT ONLY":

                table_refinement = self._make_cross_table_condition(table_name)
                query += table_refinement

            else:
                raise RuntimeError(f'Unexpected evaluation value {restrictions_belongs_to_updated_table}')

        return query

    def _make_cross_table_condition(self, table_name: str):
        """
        Make part of SQL query like:

        WHERE area_id IN (
            SELECT a.area_id
            FROM area_id_to_area_info a
            JOIN area_id_to_avg_house_info b ON a.area_id = b.area_id
            WHERE <condition_1> AND <condition_2> ... AND <condition_n>
        );

        :param table_name:
        :return:
        """
        table_names = list(db_separated_tables_columns_distribution.keys())
        assert len(table_names) == 2
        other_table = table_names[0] if table_names[0] != table_name else table_names[1]

        main_table_alias = 'a'
        other_table_alias = 'b'

        columns_distribution = self._sort_columns_into_tables(self.restrictions)
        all_conditions = "WHERE "

        # main table condition
        for this_table_columns in columns_distribution[table_name]:
            this_table_conditions = {k: v for k, v in self.restrictions.items() if k in this_table_columns}
            all_conditions += self._make_restrictions_part(restrictions=this_table_conditions,
                                                           key_prefix=main_table_alias)

        all_conditions += ' AND '

        # link table condition
        for this_table_columns in columns_distribution[other_table]:
            this_table_conditions = {k: v for k, v in self.restrictions.items() if k in this_table_columns}
            all_conditions += self._make_restrictions_part(restrictions=this_table_conditions,
                                                           key_prefix=other_table_alias)

        table_refinement = f' WHERE area_id IN (' \
                           f'SELECT {main_table_alias}.area_id ' \
                           f'FROM {table_name} {main_table_alias} ' \
                           f'JOIN {other_table} {other_table_alias} ' \
                           f'ON {main_table_alias}.area_id = {other_table_alias}.area_id ' \
                           f'{all_conditions}' \
                           f')'
        return table_refinement

    def _contains_leading_cmp_operators(self, value: str) -> bool:
        # For checking mode - string or numeric
        return any(
            list(
                map(value.startswith, self.possible_prefixes)
            )
        )

    def _make_restrictions_part(self, restrictions: Dict[str, str], key_prefix: str = "") -> str:
        """
        Make sql query 'FROM' part.
        From {value1: condition1, value2, condition2} to
            '<condition1(value1)> AND <condition2(value2)> AND <...> ... AND <...>'.
        :param restrictions:
        :param key_prefix:
        :return:
        """

        all_conditions = []
        if key_prefix:
            for key, condition in restrictions.items():
                if not self._contains_leading_cmp_operators(condition):
                    # string value case
                    all_conditions.append(f'{key_prefix}.{key} LIKE "{condition}"')
                else:
                    # numeric value case
                    all_conditions.append(f'{key_prefix}.{key}{condition}')
        else:
            for key, condition in restrictions.items():
                if not self._contains_leading_cmp_operators(condition):
                    # string value case
                    all_conditions.append(f'{key} LIKE "{condition}"')
                else:
                    # numeric value case
                    all_conditions.append(f'{key}{condition}')

        res = ' AND '.join(all_conditions)
        return res

    def _restrictions_is_belong_to_the_table(self, table_name: str, restrictions: Dict[str, str]) -> str:
        """
        Provides the ability to check whether the query applies to only one table,
        or whether it is necessary to check related data in other tables.
        :param table_name:
        :param restrictions: Any[List[str], Dict[str, str]]
        :return:
        """
        used_columns = []
        if self.restrictions:
            used_columns = list(self.restrictions.keys())

        if isinstance(restrictions, list) or isinstance(restrictions, tuple):
            used_columns += list(restrictions)
        elif isinstance(restrictions, dict):
            used_columns += list(restrictions.keys())
        else:
            raise RuntimeError(f"Unexpected params field type: {restrictions}")

        used_columns = set(used_columns)

        if used_columns.intersection(db_separated_tables_columns_distribution['area_id_to_area_info']) and \
                used_columns.intersection(db_separated_tables_columns_distribution['area_id_to_avg_house_info']):
            return "NOT ONLY"
        elif used_columns.intersection(db_separated_tables_columns_distribution[table_name]):
            return "YES"
        else:
            return "NO"

    def _sort_columns_into_tables(self,
                                  params: List[str] | Dict[str, str] | Tuple[str]) -> Dict[str, List[str]]:
        used_columns = []
        if self.restrictions:
            used_columns = list(self.restrictions.keys())

        if isinstance(params, list) or isinstance(params, tuple):
            used_columns += list(params)
        elif isinstance(params, dict) and params != self.restrictions:
            used_columns += list(params.keys())
        elif params == self.restrictions:
            pass
        else:
            raise RuntimeError(f"Unexpected params field type: {params}")

        res = dict()
        for column_name in used_columns:
            if column_name in db_separated_tables_columns_distribution['area_id_to_area_info']:
                if 'area_id_to_area_info' not in res:
                    res['area_id_to_area_info'] = []
                res['area_id_to_area_info'].append(column_name)
            elif column_name in db_separated_tables_columns_distribution['area_id_to_avg_house_info']:
                if 'area_id_to_avg_house_info' not in res:
                    res['area_id_to_avg_house_info'] = []
                res['area_id_to_avg_house_info'].append(column_name)
            else:
                raise RuntimeError(f"Unexpected column name: {column_name}")

        return res


def init_database(source_dataset: str):
    """
    Make SQL from CSV.
    It believes that the file exists, otherwise undefined behaviour.

    :param source_dataset: source valid dataset path
    :return: connection object
    """

    # Clear database if already exist
    if os.path.exists(db_name):
        os.remove(db_name)

    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()
    cursor.execute(
        '''
        CREATE TABLE source_dataset 
        (avg_area_income REAL, avg_area_house_age REAL, avg_area_number_of_rooms REAL,
         avg_area_number_of_bedrooms REAL, area_population REAL, price REAL, address TEXT)
        '''
    )

    # Load CSV to SQL
    dataset = pd.read_csv(source_dataset)
    dataset.to_sql('source_dataset', connection, if_exists='replace', index=False)

    # Third normal form.
    # area_id(int) -> avg_house_info(many fields)
    # area_id(int) -> address(text), population, income

    # Create area_id(int) -> avg_house_info(many fields) table
    select_fields_query = '''
    SELECT avg_area_house_age, avg_area_number_of_rooms,avg_area_number_of_bedrooms, price
    FROM source_dataset
    '''
    cursor.execute(select_fields_query)
    results = cursor.fetchall()
    cursor.execute(
        '''
        CREATE TABLE area_id_to_avg_house_info 
        (area_id INTEGER PRIMARY KEY AUTOINCREMENT, avg_house_age REAL, avg_number_of_rooms REAL, 
        avg_number_of_bedrooms REAL, avg_price REAL)
        '''
    )
    inserting_data_query = \
        '''
    INSERT INTO area_id_to_avg_house_info (avg_house_age, avg_number_of_rooms, avg_number_of_bedrooms, avg_price) 
    VALUES (?, ?, ?, ?)
    '''
    cursor.executemany(inserting_data_query, results)
    connection.commit()

    # Create area_id(int) -> address(text), population, income table
    select_fields_query = ''' SELECT address, area_population, avg_area_income FROM source_dataset '''
    cursor.execute(select_fields_query)
    results = cursor.fetchall()
    cursor.execute(
        ''' CREATE TABLE area_id_to_area_info 
        (area_id INTEGER PRIMARY KEY AUTOINCREMENT, address TEXT, population REAL, income REAL) '''
    )
    inserting_data_query = '''INSERT INTO area_id_to_area_info (address, population, income) VALUES (?, ?, ?)'''
    cursor.executemany(inserting_data_query, results)

    # Checking columns names correctness
    for table_name in ('area_id_to_avg_house_info', 'area_id_to_area_info'):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        columns_names = set([col[1] for col in columns])
        assert columns_names == db_separated_tables_columns_distribution[table_name]

    cursor.close()

    connection.commit()
    connection.close()


def add_record_to_database(avg_area_income: float,
                           avg_area_house_age: float,
                           avg_area_number_of_rooms: float,
                           avg_area_number_of_bedrooms: float,
                           area_population: float,
                           price: float,
                           address: str) -> int:
    # TODO: refactor according to using Selector style
    """
    Create new record in database.
    :param avg_area_income:
    :param avg_area_house_age:
    :param avg_area_number_of_rooms:
    :param avg_area_number_of_bedrooms:
    :param area_population:
    :param price:
    :param address:
    :return: index of created records
    """

    def get_last_id(table_name: str):
        nonlocal connection, cursor
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        return row_count

    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    avg_house_inserting_query = \
        '''
        INSERT INTO area_id_to_avg_house_info 
        (avg_house_age, avg_number_of_rooms, avg_number_of_bedrooms, avg_price) 
        VALUES (?, ?, ?, ?)
        '''
    cursor.execute(
        avg_house_inserting_query,
        (avg_area_house_age, avg_area_number_of_rooms, avg_area_number_of_bedrooms, price)
    )
    connection.commit()
    first_table_created_idx = get_last_id("area_id_to_avg_house_info")

    area_info_inserting_query = \
        '''
        INSERT INTO area_id_to_area_info 
        (address, population, income) 
        VALUES (?, ?, ?)
        '''
    cursor.execute(
        area_info_inserting_query,
        (address, area_population, avg_area_income)
    )
    second_table_created_idx = get_last_id("area_id_to_area_info")
    assert first_table_created_idx == second_table_created_idx

    connection.commit()
    connection.close()

    return first_table_created_idx


def delete_record_from_database(selector: Selector):
    """
    Delete records in all tables corresponding Selector value.
    :param selector:
    :return:
    """
    ##################################################
    # Create temporary table
    table_name = 'area_id_to_area_info'
    aggregation = selector.get_select_query_for_table(table_name=table_name, expected_params=('area_id', ))
    tmp_table_create_query = f'CREATE TABLE tmp_table ' \
                             f'AS SELECT * FROM (' \
                             f'{aggregation}' \
                             f')'
    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()
    cursor.execute(tmp_table_create_query)
    connection.commit()
    ##################################################

    table_1 = 'area_id_to_area_info'
    table_2 = 'area_id_to_avg_house_info'
    assert all([cur_table in db_separated_tables_columns_distribution.keys() for cur_table in (table_1, table_2)])

    deleting_query_1 = f'DELETE FROM {table_1} ' \
                       f'WHERE area_id IN (SELECT area_id FROM tmp_table);'
    deleting_query_2 = f'DELETE FROM {table_2} ' \
                       f'WHERE area_id IN (SELECT area_id FROM tmp_table);'

    cursor.execute(deleting_query_1)
    cursor.execute(deleting_query_2)
    connection.commit()

    ##################################################
    # Delete temporary table
    tmp_table_delete_query = 'DROP TABLE tmp_table'
    cursor.execute(tmp_table_delete_query)
    connection.commit()

    cursor.close()
    connection.close()
    ##################################################


def update_data_in_database(*args, selector: Selector, kwargs: dict, table_name: str):
    """
    Update kwargs['table_name'] table with other values in kwargs.
    Updated rows in the table are selected according to selector.
    :param selector:
    :param kwargs:
    :param table_name:
    :return:
    """
    assert not args

    connection = sqlite3.connect(db_name)
    cursor = connection.cursor()

    values_to_update = {k: v for k, v in kwargs.items()}
    update_query = selector.get_update_query_for_table(table_name, values_to_update)
    cursor.execute(update_query)
    connection.commit()

    cursor.close()
    connection.close()


def make_union_pandas_dataframe() -> pd.DataFrame:
    """
    Creates a merged Dataframe from two existing SQL tables.
    :return:
    """
    connection = sqlite3.connect(db_name)
    area_info = pd.read_sql_query("SELECT * FROM area_id_to_area_info", connection)
    house_info = pd.read_sql_query("SELECT * FROM area_id_to_avg_house_info", connection)
    connection.close()

    merged = area_info.merge(house_info, copy=True)
    return merged


def make_and_save_common_text_report(*args, selectable_columns: List[str],
                                     row_selection_rule: Dict[str, str],
                                     _static_mutable_values: List = [0, ]) -> str:
    """
    row_selection_rule is a map from string to string.
    Keys is a names of restriction's field, values of map is a compare sign and comparable value.

    For all numeric values you can use next comparison operators: '>', '<', '>=', '<=', '==', '!='.

    In case string value you must pass only comparable value as "<string_val>" with '==' (only equivalency checking).
    Be careful wit quotes type.

    :param args:
    :param selectable_columns:
    :param row_selection_rule:
    :param _static_mutable_values: service variable
    :return: Pandas dataframe as string
    """
    assert not args

    def make_selection_query(selection_rule: Dict[str, str]) -> str:
        res = ' and '.join([f"{key}{value}" for key, value in selection_rule.items()])
        return res

    # Using projection operations and cuts.
    # Using pandas instead of sql queries is consistent with task requirement.

    merged_dataframe = make_union_pandas_dataframe()
    if row_selection_rule:
        extra_row_discarding = merged_dataframe.query(make_selection_query(row_selection_rule)).copy()
    else:
        extra_row_discarding = merged_dataframe.copy()

    if selectable_columns:
        select_columns = extra_row_discarding[selectable_columns]
    else:
        # select all
        select_columns = extra_row_discarding

    result = select_columns.to_string(justify='left')
    with open(root_path + '/Graphics/' + f'common_report{_static_mutable_values[0]}.txt', 'w') as f:
        _static_mutable_values[0] += 1  # increase report_counter
        f.write(result)

    return result


def make_and_save_statistic_text_report(attribute_name: str) -> str:
    # TODO: save
    """
    Generates a statistical report based on the type of the passed attribute name.

    The name of the passed parameter must match the name of this parameter
    in the corresponding table obtained by casting to the third normal form.

    The function uses long operations. Asymptotic complexity O(n^2), where n is size of dataset.

    :param attribute_name: name of attribute
    :return:
    """
    assert attribute_name in (db_descriptive_characteristics + db_numeric_characteristics)

    union_dataframe = make_union_pandas_dataframe()
    selected_column = pd.DataFrame(union_dataframe[attribute_name])

    if attribute_name in db_descriptive_characteristics:

        statistics_dataframe = pd.DataFrame({
            attribute_name: [],
            'count': [],
            'percent': []
        })

        unique_attribute_values = selected_column[attribute_name].unique()

        for unique_val in unique_attribute_values:
            cur_unique_val_count = selected_column[attribute_name].value_counts()[unique_val]
            cur_unique_val_percent = (cur_unique_val_count / len(selected_column.index)) * 100
            new_row = [unique_val, cur_unique_val_count, cur_unique_val_percent]
            statistics_dataframe.loc[len(statistics_dataframe.index)] = new_row

    elif attribute_name in db_numeric_characteristics:

        statistics_dataframe = pd.DataFrame({
            'max': [selected_column[attribute_name].max(), ],
            'min': [selected_column[attribute_name].min(), ],
            'mean': [selected_column[attribute_name].mean(), ],
            'variance': [selected_column[attribute_name].var(), ],
            'standard deviation': [selected_column[attribute_name].std(), ],
        })

    else:
        raise RuntimeError(f"Unexpected attribute name {attribute_name}")

    return statistics_dataframe.to_string()


def text_report_summary_tables():
    # TODO: пересмотреть подход что есть качественная, что есть количественная характеристика
    # Due to the absence of two descriptive parameters in the database, the function is not implemented.
    raise RuntimeError("Function call not expected")


def graph_report_clustered_bar_chart():
    # TODO: пересмотреть подход что есть качественная, что есть количественная характеристика
    # Due to the absence of two descriptive parameters in the database, the function is not implemented.
    raise RuntimeError("Function call not expected")


def get_graph_report_frequency_histogram(*args, numeric_characteristics: List[str],
                                         precision: int = 0, _static_mutable_values: List = [0, ]) -> str:
    """
    Analogue of graph report categorized histogram. Save result as .png file to Graphics directory.
    :param args: for keyword-only argument passing
    :param numeric_characteristics: analyzed characteristics
    :param precision: the accuracy with which the values differ from each other
                (affects whether two close values fall into different buckets)
    :param _static_mutable_values: instead of function closure; use as report counter
    :return: path to new report
    """
    assert not args
    assert all([field in db_numeric_characteristics for field in numeric_characteristics])

    union_dataframe = make_union_pandas_dataframe()
    if precision:
        max_range = 0
        for characteristic in numeric_characteristics:
            max_range = max(max_range, union_dataframe[characteristic].max() -
                            union_dataframe[characteristic].min())
        bins_count = max_range // precision
        assert bins_count > 0
    else:
        bins_count = 100

    plt.hist([union_dataframe[param] for param in numeric_characteristics],
             bins=bins_count,
             label=numeric_characteristics)

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')

    plt.legend()
    new_name = root_path + '/Graphics/' + f'report{_static_mutable_values[0]}.png'
    plt.savefig(new_name)
    _static_mutable_values[0] += 1  # increase report_counter
    return new_name


# todo: structure
def graph_report_categorized_box_whisker_diagram():
    pass


# todo: structure
def graph_report_categorized_scatter_plot():
    pass
