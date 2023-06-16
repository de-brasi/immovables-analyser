# Project entry point
import os.path
import sys

from typing import Dict, List

from PyQt5 import QtWidgets, QtGui, QtCore

from greeting_window import Ui_StartWidget
from main_window import Ui_MainWindow

import Library.core_api as core
from Scripts.settings import db_separated_tables_columns_distribution


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.dataset_path = ''
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # setup buttons
        # Create Update Delete
        self.ui.button_add_record.clicked.connect(self.add_record)
        self.ui.button_update_data.clicked.connect(self.update_records)
        self.ui.button_delete_record.clicked.connect(self.delete_records)

        # Reports
        self.ui.button_make_graph_frequency_report.clicked.connect(self.make_and_save_graph_frequency_report)
        self.ui.button_make_common_text_report.clicked.connect(self.make_and_save_common_text_report)
        self.ui.button_make_statistic_text_report.clicked.connect(self.make_and_save_statistic_text_report)

    # ---------Data operations---------
    def add_record(self):
        taken_data = self._get_data_from_interface(
            income_input=self.ui.input_income,
            age_input=self.ui.input_house_age,
            room_count_input=self.ui.input_room_count,
            bedroom_count_input=self.ui.input_badroom_count,
            population_count_input=self.ui.input_population_count,
            price_input=self.ui.input_price,
            address_input=self.ui.input_address
        )
        core.add_record_to_database(
            area_population=float(taken_data['population']),
            avg_area_income=float(taken_data['income']),
            avg_area_house_age=float(taken_data['avg_house_age']),
            avg_area_number_of_rooms=float(taken_data['avg_number_of_rooms']),
            avg_area_number_of_bedrooms=float(taken_data['avg_number_of_bedrooms']),
            price=float(taken_data['avg_price']),
            address=taken_data['address'],
        )

    def update_records(self):
        restrictions = self._get_restrictions()
        self._clear_restriction_fields()
        selector = core.Selector(restrictions=restrictions)
        data_for_update = self._get_data_from_interface(
            income_input=self.ui.input_update_income,
            age_input=self.ui.input_update_house_age,
            room_count_input=self.ui.input_update_room_count,
            bedroom_count_input=self.ui.input_update_badroom_count,
            population_count_input=self.ui.input_update_population_count,
            price_input=self.ui.input_update_price,
            address_input=self.ui.input_update_address
        )
        table_names = self._get_updated_tables(data_for_update)
        for table in table_names:
            core.update_data_in_database(selector=selector, kwargs=data_for_update, table_name=table)

    def delete_records(self):
        restrictions = self._get_restrictions()
        self._clear_restriction_fields()
        selector = core.Selector(restrictions=restrictions)
        core.delete_record_from_database(selector=selector)

    # ---------Reports making---------
    def make_and_save_graph_frequency_report(self):
        pass

    def make_and_save_common_text_report(self):
        restrictions = self._get_restrictions()
        self._clear_restriction_fields()

        columns = []
        # TODO: добавить выбор колонок для вывода

        report = core.make_common_text_report(selectable_columns=columns,
                                              row_selection_rule=restrictions)
        self.ui.field_text_report.setText(report)
        print(report)

    def make_and_save_statistic_text_report(self):
        selected_value = self.ui.one_field_comboBox.currentText()
        name_to_attribute = {
            'Доход': 'income',
            'Возраст': 'avg_house_age',
            'Число комнат': 'avg_number_of_rooms',
            'Число спален': 'avg_number_of_bedrooms',
            'Число населения': 'population',
            'Цена': 'avg_price',
        }
        report = core.make_statistic_text_report(name_to_attribute[selected_value])
        self.ui.field_text_report.setText(report)
        print(report)

    # ---------Utils---------
    def _get_restrictions(self) -> Dict[str, str]:
        res = dict()

        if self.ui.input_selector_income.text():
            res['income'] = self.ui.input_selector_income.text()

        if self.ui.input_selector_house_age.text():
            res['avg_house_age'] = self.ui.input_selector_house_age.text()

        if self.ui.input_selector_room_count.text():
            res['avg_number_of_rooms'] = self.ui.input_selector_room_count.text()

        if self.ui.input_selector_badroom_count.text():
            res['avg_number_of_bedrooms'] = self.ui.input_selector_badroom_count.text()

        if self.ui.input_selector_population_count.text():
            res['population'] = self.ui.input_selector_population_count.text()

        if self.ui.input_selector_price.text():
            res['avg_price'] = self.ui.input_selector_price.text()

        if self.ui.input_selector_address.text():
            res['address'] = self.ui.input_selector_address.text()

        return res

    def _get_data_from_interface(self,
                                 income_input: QtWidgets.QLineEdit,
                                 age_input: QtWidgets.QLineEdit,
                                 room_count_input: QtWidgets.QLineEdit,
                                 bedroom_count_input: QtWidgets.QLineEdit,
                                 population_count_input: QtWidgets.QLineEdit,
                                 price_input: QtWidgets.QLineEdit,
                                 address_input: QtWidgets.QLineEdit) -> Dict[str, str]:
        """
        Takes the entered data (for full or partial recording in the database) from the interface.
        Returns a dictionary with only those fields that were filled with something.
        :param income_input:
        :param age_input:
        :param room_count_input:
        :param bedroom_count_input:
        :param population_count_input:
        :param price_input:
        :param address_input:
        :return:
        """
        res = dict()
        self._clear_restriction_fields()

        if income_input.text():
            res['income'] = income_input.text()
            income_input.clear()

        if age_input.text():
            res['avg_house_age'] = age_input.text()
            age_input.clear()

        if room_count_input.text():
            res['avg_number_of_rooms'] = room_count_input.text()
            room_count_input.clear()

        if bedroom_count_input.text():
            res['avg_number_of_bedrooms'] = bedroom_count_input.text()
            bedroom_count_input.clear()

        if population_count_input.text():
            res['population'] = population_count_input.text()
            population_count_input.clear()

        if price_input.text():
            res['avg_price'] = price_input.text()
            price_input.clear()

        if address_input.text():
            res['address'] = address_input.text()
            address_input.clear()

        return res

    def _get_updated_tables(self, update_data: Dict[str, str]) -> List[str]:
        used_tables = set()
        for key in update_data:
            for table_name in db_separated_tables_columns_distribution.keys():
                if key in db_separated_tables_columns_distribution[table_name]:
                    used_tables.add(table_name)
        return list(used_tables)

    def _clear_restriction_fields(self):
        for field in (
                self.ui.input_selector_income, self.ui.input_selector_house_age, self.ui.input_selector_room_count,
                self.ui.input_selector_badroom_count, self.ui.input_selector_population_count,
                self.ui.input_selector_population_count, self.ui.input_selector_price,
                self.ui.input_selector_address
        ):
            field.clear()


class GreetingWindow(QtWidgets.QWidget):
    def __init__(self, main: MainWindow):
        super(GreetingWindow, self).__init__()
        self.filename = ""
        self.main_window = main
        self.ui = Ui_StartWidget()
        self.ui.setupUi(self)
        self.ui.enter_button.clicked.connect(self.validate_url)

    def validate_url(self):
        file_path = self.ui.input_field.text()
        self.ui.input_field.clear()

        file_extension = file_path.split('.')[-1]   # must be a csv
        file_exists = os.path.isfile(file_path)

        if file_extension == 'csv' and file_exists:
            self.filename = file_path
            self.main_window.dataset_path = self.filename
            self.close()
        else:
            self.ui.input_field.setPlaceholderText("Неправильный путь к файлу")

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.main_window.show()


if __name__ == "__main__":
    """
    Each launch of the program creates a new database from the dataset, and can delete previous changes. 
    Each launch of the program can delete all saved reports and replace them with new ones with the same names.
    """

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Auto Comment')

    main_window = MainWindow()
    greeting_window = GreetingWindow(main_window)
    greeting_window.show()
    app.exit(app.exec())

    ##############################################
    # dataset_path = './USA_Housing_dataset.csv'
    # core.init_database(dataset_path)
    # core.add_record_to_database(0, 0, 0, 0, 0, 0, "hello world")
    #
    # core.update_data_in_database(
    #     core.Selector(restrictions={'address': 'hello world', 'avg_house_age': '=0'}),
    #     {'table_name': 'area_id_to_avg_house_info', 'avg_house_age': '0'}
    # )
    #
    # stat_report = core.make_statistic_text_report('avg_house_age')
    # print(stat_report)
    #
    # print()
    # report = core.make_common_text_report(
    #     selectable_columns=['area_id', 'address', 'avg_house_age'],
    #     row_selection_rule={'address': '=="hello world"'}
    # )
    # print(report)

    # core.get_graph_report_frequency_histogram(numeric_characteristics=['income'])
    # core.get_graph_report_frequency_histogram(numeric_characteristics=['avg_house_age'])


# /home/ilya/WorkSpace/Projects/immovables_analyser/USA_Housing_dataset.csv
