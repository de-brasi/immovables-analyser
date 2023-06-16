import os.path
import sys
import pathlib

# append the path of the project's root directory
root_path = str(pathlib.Path(__file__).parent.parent)
sys.path.append(root_path)

from Scripts.settings import db_dataset_columns


class DatasetValidation(BaseException):
    def __init__(self, message):
        self.msg = message

    def __str__(self):
        return f"Incorrect dataset passed! {self.msg}"


def validate_dataset(dataset_path: str) -> None:
    """
    Проверяет корректность файла dataset по следующим критериям:
    1) файл существует;
    2) расширение файла - csv;
    3) набор ключей в строке заголовка (при наличии) соответствует
        указанному в конфигурационном файле проекта;
    4) при отсутствии заголовочной строки считается что все
        остальные записи схожи по структуре с первой;
        идет проверка количества записей (колонок) с ожидаемым;
    :param dataset_path: путь до файла с данными
    :return: None
    """

    # check existing
    if not os.path.isfile(dataset_path):
        raise DatasetValidation(f"Incorrect path {dataset_path}")

    # check format
    if not dataset_path.split('.')[-1] == 'csv':
        raise DatasetValidation(f"Incorrect file's format {dataset_path}, "
                                f"{dataset_path.split('.')[-1]} expected")

    # Assert that real estate data contains numeric values
    with open(dataset_path, 'r') as dataset:

        first_row_contains_keys = True
        keys = dataset.readline().split(',')

        for key in keys:
            try:
                # Common sense says that a number cannot be a key
                # in any type of human-readable table.
                float(key)
                first_row_contains_keys = False
            except ValueError:
                pass

        if first_row_contains_keys:
            try:
                assert set(keys) == set(db_dataset_columns)
            except AssertionError:
                raise DatasetValidation(f"Incorrect header! "
                                        f"Got {keys} but expected {db_dataset_columns}")
        else:
            try:
                assert len(key) == len(db_dataset_columns)
            except AssertionError:
                raise DatasetValidation(f"Incorrect header! "
                                        f"Got {len(keys)} keys but expected "
                                        f"{len(db_dataset_columns)}")
