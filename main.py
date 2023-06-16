# Project entry point
import Library.core_api as core

if __name__ == "__main__":
    dataset_path = './USA_Housing_dataset.csv'
    core.init_database(dataset_path)
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
    #
    core.get_graph_report_frequency_histogram(numeric_characteristics=['income'])
