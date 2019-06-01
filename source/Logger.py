import pandas as pd
import os.path
from source import utils
import datetime
_logger = None
SOURCE = os.path.abspath(os.path.join(__file__, '../'))


class Logger:
    def __init__(self):
        self.datetime_object = datetime.datetime.now()
        self.folder_name = os.path.join(os.path.join(SOURCE, "logger_info"), "Logger_" + str(self.datetime_object).replace(':', '-'))
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

    def write_performances(self, auc_list, performances_list):
        performances_df = pd.DataFrame(columns=['name', 'date', 'f-score', 'precision', 'recall', 'auc'])
        for name, per in performances_list.items():
            performances_df = performances_df.append({'name': name, 'date': self.datetime_object,
                                                      'f-score': per['f-score'],
                                                      'precision': per['precision'], 'recall': per['recall'],
                                                      'auc': auc_list[name]}, ignore_index=True)
        performances_df.to_csv(os.path.join(self.folder_name, 'logger.csv'), sep=',', mode='a')

    def write_features(self, features):
        my_features = open(os.path.join(self.folder_name, 'features.txt'), 'w')
        for f in features:
            my_features.write(str(f)+'\n')
        my_features.close()

    def save_picture(self, picture_name):
        utils.save_picture(os.path.join(self.folder_name, picture_name))


def get_logger_instance():
    global _logger
    if _logger is None:
        _logger = Logger()
    return _logger
