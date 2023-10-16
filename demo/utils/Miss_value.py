'''
This is an utility to predict missed values by using other indexes through Random Forest.
'''
import pandas as pd
import numpy as np
from prettytable import PrettyTable

class MissValueProcessor(object):
    def __init__(self, LaxDatabase):
        self.origin_DF = LaxDatabase.dataFrame
        self.dataFrame = LaxDatabase.dataFrame.iloc[:,3:]
        self.indexes = LaxDatabase.indexes
        self.miss_values_info = {}
        self.miss_indexes = self.miss_value_indexes()

    def miss_value_indexes(self):
        indexes = []
        for index in self.indexes:
            if np.isnan(np.asarray(self.dataFrame[index])).sum():
                indexes.append(index)
        if len(indexes) == 0:
            print('no miss values found')
        else:
            print('Found miss values:')
            miss_values_report = PrettyTable(['Index', 'Time', 'Pos'])
            for index in self.indexes:
                self.miss_values_info[index] = {}
                self.miss_values_info[index]['Time'] = []
                self.miss_values_info[index]['Pos'] = []

                for row in range(len(self.origin_DF)):
                    if np.isnan(self.origin_DF[index][row]):
                        miss_values_report.add_row([index, self.origin_DF['Time'][row], self.origin_DF['Pos'][row]])
                        self.miss_values_info[index]['Time'].append(self.origin_DF['Time'][row])
                        self.miss_values_info[index]['Pos'].append(self.origin_DF['Pos'][row])

            print(miss_values_report)

        return indexes

    def train_test_data(self):
        train_data = {}
        test_data = {}
        log_y = {}
        '''
        need to remove features with nan values
        '''
        for index in self.miss_indexes:
            train_data[index], test_data[index], log_y[index] = self.split_data(index)

        return train_data, test_data, log_y

    def split_data(self, index):
        miss_indexes = self.miss_indexes.copy()
        miss_indexes.remove(index)
        dataFrame = self.dataFrame.copy()
        dataFrame = dataFrame.drop(miss_indexes, axis = 1)
        train_pos = pd.Series([not Bool for Bool in np.isnan(np.asarray(dataFrame[index]))])
        test_pos = pd.Series(np.isnan(np.asarray(dataFrame[index])))
        train_data = {}
        tmp_train_data = dataFrame[train_pos.values]
        if min(tmp_train_data[index].tolist()) < 100:
            train_data['y'] = tmp_train_data[index]
            logy = False
        else:
            train_data['y'] = np.log(tmp_train_data[index])
            logy = True
        train_data['x'] = tmp_train_data.drop([index], axis = 1)
        test_data = dataFrame[test_pos.values].drop([index], axis = 1)

        return train_data, test_data, logy