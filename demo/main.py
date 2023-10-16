import numpy as np
from prettytable import PrettyTable
from dataset import LaxDataset
from utils.Miss_value import MissValueProcessor
from utils.models import MissValueEstimator

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_path = '/Users/liaoxin/Documents/Tsinghua/data/'
    exp_path = '/Users/liaoxin/Documents/Tsinghua/data/output/'

    # load and process raw data
    dataset = LaxDataset(data_path,plot = True, exp_path=exp_path)
    dataset.disinfection()
    data = PrettyTable(['index', 'min', 'max', 'class'])
    for index in dataset.indexes:
        data.add_row([index, np.asarray(dataset.dataFrame[index]).min(), np.asarray(dataset.dataFrame[index]).max(), dataset.standard_class[index]])
    print(data)

    # detect miss values
    predict = False
    if predict:
        MVP = MissValueProcessor(dataset)
        miss_values_info = MVP.miss_values_info
        if len(miss_values_info) != 0:
            # predict miss values by RF
            prediction_table = dataset.dataFrame.copy()
            X_train_dict, X_test, log_y = MVP.train_test_data()
            estimaotr = MissValueEstimator('RF')  # Random Forest
            predicted_values = PrettyTable(['index', 'Time', 'Pos', 'Predicted values'])

            for index in X_train_dict.keys():
                estimaotr.train(X_train_dict[index]['x'], X_train_dict[index]['y'])
                print(f'predicting {index} by models with scores as {list(estimaotr.cv_scores.values())}')
                y_pred = estimaotr.predict(X_test[index],log_y[index])

                for i, y_p in enumerate(y_pred):
                    time = miss_values_info[index]['Time'][i]
                    pos = miss_values_info[index]['Pos'][i]
                    predicted_values.add_row([index, time, pos, y_p])
                    row = prediction_table[(prediction_table['Time'] == time) & (prediction_table['Pos'] == pos)].index.item()
                    prediction_table.at[row, index] = y_p
            prediction_table.to_excel(f'{exp_path}prediction.xlsx',index = False)
            print(predicted_values)