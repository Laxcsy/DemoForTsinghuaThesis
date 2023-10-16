import numpy as np  # 导入NumPy库，用于数值计算和数组操作
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

class MissValueEstimator(object):
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = {}
        self.cv_scores = {}

    def get_model(self, model_type):
        random_state = 17
        regressors = {"RF": RandomForestRegressor(random_state=random_state),
                      "SVR": SVR()
                      }
        return regressors[model_type]

    def train(self, X_train, y_train, cv = 5, random_state = 17):
        # 初始化交叉验证的变量和参数
        kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)  # 5折交叉验证
        for i, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
            trn_x = X_train.iloc[train_index]
            trn_y = y_train.iloc[train_index]
            val_x, val_y = X_train.iloc[valid_index], y_train.iloc[valid_index]
            self.model[i] = self.get_model(self.model_type)
            self.model[i].fit(trn_x, trn_y)

            # validate
            val_pred = self.model[i].predict(val_x)  # 在验证集上进行预测
            self.cv_scores[i] = r2_score(val_y, val_pred)  # 计算并存储验证集的R2

    def predict(self, X_test, logy):
        y_pred = np.zeros(X_test.shape[0])
        for model in self.model.values():
            if logy:
                y_pred += np.exp(model.predict(X_test))
            else:
                y_pred += model.predict(X_test)

        y_pred = y_pred/len(self.model)

        return y_pred