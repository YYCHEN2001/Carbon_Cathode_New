import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json

# 读取数据
data = pd.read_csv("../../data/dataset.csv")

# 数据分割
data['target_class'] = pd.qcut(data['Cs'], q=10, labels=False)
X = data.drop(['Cs', 'target_class'], axis=1)
y = data['Cs']
stratify_column = data['target_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=stratify_column)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)

X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# 定义参数空间
space = {
    'C': hp.loguniform('C', np.log(0.1), np.log(1000)),
    'epsilon': hp.loguniform('epsilon', np.log(0.001), np.log(1)),
    'kernel': 'poly',  # 固定 kernel 为 'poly'
    'degree': scope.int(hp.quniform('degree', 2, 8, 1)),  # 搜索 degree 范围
    'coef0': hp.uniform('coef0', 0, 10),  # 搜索 coef0 范围
    'gamma': 'scale'  # 固定 gamma 为 'scale'
}


# 定义目标函数
def objective(params):
    # 根据内核选择参数
    if params['kernel'] != 'poly':
        del params['degree']  # 删除多余的 'degree' 参数

    model = SVR(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 计算 MAPE 作为损失
    mape = mean_absolute_percentage_error(y_test, y_pred)

    return {'loss': mape, 'status': STATUS_OK}


# 进行超参数搜索
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,
    trials=trials
)

# 使用json保存
with open('SVM_best_hyperparameters.json', 'w') as f:
    json.dump(best, f)