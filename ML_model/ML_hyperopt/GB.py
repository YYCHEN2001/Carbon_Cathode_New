import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
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

# 定义参数空间 (GBR)
gbr_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 350, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 20, 1)),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
    'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
    'max_features': hp.quniform('max_features', 0.1, 1, 0.01),
}


# 定义目标函数
def objective_gbr(params):
    model = GradientBoostingRegressor(**params, random_state=21)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {'loss': mape, 'status': STATUS_OK}


def hyperopt_search(space, objective, file_name):
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=500,
        trials=trials
    )
    # 转换为标准 Python 类型
    best = {key: float(value) if isinstance(value, np.float64) else int(value) if isinstance(value, np.int64) else value
            for key, value in best.items()}

    # 保存最佳超参数
    with open(file_name, 'w') as f:
        json.dump(best, f)

    print("Best hyperparameters:", best)

# 搜索 GB, XGB, LGBM 超参数
hyperopt_search(gbr_space, objective_gbr, 'GBR_best_hyperparameters.json')
