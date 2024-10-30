import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from lightgbm import LGBMRegressor
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

lgbm_space = {
    'num_leaves': scope.int(hp.quniform('num_leaves', 20, 200, 1)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 30, 1)),
    'learning_rate': hp.quniform('learning_rate', 0.01, 0.3, 0.01),
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 1050, 50)),
    'min_child_samples': scope.int(hp.quniform('min_child_samples', 1, 20, 1)),
    'subsample': hp.quniform('subsample', 0.5, 1.0, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.05),
    'reg_alpha': hp.quniform('reg_alpha', 0.1, 1, 0.01),
    'reg_lambda': hp.quniform('reg_lambda', 0.1, 1, 0.01),
    'extra_trees': hp.choice('extra_trees', [True, False])  # 增加额外的随机分裂尝试
}


def objective_lgbm(params):
    model = LGBMRegressor(**params, random_state=21)
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

hyperopt_search(lgbm_space, objective_lgbm, 'LGBM_best_hyperparameters.json')