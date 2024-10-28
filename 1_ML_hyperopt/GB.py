import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json

# 读取数据
data = pd.read_csv("../data/dataset.csv")

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

# 定义参数空间 (GBR)
gbr_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
}

# 定义参数空间 (XGB)
xgb_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
}

# 定义参数空间 (LGBM)
lgbm_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
    'num_leaves': scope.int(hp.quniform('num_leaves', 20, 150, 5)),
    'subsample': hp.uniform('subsample', 0.5, 1),
}

# 定义目标函数
def objective_gbr(params):
    model = GradientBoostingRegressor(**params, random_state=21)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {'loss': mape, 'status': STATUS_OK}

def objective_xgb(params):
    model = XGBRegressor(**params, random_state=21, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {'loss': mape, 'status': STATUS_OK}

def objective_lgbm(params):
    model = LGBMRegressor(**params, random_state=21)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {'loss': mape, 'status': STATUS_OK}

# 超参数搜索函数
def hyperopt_search(space, objective, file_name):
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    best = {key: int(value) if isinstance(value, np.int64) else value for key, value in best.items()}
    with open(file_name, 'w') as f:
        json.dump(best, f)
    print("Best hyperparameters:", best)

# 搜索 GB, XGB, LGBM 超参数
hyperopt_search(gbr_space, objective_gbr, 'GBR_best_hyperparameters.json')
hyperopt_search(xgb_space, objective_xgb, 'XGB_best_hyperparameters.json')
hyperopt_search(lgbm_space, objective_lgbm, 'LGBM_best_hyperparameters.json')
