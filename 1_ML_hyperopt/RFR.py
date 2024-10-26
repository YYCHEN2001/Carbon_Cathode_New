import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
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

# 定义参数空间
space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),  # 树的数量
    'max_depth': scope.int(hp.quniform('max_depth', 5, 50, 1)),  # 最大树深
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),  # 分裂所需的最小样本数
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1)),  # 叶节点的最小样本数
    'max_features': hp.choice('max_features', [None, 'sqrt', 'log2'])  # 修改为 None, 'sqrt', 'log2'
}

# 定义目标函数
def objective(params):
    # 创建随机森林回归器
    model = RandomForestRegressor(**params, random_state=21, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

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

# 将 numpy 类型转换为标准 Python 类型
best = {key: int(value) if isinstance(value, np.int64) else value for key, value in best.items()}

# 使用json保存
with open('RFR_best_hyperparameters.json', 'w') as f:
    json.dump(best, f)

print("Best hyperparameters:", best)
