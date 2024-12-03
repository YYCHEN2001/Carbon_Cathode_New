from lightgbm import LGBMRegressor

def trained_lgbm(x, y):
    # 训练LightGBM回归模型
    lgbm_params = {
        'n_estimators': 300,
        'learning_rate': 0.18,
        'max_depth': 5,
        'min_child_samples': 2,
        'colsample_bytree': 0.3,
        'num_leaves': 31,
        'reg_alpha': 0.6,
        'reg_lambda': 0,
        'verbose': -1,
        'random_state': 21
    }
    lgbm = LGBMRegressor(**lgbm_params)
    lgbm.fit(x, y)
    return lgbm