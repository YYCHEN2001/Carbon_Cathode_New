import pandas as pd
from function import split_data
from lgbm_function import trained_lgbm
from function import save_pdp

# 读取数据
data = pd.read_csv("../data/dataset_reduced.csv")
X_train, X_test, y_train, y_test = split_data(data, 'Cs')
lgbm = trained_lgbm(X_train, y_train)
y_train_pred_lgbm = lgbm.predict(X_train)
y_test_pred_lgbm = lgbm.predict(X_test)

# 将ID/IG列的名字改为IDIG
X_train = X_train.rename(columns={'ID/IG': 'IDIG'})

# 遍历X_train特征名称
for feature_name in X_train.columns:
    # 指定特征名称和保存路径
    savepath = f'output/pdp_{feature_name}.csv'
    # 保存PDP数据
    save_pdp(lgbm, X_train, feature_name, savepath)