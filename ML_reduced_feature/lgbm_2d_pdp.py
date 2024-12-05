import pandas as pd
from function import split_data
from lgbm_function import trained_lgbm
from function import save_2d_pdp

# Read data
data = pd.read_csv("../data/dataset_reduced.csv")
X_train, X_test, y_train, y_test = split_data(data, 'Cs')
lgbm = trained_lgbm(X_train, y_train)

# Rename 'ID/IG' column to 'IDIG'
X_train = X_train.rename(columns={'ID/IG': 'IDIG'})

# Iterate over pairs of feature names
for i, feature_name1 in enumerate(X_train.columns):
    for feature_name2 in X_train.columns[i+1:]:
        features = (feature_name1, feature_name2)
        # Specify feature names and save path
        savepath = f'pdp/pdp_2d_{feature_name1}_{feature_name2}.csv'
        # Save 2D PDP data
        save_2d_pdp(lgbm, X_train, features, savepath)