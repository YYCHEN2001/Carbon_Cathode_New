import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv("../../data/dataset_reduced.csv")
data['target_class'] = pd.qcut(data['Cs'], q=10, labels=False)
X = data.drop(['Cs', 'target_class'], axis=1)
y = data['Cs']
stratify_column = data['target_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=stratify_column)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
train_df.to_csv('train_reduced.csv', index=False)
test_df.to_csv('test_reduced.csv', index=False)