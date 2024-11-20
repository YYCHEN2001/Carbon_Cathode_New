| model                        |   R2_train |   MAE_train |   MAPE_train |   RMSE_train |   R2_test |   MAE_test |   MAPE_test |   RMSE_test |
|------------------------------|------------|-------------|--------------|--------------|-----------|------------|-------------|-------------|
| Linear Regression            |      0.391 |      23.193 |       33.52  |       31.149 |     0.477 |     21.809 |      27.622 |      28.707 |
| Support Vector Regression    |      0.869 |       8.404 |        8.728 |       14.422 |     0.779 |     10.91  |      12.891 |      18.647 |
| Randon Forest Regression     |      0.982 |       3.771 |        4.846 |        5.416 |     0.926 |      8.212 |      11.584 |      10.776 |
| Gradient Boosting Regression |      0.995 |       1.823 |        2.248 |        2.946 |     0.974 |      4.146 |       5.404 |       6.425 |
| LightGBM                     |      0.995 |       1.288 |        1.561 |        2.919 |     0.986 |      3.377 |       4.623 |       4.772 |
| XGBoost                      |      0.991 |       2.454 |        3.054 |        3.827 |     0.978 |      3.979 |       5.078 |       5.869 |