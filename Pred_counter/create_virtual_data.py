import numpy as np
import pandas as pd

# Setting parameters based on provided statistics
num_samples = 100

# Defining columns
O_values = np.linspace(0, 25, num_samples)
N_values = np.linspace(0, 15, num_samples)

# Defining fixed values based on provided mean and median
SSA_mean = 1752.51
PV_mean = 1.25
Dap_mean = 3.04
RMIC_mean = 45.16
IDperIG_mean = 1.26

M_median = 2
Anion_median = 1
AML_median = 1.5
PW_median = 1.6

CD_fixed = 1

# Creating a DataFrame with 10000 rows
O_grid, N_grid = np.meshgrid(O_values, N_values)
O_flat = O_grid.flatten()
N_flat = N_grid.flatten()

# Creating data dictionary
data = {
    'O': O_flat,
    'N': N_flat,
    'SSA': [SSA_mean] * len(O_flat),
    'PV': [PV_mean] * len(O_flat),
    'RMIC': [RMIC_mean] * len(O_flat),
    'Dap': [Dap_mean] * len(O_flat),
    'IDperIG': [IDperIG_mean] * len(O_flat),
    'M': [M_median] * len(O_flat),
    'Anion': [Anion_median] * len(O_flat),
    'AML': [AML_median] * len(O_flat),
    'PW': [PW_median] * len(O_flat),
    'CD': [CD_fixed] * len(O_flat)
}

# Generating DataFrame
df = pd.DataFrame(data)

# Displaying DataFrame
print(df)

# Saving to CSV
df.to_csv('synthetic_data.csv', index=False)