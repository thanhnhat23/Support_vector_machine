import pandas as pd
import numpy as np

n_samples = 120

np.random.seed(42)
data = {
    'Toan': np.random.uniform(3, 10, n_samples),
    'Ly': np.random.uniform(3, 10, n_samples),
    'Hoa': np.random.uniform(3, 10, n_samples),
    'Anh': np.random.uniform(3, 10, n_samples),
}
df = pd.DataFrame(data)

# TB >= 6.5 PASS, TB < 6.5 FAIL
df['TB'] = df[['Toan', 'Ly', 'Hoa', 'Anh']].mean(axis=1)
df['Result'] = df['TB'].apply(lambda x: 'Pass' if x >= 6.5 else 'Fail')
df = df.drop('TB', axis=1)  # Remove TB column

# Save file CSV
df.to_csv('student_data.csv', index=False)
print("Generate data successfully!")