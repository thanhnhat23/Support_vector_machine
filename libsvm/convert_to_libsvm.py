import pandas as pd

# Read file CSV
data = pd.read_csv('student_data.csv')

# Pass -> +1, Fail -> -1
data['Result'] = data['Result'].map({'Pass': '+1', 'Fail': '-1'})

# 100 train data, 20 test data
train_data = data.iloc[:100]
test_data = data.iloc[100:]

# Convert to libsvm
with open('train.txt', 'w') as f:
    for index, row in train_data.iterrows():
        label = row['Result']
        features = [f"{i+1}:{row[col]:.2f}" for i, col in enumerate(['Toan', 'Ly', 'Hoa', 'Anh'])]
        f.write(f"{label} {' '.join(features)}\n")

# Convert to libsvm
with open('test.txt', 'w') as f:
    for index, row in test_data.iterrows():
        label = row['Result']
        features = [f"{i+1}:{row[col]:.2f}" for i, col in enumerate(['Toan', 'Ly', 'Hoa', 'Anh'])]
        f.write(f"{label} {' '.join(features)}\n")

print("Convert to libsvm successfully!")