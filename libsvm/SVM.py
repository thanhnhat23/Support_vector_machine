import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

# Read data file
def load_data(filename):
    data = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            features = [0.0] * 4  # 4 features (Toan, Ly, Hoa, Anh)
            for part in parts[1:]:
                idx, val = part.split(':')
                features[int(idx)-1] = float(val)
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

# Load data
X_train, y_train = load_data('train.txt')
X_test, y_test = load_data('test.txt')

# Train SVM with linear kernel
model = svm.SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize decision boundary
def plot_decision_boundary(X, y, model, title):
    # Chỉ lấy 2 features đầu tiên (Toán và Lý) để vẽ
    X_2d = X[:, :2]
    model.fit(X_2d, y)  # Huấn luyện lại trên 2D
    
    # Draw decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Predict
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Draw contour and scatter
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.xlabel('Toán')
    plt.ylabel('Lý')
    plt.title(title)
    plt.show()

# Draw decision boundary
plot_decision_boundary(X_train, y_train, model, 'Decision Boundary (Train Data)')
plot_decision_boundary(X_test, y_test, model, 'Decision Boundary (Test Data)')

# (Optional) Print support vectors
print(f"Number of support vectors: {len(model.support_vectors_)}")