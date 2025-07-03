import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# 1. Load and Prepare Dataset
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train Linear SVM
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(X_train_scaled, y_train)

# 3. Train RBF Kernel SVM (Non-linear)
svm_rbf = SVC(kernel='rbf', C=1, gamma=1)
svm_rbf.fit(X_train_scaled, y_train)

# 4. Visualize Decision Boundaries
def plot_decision_boundary(clf, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
    y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

plot_decision_boundary(svm_linear, X_train_scaled, y_train, "Linear SVM")
plot_decision_boundary(svm_rbf, X_train_scaled, y_train, "RBF Kernel SVM")

# 5. Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)

print("Best Parameters:", grid.best_params_)

# 6. Cross-validation Performance
cv_scores = cross_val_score(grid.best_estimator_, X_train_scaled, y_train, cv=5)
print("Cross-Validation Accuracy: %.2f" % cv_scores.mean())

# Final Evaluation
y_pred = grid.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
