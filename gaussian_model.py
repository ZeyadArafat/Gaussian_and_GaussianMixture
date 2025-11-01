import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as mvn
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(float).to_numpy()
y = mnist.target.astype(int).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
classes = np.arange(0, 10, 1)
distributions = []
proiers = []
for cls in classes:
    Xc = X_train[y_train == cls]
    class_mean = np.mean(Xc, axis=0)
    class_cov = np.cov(Xc, rowvar=False)
    distributions.append(mvn(class_mean, class_cov))
    proiers.append(len(Xc) / len(X_train))


def gaussian_predict(X, distributions, proiers, classes):
    predictions = []
    for x in X_test:
        probabilities = [distributions[c].pdf(x) * proiers[c] for c in classes]
        predicted_class = np.argmax(probabilities)
        predictions.append(predicted_class)
    return predictions



predictions = gaussian_predict(X_test, distributions, proiers, classes)

acc = accuracy_score(y_test, predictions)
print("Accuracy:", acc)
