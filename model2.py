import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(float).to_numpy()
y = mnist.target.astype(int).to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pca = PCA(n_components=50)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
classes = np.arange(0, 10)
n_components = 3   

gmms = []
priors = []
for cls in classes:
    Xc = X_train[y_train == cls]
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(Xc)
    gmms.append(gmm)
    priors.append(len(Xc) / len(X_train)) 
def gmm_predict(X, gmms, priors, classes):
    predictions = []
    for x in X:
        likelihoods = [gmms[c].score_samples(x.reshape(1, -1)) for c in classes]
        likelihoods = np.exp(likelihoods)
        posteriors = [likelihoods[c] * priors[c] for c in range(len(classes))]
        predictions.append(np.argmax(posteriors))
    return np.array(predictions)
y_pred = gmm_predict(X_test, gmms, priors, classes)
acc = accuracy_score(y_test, y_pred)
print("GMM Classifier Accuracy:", acc)
