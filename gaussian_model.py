import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal as mvn

# ----------- Config ----------
SEED = 42
USE_PCA = True
PCA_DIM = 50            
COV_EPS = 1e-4         
TEST_SIZE = 10000       

#1) Load MNIST 
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

X_train, y_train = X[:-TEST_SIZE], y[:-TEST_SIZE]
X_test, y_test = X[-TEST_SIZE:], y[-TEST_SIZE:]

if USE_PCA:
    print(f"Applying PCA to {PCA_DIM} dims...")
    pca = PCA(n_components=PCA_DIM, random_state=SEED, svd_solver='randomized', whiten=False)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

classes = np.unique(y_train)
K = len(classes)
priors = np.array([(y_train == k).mean() for k in classes])
log_priors = np.log(priors + 1e-12)

# ============== Model A: Single Gaussian per class (MLE) ==============
print("Fitting single Gaussians per class...")
gauss_params = {}  # k -> (mu_k, Sigma_k)
for k in classes:
    Xk = X_train[y_train == k]
    mu_k = Xk.mean(axis=0)
    S_k = np.cov(Xk, rowvar=False)
    S_k = S_k + COV_EPS * np.eye(S_k.shape[0]) 
    gauss_params[k] = (mu_k, S_k)

def gaussian_loglik_all_classes(X_batch):
    n = X_batch.shape[0]
    LL = np.zeros((n, K), dtype=np.float64)
    for idx, k in enumerate(classes):
        mu_k, S_k = gauss_params[k]
        LL[:, idx] = mvn.logpdf(X_batch, mean=mu_k, cov=S_k, allow_singular=False)
    return LL

# Predict (Gaussian)
LL_gauss_test = gaussian_loglik_all_classes(X_test)
logpost_gauss_test = LL_gauss_test + log_priors  # log posterior up to normalization
yhat_gauss = classes[np.argmax(logpost_gauss_test, axis=1)]
acc_gauss = accuracy_score(y_test, yhat_gauss)
print(f"[Gaussian] Test accuracy: {acc_gauss:.4f}")

scores_gauss = logpost_gauss_test  # shape: (n_test, K)

def compute_roc_curve(y_true, scores, num_classes=10):
    fpr = {}
    tpr = {}
    thresholds = {}
    aucs = {}

    for i in range(num_classes):
        true_binary = (y_true == i).astype(int)
        fpr_list = []
        tpr_list = []
        thresholds_list = []

        #  predicted scores for class i
        class_scores = scores[:, i]

        thresholds_range = np.linspace(np.min(class_scores), np.max(class_scores), num=100)

        for threshold in thresholds_range:
            predicted_binary = (class_scores >= threshold).astype(int)

            TP = np.sum((true_binary == 1) & (predicted_binary == 1))
            TN = np.sum((true_binary == 0) & (predicted_binary == 0))
            FP = np.sum((true_binary == 0) & (predicted_binary == 1))
            FN = np.sum((true_binary == 1) & (predicted_binary == 0))

            TPR = TP / (TP + FN) if TP + FN > 0 else 0
            FPR = FP / (FP + TN) if FP + TN > 0 else 0

            tpr_list.append(TPR)
            fpr_list.append(FPR)
            thresholds_list.append(threshold)

        fpr[i] = np.array(fpr_list)
        tpr[i] = np.array(tpr_list)
        thresholds[i] = np.array(thresholds_list)

        aucs[i] = np.trapz(tpr[i], fpr[i])  # AUC via trapezoidal rule

    return fpr, tpr, aucs

def plot_roc_curves(fpr, tpr, aucs, num_classes=10):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {aucs[i]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - One-vs-Rest')
    plt.legend(loc='lower right')
    plt.show()

# ROC curves and AUCs for each class
fpr, tpr, aucs = compute_roc_curve(y_test, scores_gauss)

# Plot the ROC curves
plot_roc_curves(fpr, tpr, aucs)
