import numpy as np
import qpsolvers as qps
import xlwings.main
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools


def svm_primal(X, y, max_iter=50000, verbose=False):
    N = X.shape[0]
    X = np.c_[X, np.ones(N)]
    n = X.shape[1]
    P = np.eye(n)
    q = np.zeros(n)
    G = -np.diag(y) @ X
    h = -np.ones(N)
    P= csc_matrix(P)
    G =csc_matrix(G)
    w = qps.solve_qp(P, q, G, h, solver='osqp', max_iter=max_iter, verbose=verbose)

    return w
def svm_dual(X, y, max_iter=4000, verbose=False):
    numberOfSamples = X.shape[0]
    X = np.c_[X, np.ones(numberOfSamples)]
    G = np.diag(y) @ X
    P = 0.5 * G @ G.T
    q = -np.ones(numberOfSamples)
    GG = -np.eye(numberOfSamples)
    h = np.zeros(numberOfSamples)
    P = csc_matrix(P)
    GG = csc_matrix(GG)
    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    w = G.T @ alpha
    return alpha, 0.5 * w


def plot_data(X, y, zoom_out=False, s=None):
    if zoom_out:
        x_min = np.amin(X[:, 0])
        x_max = np.amax(X[:, 0])
        y_min = np.amin(X[:, 1])
        y_max = np.amax(X[:, 1])

        plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])

    plt.scatter(X[:, 0], X[:, 1], c=y, s=s, cmap=pltcolors.ListedColormap(['blue', 'red']))


def support_vectors(alpha,percentile = 80):
    alpha = np.abs(alpha)
    non_zero_alpha = alpha[alpha>1e-5]
    theshold = np.percentile(non_zero_alpha,percentile)
    return np.where(np.abs(alpha) > theshold)


def plot_classifier(w, X, y, sv=None,Name="plot"):
    plot_data(X, y)
    lx = np.linspace(np.amin(X[:, 0]), np.amax(X[:, 0]), 60)
    plt.title(Name)
    ly = [(-w[-1] - w[0] * p) / w[1] for p in lx]
    plt.plot(lx, ly, color='black')

    ly1 = [(-w[-1] - w[0] * p - 1) / w[1] for p in lx]
    plt.plot(lx, ly1, "--", color='red')

    ly2 = [(-w[-1] - w[0] * p + 1) / w[1] for p in lx]
    plt.plot(lx, ly2, "--", color='blue')

    if (sv is not None):
        plt.scatter(X[sv, 0], X[sv, 1], s=300, linewidths=1, edgecolors='black', facecolors='none')
    plt.show()


def svm_dual_kernel(X, y, ker, arg, max_iter=4000, verbose=False):
    numberOfSamples = X.shape[0]
    P = np.empty((numberOfSamples, numberOfSamples))
    for i, j in itertools.product(range(numberOfSamples), range(numberOfSamples)):
        P[i, j] = y[i] * y[j] * ker(X[i, :], X[j, :],arg)
    P = 0.5 * (P + P.T)
    P = 0.5 * P
    q = -np.ones(numberOfSamples)
    GG = -np.eye(numberOfSamples)
    h = np.zeros(numberOfSamples)
    P = csc_matrix(P)
    GG = csc_matrix(GG)
    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    return alpha


def polyKernel(x, y, deg=2):
    return (1 + x.T @ y) ** deg

def poly_kernel_matrix(X1, X2, deg=2):
    return (1 + np.dot(X1, X2.T)) ** deg
def RBFKernel(x, y, gamma=1):
    return np.e**(-gamma*np.linalg.norm(x-y)**2)

def rbf_kernel_matrix(X1, X2, gamma):
    K = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)
    return K
def sigmoid_karnel(x,y,gamma):
    return np.tanh(gamma * np.dot(x,y.T))
def predictWithKernels(X,sv,ySV,alpha,kernelMetrixFunc, arg ):
    K = kernelMetrixFunc(sv,X,arg)
    decisionFunction = K.T @ (alpha * ySV)
    predictions = np.sign(decisionFunction)
    return predictions

def error(gt , predictions):
    tp = np.sum((gt == predictions) & (gt == 1))
    tn = np.sum((gt == predictions) & (gt == -1))
    fp = np.sum((gt != predictions) & (gt ==-1))
    fn = np.sum((gt!= predictions) & (gt == 1))
    return (fp+fn)/(tn+tp+fp+fn)


def plot_classifier_z_kernel(alpha, X, y, ker,arg,Name = 'plot', sv = None, s=None):
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])

    xx = np.linspace(x_min, x_max)
    yy = np.linspace(y_min, y_max)

    xx, yy = np.meshgrid(xx, yy)

    N = X.shape[0]
    z = np.zeros(xx.shape)
    for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
        z[i, j] = sum([y[k] * alpha[k] * ker(X[k, :], np.array([xx[i, j], yy[i, j]]),arg) for k in range(N)])

    plt.rcParams["figure.figsize"] = [15, 10]

    plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])
    plt.title(Name)
    plot_data(X, y, s=s)

    if (sv is not None):
        plt.scatter(X[sv, 0], X[sv, 1], s=300, linewidths=1, edgecolors='black', facecolors='none')

    plt.show()

