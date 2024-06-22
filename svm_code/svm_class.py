import numpy as np
import qpsolvers as qps
from scipy.sparse import csc_matrix
import itertools
import svm

class SVM:
    def __init__(self,kernel = None, degree=None, c=np.inf, gamma=None):
        self._supportVectors = None
        self._percentileTreshold = 80
        self._alpha = None
        self._w =None
        self._kernel = kernel
        self._degree = degree
        self._gamma = gamma
        self._C = c
        self._Xvs=None
        self._yvs = None
        self._argvs=None

    def soft_svm_dual(self, X, y, max_iter=4000, verbose=False):
        N = X.shape[0]
        X = np.c_[X, np.ones(N)]
        q = -np.ones(N)
        GG = np.block([[-np.eye(N)], [np.eye(N)]])
        h = np.block([np.zeros(N), self._C * np.ones(N)])
        if(self._kernel is None):
            G = np.diag(y) @ X
            P = 0.5 * G @ G.T
        else:
            P = np.empty((N, N))
            arg = self.getArg()
            for i, j in itertools.product(range(N), range(N)):
                P[i, j] = y[i] * y[j] * self._kernel(X[i, :], X[j, :], arg)
        P = csc_matrix(P)
        GG = csc_matrix(GG)
        self._alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
        if(self._kernel is None):
            self._w = 0.5 * G.T @ self._alpha

    def fit(self,  X, y, max_iter=4000, verbose=False):
        self.soft_svm_dual(X,y,max_iter,verbose)
        self.support_vectors()
        self._Xvs = X[self._argvs]
        self._yvs = y[self._argvs]

    def support_vectors(self, percentile=80):
        alpha = np.abs(self._alpha)
        non_zero_alpha = alpha[alpha > 1e-5]
        theshold = np.percentile(non_zero_alpha, percentile)
        self._argvs = np.where(np.abs(alpha) > theshold)

    def predictWithKernels(self, X):
        predict_vals = self.decision_function(X)
        predictions = np.sign(predict_vals)
        return predictions
    def decision_function(self,X):
        arg = self.getArg()
        matrixFunc = self.getKMatrixFunc()
        K = matrixFunc(self._Xvs, X, arg)
        return K.T @ (self._alpha[self._argvs] * self._yvs)

    def error(self,gt, predictions):
        tp = np.sum((gt == predictions) & (gt == 1))
        tn = np.sum((gt == predictions) & (gt == -1))
        fp = np.sum((gt != predictions) & (gt == -1))
        fn = np.sum((gt != predictions) & (gt == 1))
        return (fp + fn) / (tn + tp + fp + fn)

    def getKMatrixFunc(self):
        if(self._kernel == svm.polyKernel):
            return svm.poly_kernel_matrix
        else:
            return svm.rbf_kernel_matrix
    def getArg(self):
        if (self._degree is not None):
            return self._degree
        else:
            return self._gamma

