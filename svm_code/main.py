import svm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from svm_class import SVM


def tranformedData(df,changeY = False):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    if(changeY):
        y = np.where(y == 0, -1, 1)
    X= np.array(X)
    y= np.array(y)
    return X, y


def question1A(X, y):

    wA = svm.svm_primal(X, y)
    print("question 1A  weights of svm primal model is: ", wA)
    svm.plot_classifier(wA, X, y,Name="question 1A")


def question1B(X, y):
    alpha, wB = svm.svm_dual(X, y)
    print("question 1B weights of svm dual model is: ", wB)
    sv = svm.support_vectors(alpha,percentile= 97)
    svm.plot_classifier(wB, X, y, sv, "question 1B")


def question1():
    df = pd.read_csv('simple_classification.csv', delimiter=',')
    X, y = tranformedData(df, changeY=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    question1A(X, y)
    question1B(X, y)


def kernelWithDiffrentParma(kernel, kernel_metrix, X, y, Xtest, yTest, limit, pram, kernel_name,plotClassifaier = True):
    error_rates = []
    for i in range(2, limit):
        alpha = svm.svm_dual_kernel(X, y, kernel, i)
        sv = svm.support_vectors(alpha)
        ySV = y[sv]
        Xsv = X[sv]
        title = "{kernel} (gamma={gamma})".format(kernel = kernel_name, gamma=i)
        if( plotClassifaier):
            svm.plot_classifier_z_kernel(alpha, X, y, kernel, i, title, sv)
        predictions = svm.predictWithKernels(Xtest, Xsv, ySV, alpha[sv], kernel_metrix, i)
        error_rates.append(svm.error(yTest, predictions))
    plt.figure(figsize=(10, 5))
    plt.scatter([i for i in range(2, limit)], error_rates, color='blue')
    plt.plot([i for i in range(2, limit)], error_rates, color='blue')
    plt.xlabel(pram)
    plt.ylabel('Error Rate')
    plt.title("Error Rates for Different {parm} using {kernel}".format(parm=pram, kernel=kernel_name))
    plt.xticks(rotation=45)
    plt.show()

def question2():
    df = pd.read_csv('simple_nonlin_classification.csv', delimiter=',')
    X, y= tranformedData(df)
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2,train_size=0.8, random_state=42)
    kernelWithDiffrentParma(svm.polyKernel, svm.poly_kernel_matrix, XTrain, yTrain, XTest,yTest,10, "degree", "Polynomial kernel" )
    kernelWithDiffrentParma(svm.RBFKernel, svm.rbf_kernel_matrix, XTrain, yTrain, XTest, yTest, 10, "gamma",
                            "RBF kernel")
def question4():
    df = pd.read_csv('simple_nonlin_classification.csv', delimiter=',')
    X, y = tranformedData(df)
    XTrain, Xtest,yTrain, yTest = train_test_split(X,y,test_size=0.2, random_state=42);
    error_rates = []
    kernels = [
        ("Polynomial (deg=2), not soft svm", svm.polyKernel, 2, svm.poly_kernel_matrix,np.inf),
        ("Polynomial (deg=2), soft svm, c=1", svm.polyKernel, 2, svm.poly_kernel_matrix, 1),
        ("Polynomial (deg=3), not soft svm", svm.polyKernel, 3, svm.poly_kernel_matrix,np.inf),
        ("Polynomial (deg=4), not soft svm", svm.polyKernel, 4, svm.poly_kernel_matrix,np.inf),
        ("RBF (gamma=1), not soft svm", svm.RBFKernel, 1, svm.rbf_kernel_matrix,np.inf),
        ("RBF (gamma=2), not soft svm", svm.RBFKernel, 2, svm.rbf_kernel_matrix,np.inf),
        ("RBF (gamma=2), soft svm, c=1", svm.RBFKernel, 2, svm.rbf_kernel_matrix, 1),
        ("RBF (gamma=3), not soft svm", svm.RBFKernel, 3, svm.rbf_kernel_matrix,np.inf),
        ("sigmoid (gamma=1), not soft svm", svm.sigmoid_karnel, 1, svm.sigmoid_karnel, np.inf),
        ("sigmoid (gamma=2),  soft svm", svm.sigmoid_karnel, 2, svm.sigmoid_karnel, 1),
        ("sigmoid (gamma=3),  soft svm", svm.sigmoid_karnel, 3, svm.sigmoid_karnel, 1),
        ("sigmoid (gamma=3), t soft svm", svm.sigmoid_karnel, 3, svm.sigmoid_karnel, 2)
    ]
    svmc = SVM()
    for kernel in kernels:
        if kernel[1] ==  svm.polyKernel:
            svmc= SVM(kernel=kernel[1], degree=kernel[2],c=kernel[4])
        else:
            if kernel[1] == svm.RBFKernel:
                svmc = SVM(kernel=kernel[1], gamma=kernel[2],c=kernel[4])
        svmc.fit(XTrain, yTrain)
        predictions = svmc.predictWithKernels(Xtest)
        error_rates.append(svmc.error(yTest,predictions))
    plt.scatter([k[0] for k in kernels], error_rates, color='blue')
    plt.plot([k[0] for k in kernels], error_rates, color='blue')
    plt.xlabel('Kernel')
    plt.ylabel('Error Rate')
    plt.title('Error Rates for Different Kernels')
    plt.xticks(rotation=45)
    plt.show()
    i=0
    for kernel in kernels:
        print(kernel[0], "error rate: ", error_rates[i])
        i+=1






question1()
question2()
question4()