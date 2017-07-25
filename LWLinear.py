from numpy import *

def lwlr(point, y, k, weights):

    m = xMat.shape[0]
    weights = mat(eye((m)))
    for i in range(m):
        diffMat = point - xMat[i, :]
        weights[i, i] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    theta = xTx.T * (xMat.T * (weights * yMat))
    res = testPoint * theta
    print res
    return res

def lwLinear(testset, X, y, k):
    m = mat(testset).shape[0]
    y_pred = zeros(m)
    for i in range(m):
        y_pred[i] = lwlr(testset[i], X, y, k)
    return y_pred