import numpy as np
import pandas as pd
import scipy.sparse as sp

import implicit
from rlscore.learner import KronRLS
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import LinearOperator, svds


class Popularity():
    
    def fit(self, mf):
        self.Y = mf
        self.popularity = mf.sum(axis=0).A.flatten() / mf.shape[0]

    def predict(self, row):
        return(self.popularity)
		
    def predict1(self):
        predictions = np.repeat(self.popularity.reshape(1,-1), self.Y.shape[0], axis=0)
        return(predictions)
    
    def predict2(self, item_features):
        n, m = self.Y.shape
        t, k = item_features.shape
        return(np.zeros((n,t)))

    def predict3(self, user_features):
        s, k = user_features.shape
        predictions = np.repeat(self.popularity.reshape(1,-1), s, axis=0)
        return(predictions)
        #s, k = user_features.shape
        #n, m = self.Y.shape
        #return(np.zeros((s,m)))

    def predict4(self, user_features, item_features):
        s, k = user_features.shape
        t, k = item_features.shape
        return(np.zeros((s,t)))


class Random():
    
    def fit(self, mf):
        self.Y = mf
        self.random = np.random.randn(mf.shape[1])

    def predict(self, row):
        return(self.random)

    def predict1(self):
        predictions = np.repeat(self.random.reshape(1,-1), self.Y.shape[0], axis=0)
        return(predictions)
    
    def predict2(self, item_features):
        n, m = self.Y.shape
        t, k = item_features.shape
        return(np.zeros((n,t)))

    def predict3(self, user_features):
        s, k = user_features.shape
        n, m = self.Y.shape
        return(np.zeros((s,m)))

    def predict4(self, user_features, item_features):
        s, k = user_features.shape
        t, k = item_features.shape
        return(np.zeros((s,t)))



class MVN(object):
    
    def __init__(self, X):
        self.X = sp.csr_matrix(X, dtype=float)
        n, m = self.X.shape
        self.mu = self.X.sum(axis=0).A / n
        self.cov = (self.X.T.dot(self.X) / n - self.mu.T.dot(self.mu)).A
    
    def predict(self, userid):
        y = self.X[userid,].A.flatten()
        n = len(y)
        i1 = np.arange(n)[y == 1.0]
        i0 = np.arange(n)[y == 0.0]

        mu2 = self.mu.T[i1,]
        mu1 = self.mu.T[i0]
        a = np.ones_like(mu2)
        sigma12 = self.cov[np.ix_(i0,i1)]
        sigma22 = self.cov[np.ix_(i1,i1)]

        p = np.ones(n)
        p[i0] = mu1.flatten() + np.dot(sigma12, np.dot(np.linalg.pinv(sigma22), a-mu2)).flatten()
        return(p)

    def predict_online(self, y):
        #y = self.X[userid,].A.flatten()
        n = len(y)
        i1 = np.arange(n)[y == 1.0]
        i0 = np.arange(n)[y == 0.0]

        mu2 = self.mu.T[i1,]
        mu1 = self.mu.T[i0]
        a = np.ones_like(mu2)
        sigma12 = self.cov[np.ix_(i0,i1)]
        sigma22 = self.cov[np.ix_(i1,i1)]

        p = np.ones(n)
        p[i0] = mu1.flatten() + np.dot(sigma12, np.dot(np.linalg.pinv(sigma22), a-mu2)).flatten()
        return(p)

    def predict1(self):
        predictions = np.zeros(self.X.shape)
        for row in range(self.X.shape[0]):
            predictions[row,] = self.predict(row)
        return(predictions)
    
    def predict2(self, item_features):
        n, m = self.X.shape
        t, k = item_features.shape
        return(np.zeros((n,t)))

    def predict3(self, user_features):
        s, k = user_features.shape
        predictions = np.repeat(self.mu.reshape(1,-1), s, axis=0)
        return(predictions)

    def predict4(self, user_features, item_features):
        s, k = user_features.shape
        t, k = item_features.shape
        return(np.zeros((s,t)))



class MVN_nobias(object):
    
    def __init__(self, X):
        self.X = X
        n, m = self.X.shape
        self.mu = self.X.sum(axis=0).A / n
        self.std = np.where(self.mu > 0.0, np.sqrt(self.mu*(1.0-self.mu)), 1.0)
        self.cov = ((self.X.T.dot(self.X) / n - self.mu.T.dot(self.mu))/self.std.T.dot(self.std)).A
    
    def predict(self, userid):
        y = self.X[userid,].A.flatten()
        n = len(y)
        i1 = np.arange(n)[y == 1.0]
        i0 = np.arange(n)[y == 0.0]

        mu2 = self.mu.T[i1,]
        mu1 = self.mu.T[i0]
        a = np.ones_like(mu2)
        sigma12 = self.cov[np.ix_(i0,i1)]
        sigma22 = self.cov[np.ix_(i1,i1)]

        p = np.ones(n)
        p[i0] = (np.dot(sigma12, np.dot(np.linalg.pinv(sigma22), a-mu2))).flatten()
        
        return(p)

    def predict_online(self, y):
        #y = self.X[userid,].A.flatten()
        n = len(y)
        i1 = np.arange(n)[y == 1.0]
        i0 = np.arange(n)[y == 0.0]

        mu2 = self.mu.T[i1,]
        mu1 = self.mu.T[i0]
        a = np.ones_like(mu2)
        sigma12 = self.cov[np.ix_(i0,i1)]
        sigma22 = self.cov[np.ix_(i1,i1)]

        p = np.ones(n)
        p[i0] = np.dot(sigma12, np.dot(np.linalg.pinv(sigma22), a-mu2)).flatten()
        return(p)


class SVD():
    def __init__(self, k, epochs, reg):
        self.model = implicit.als.AlternatingLeastSquares(factors=k, iterations=epochs, regularization=reg)
        self.reg = reg

    def fit(self, Y):
        self.Y = Y
        self.model.fit(Y.T.tocoo(), show_progress=False)
        X = self.model.item_factors
        self.hat = np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)+np.eye(X.shape[1])*self.reg), X.T))

    def predict(self, row):
        predictions = self.model.user_factors[row,:].dot(self.model.item_factors.T)
        return(predictions)

    def predict_online(self, x):
        p = x.dot(self.hat).flatten()
        return(p)

    def predict1(self):
        predictions = self.model.user_factors.dot(self.model.item_factors.T)
        return(predictions)
    
    def predict2(self, item_features):
        n, m = self.Y.shape
        t, k = item_features.shape
        return(np.zeros((n,t)))

    def predict3(self, user_features):
        s, k = user_features.shape
        n, m = self.Y.shape
        return(np.zeros((s,m)))

    def predict4(self, user_features, item_features):
        s, k = user_features.shape
        t, k = item_features.shape
        return(np.zeros((s,t)))


class PureSVD():
    def __init__(self, k):
        self.k = k

    def fit(self, Y):
        self.Y = Y
        u, s, vt = svds(self.Y, k=self.k)
        self.user_factors = np.dot(u, np.diag(np.sqrt(s)))
        self.item_factors = np.dot(vt.T, np.diag(np.sqrt(s)))

    def predict(self, row):
        predictions = self.user_factors[row,:].dot(self.item_factors.T)
        return(predictions)

    def predict_online(self, x):
        if self.hat is None:
            X = self.item_factors
            self.hat = np.dot(X, np.dot(np.linalg.inv(np.dot(X.T, X)+np.eye(X.shape[1])*self.reg), X.T))
        base = self.popularity.A.flatten()
        predictions = base + (x - base).dot(self.hat).flatten()
        return(predictions)

    def predict1(self):
        predictions = self.user_factors.dot(self.item_factors.T)
        return(predictions)
    
    def predict2(self, item_features):
        n, m = self.Y.shape
        t, k = item_features.shape
        return(np.zeros((n,t)))

    def predict3(self, user_features):
        s, k = user_features.shape
        n, m = self.Y.shape
        return(np.zeros((s,m)))

    def predict4(self, user_features, item_features):
        s, k = user_features.shape
        t, k = item_features.shape
        return(np.zeros((s,t)))


class Questions():

    def fit(self, Y, user_features, reg, verbose=False):
        self.Y = np.array(Y.toarray(), dtype=float)        
        self.user_features = np.array(user_features.toarray(), dtype=float)

        clf = Ridge(alpha=reg, fit_intercept=False)

        coefs = []
        for i in range(self.Y.shape[1]):
            if (i % 100 == 0) & verbose:
                print(i, end=',')

            y = self.Y[:,i].flatten()
            clf.fit(user_features, y)
            coefs.append(clf.coef_)
        self.item_features = np.vstack(coefs)
        if verbose:
            print()

    def predict(self, row):
        predictions = self.user_features[row,].dot(self.item_features.T).flatten()
        return(predictions)

    def predict_online(self, x):
        predictions = self.item_features.dot(x).flatten()
        return(predictions)
        
    def predict1(self):
        predictions = self.user_features.dot(self.item_features.T)
        return(predictions)
        
    def predict2(self, item_features):
        n, m = self.Y.shape
        t, k = item_features.shape
        return(np.zeros((n,t)))

    def predict3(self, user_features):
        predictions = user_features.dot(self.item_features.T)
        return(predictions)

    def predict4(self, user_features, item_features):
        s, k = user_features.shape
        t, k = item_features.shape
        return(np.zeros((s,t)))

class Tags():

    def fit(self, Y, item_features, reg, verbose=False):
        self.Y = np.array(Y.toarray(), dtype=float)        
        self.item_features = np.array(item_features.toarray(), dtype=float)

        clf = Ridge(alpha=reg, fit_intercept=False)

        coefs = []
        for i in range(self.Y.shape[0]):
            if (i % 100 == 0) & verbose:
                print(i, end=',')

            y = self.Y[i,].flatten()
            clf.fit(item_features, y)
            coefs.append(clf.coef_)
        self.user_features = np.vstack(coefs)
        if verbose:
            print()

    def predict(self, row):
        predictions = self.user_features[row,:].dot(self.item_features.T).flatten()
        return(predictions)

    def predict_online(self, y):
        clf = Ridge(alpha=1, fit_intercept=False)
        #y = y - self.popularity
        clf.fit(self.item_features, y)
        predictions = clf.predict(self.item_features)
        return(predictions)

    def predict1(self):
        predictions = self.user_features.dot(self.item_features.T)
        return(predictions)
    
    def predict2(self, item_features):
        predictions = self.user_features.dot(item_features.T)
        return(predictions)

    def predict3(self, user_features):
        s, k = user_features.shape
        n, m = self.Y.shape
        predictions = np.zeros((s,m))
        return(predictions)

    def predict4(self, user_features, item_features):
        s, k = user_features.shape
        t, k = item_features.shape
        return(np.zeros((s,t)))


class QuestionsXTags():

    def fit(self, Y, user_features, item_features, reg):

        self.Y = np.array(Y.toarray(), dtype=float)
        
        self.m, self.n = Y.shape
        self.Y = self.Y.ravel(order='F')
        
        self.user_features = user_features
        self.item_features = item_features
        
        X_users = np.array(user_features.toarray(), dtype=float)
        X_items = np.array(item_features.toarray(), dtype=float)

        self.learner = KronRLS(X1=X_users, X2=X_items, Y=self.Y, regparam=reg)

    def predict(self, row):
        x = self.user_features.getrow(row)
        W = self.learner.predictor.W
        X = sp.kron(self.item_features, x)
        predictions = X.dot(W)
        return(predictions)

    def predict_online(self, x):
        #x = x.reshape(1,-1)
        W = self.learner.predictor.W
        X = sp.kron(self.item_features, x)
        predictions = X.dot(W)
        return(predictions)

    def predict1(self):
        m, n = self.m, self.n
        X_users = np.array(self.user_features.toarray(), dtype=float)
        X_items = np.array(self.item_features.toarray(), dtype=float)
        predictions = self.learner.predict(X_users, X_items).reshape((m, n), order='F')
        return(predictions)
    
    def predict2(self, item_features):
        m, n = self.m, self.n
        t, k = item_features.shape
        X_users = np.array(self.user_features.toarray(), dtype=float)
        predictions = self.learner.predict(X_users, item_features).reshape((m, t), order='F')
        return(predictions)

    def predict3(self, user_features):
        m, n = self.m, self.n
        t, k = user_features.shape
        X_items = np.array(self.item_features.toarray(), dtype=float)
        predictions = self.learner.predict(user_features, X_items).reshape((t, n), order='F')
        return(predictions)

    def predict4(self, user_features, item_features):
        t, k = user_features.shape
        s, k = item_features.shape
        predictions = self.learner.predict(user_features, item_features).reshape((t, s), order='F')
        return(predictions)

