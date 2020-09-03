import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

def split_data_k(games, k, frac=0.25):
    n, d = games.shape
    row_games = games.indptr[1:] - games.indptr[:-1]
    row_test = np.random.choice(np.arange(n)[row_games > k], int(n*frac), replace=False)
    train_data = []
    test_data = []
    train_indices = []
    test_indices = []
    train_indptr = [0]
    test_indptr = [0]
    for row_index, row in enumerate(games):
        if row_index in row_test:
            rnd = np.random.permutation(len(row.indices))
            train_indices.extend(row.indices[rnd][:k])
            train_data.extend(row.data[rnd][:k])
            test_indices.extend(row.indices[rnd][k:])
            test_data.extend(row.data[rnd][k:])
        else:
            train_indices.extend(row.indices)
            train_data.extend(row.data)

        train_indptr.append(len(train_indices))
        test_indptr.append(len(test_indices))

    train_matrix = sp.csr_matrix((train_data, train_indices, train_indptr), shape=(n, d), dtype=np.float32)
    test_matrix = sp.csr_matrix((test_data, test_indices, test_indptr), shape=(n, d), dtype=np.float32)
    return(train_matrix, test_matrix)


def auc_score_player(predict, ratings_test, ratings_train, verbose=False):
    n_players, n_games = ratings_train.shape
    auc = 0.0
    n_test = 0
    for row_index, row in enumerate(ratings_test):
        if len(row.indices) > 0:
            exclude = ratings_train.getrow(row_index).indices
            y_pred = predict(row_index)
            y_pred = np.delete(y_pred, exclude)
            y_true = np.zeros(n_games)
            y_true[row.indices] = 1
            y_true = np.delete(y_true, exclude)
            auc += roc_auc_score(y_true, y_pred)
            n_test += 1
        if verbose and (row_index % 1000 == 0):
            print(row_index, end=",")
    auc /= n_test
    if verbose:
        print()
    return auc


def _ndcg(y_true, y_pred):
    temp = np.argsort(-y_pred)
    rank = np.empty_like(temp)
    rank[temp] = np.arange(len(y_pred)) + 1
    idcg = (1 / np.log2(np.arange(y_true.sum()) + 1 + 1)).sum()
    udcg = (y_true / np.log2(rank + 1)).sum()
    dcg = (udcg / idcg)
    return(dcg)

def dcg_score_player(predict, ratings_test, ratings_train=None, verbose=False):
    n_players, n_games = ratings_test.shape
    dcg = 0.0
    n_test = 0
    for row_index, row in enumerate(ratings_test):
        if len(row.indices) > 0:
            y_pred = predict(row_index)
            y_true = np.zeros(n_games)
            y_true[row.indices] = 1
            if not ratings_train is None:
                exclude = ratings_train.getrow(row_index).indices
                y_pred = np.delete(y_pred, exclude)
                y_true = np.delete(y_true, exclude)
            dcg += _ndcg(y_true, y_pred)
            n_test += 1
        if verbose and (row_index % 100 == 0):
            print(row_index, dcg / n_test)
    dcg /= n_test
    if verbose:
        print()
    return dcg

def dcg_score_array(predictions, ratings_test, ratings_train=None, verbose=False):
    n_players, n_games = ratings_test.shape
    dcg = 0.0
    n_test = 0
    for row_index, row in enumerate(ratings_test):
        if len(row.indices) > 0:
            y_pred = predictions[row_index,]
            y_true = np.zeros(n_games)
            y_true[row.indices] = 1
            if not ratings_train is None:
                exclude = ratings_train.getrow(row_index).indices
                y_pred = np.delete(y_pred, exclude)
                y_true = np.delete(y_true, exclude)
            dcg += _ndcg(y_true, y_pred)
            n_test += 1
        if verbose and (row_index % 1000 == 0):
            print(row_index, end=",")
    dcg /= n_test
    if verbose:
        print()
    return dcg

def _precision_at_k(y_true, y_pred, k):
    best_k = np.argsort(y_pred)[-k:]
    prec = y_true[best_k].sum() / len(best_k)
    return(prec)

def precision_score_player(predict, ratings_test, ratings_train, k=20, verbose=False):
    n_players, n_games = ratings_train.shape
    precision = 0.0
    n_test = 0
    for row_index, row in enumerate(ratings_test):
        if len(row.indices) > 0:
            exclude = ratings_train.getrow(row_index).indices
            y_pred = predict(row_index)
            y_pred = np.delete(y_pred, exclude)
            y_true = np.zeros(n_games)
            y_true[row.indices] = 1
            y_true = np.delete(y_true, exclude)
            precision += _precision_at_k(y_true, y_pred, k)
            n_test += 1
        if verbose and (row_index % 1000 == 0):
            print(row_index, end=",")
    precision /= n_test
    if verbose:
        print()
    return precision

def precision_score_array(predictions, ratings_test, ratings_train=None, k=20, verbose=False):
    n_players, n_games = ratings_test.shape
    precision = 0.0
    n_test = 0
    for row_index, row in enumerate(ratings_test):
        if len(row.indices) > 0:
            y_pred = predictions[row_index,]
            y_true = np.zeros(n_games)
            y_true[row.indices] = 1
            if not ratings_train is None:
                exclude = ratings_train.getrow(row_index).indices
                y_pred = np.delete(y_pred, exclude)
                y_true = np.delete(y_true, exclude)
            precision += _precision_at_k(y_true, y_pred, k)
            n_test += 1
        if verbose and (row_index % 1000 == 0):
            print(row_index, end=",")
    precision /= n_test
    if verbose:
        print()
    return precision

