#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec7_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Decision Trees ancilliary files - ADL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score, recall_score, precision_score, f1_score, precision_recall_curve, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
import pydotplus


color_palette_sequential = [ '#ece3f0', '#d0d1e6', '#a6bddb', '#67a9cf', '#3690c0', '#02818a', '#016c59', '#014636']
markers = ['o', '^', '*','H', 'P', 'D', 'X', 'h', 'p', 'd', 'c']


def get_mu_sigma(train_vector, test_vector):
    """TODO: Docstring for get_mu_sigma.

    :train_vector: TODO
    :test_vector: TODO
    :returns: TODO

    """
    return np.mean(train_vector, axis=1), np.std(train_vector, axis=1), np.mean(test_vector, axis=1), np.std(test_vector, axis=1)

def generate_mesh_grid(df, x1, x2):
    """TODO: Docstring for generate_mesh_grid.

    :df: TODO
    :x1: TODO
    :x2: TODO
    :returns: TODO

    """
    tmp_X = df.loc[:, [x1, x2]]
    tmp_x, tmp_y = np.meshgrid(
        np.linspace(np.min(tmp_X[x1]), np.max(tmp_X[x1]), num=100),
        np.linspace(np.min(tmp_X[x2]), np.max(tmp_X[x2]), num=100)
    )
    joint_xy = np.vstack([
        tmp_x.ravel(), tmp_y.ravel()
    ]).T

    return tmp_x, tmp_y, joint_xy

def train_test_over_params(model, params, X_train, X_test, y_train, y_test):
    """TODO: Docstring for train_test_over_params.

    :model: TODO
    :params: TODO
    :X_train: TODO
    :X_test: TODO
    :y_train: TODO
    :y_test: TODO
    :returns: TODO

    """
    tmp_train, tmp_test = [], []
    values = list(params.values())[0]
    hyperparam = str(list(params.keys())[0])


    for i in values:
        params_spec = {hyperparam: i}
        tmp_model = model.set_params(**params_spec).fit(X_train, y_train)
        tmp_train.append(mean_squared_error(y_train, tmp_model.predict(X_train)))
        tmp_test.append(mean_squared_error(y_test, tmp_model.predict(X_test)))

        # if model is DecisionTreeRegressor():
            # # tmp_train.append(mean_squared_error(y_train, tmp_model.predict(X_train)))
            # # tmp_test.append(mean_squared_error(y_test, tmp_model.predict(X_test)))
        # elif model is DecisionTreeClassifier():
            # # tmp_train.append(roc_auc_score(y_train, tmp_model.predict(X_train)))
            # # tmp_test.append(roc_auc_score(y_test, tmp_model.predict(X_test)))


    plt.plot(values, tmp_train, 'o-',color='dodgerblue', label='Train')
    plt.plot(values, tmp_test,'o-', color='tomato', label='Test')
    plt.legend()
    plt.title(hyperparam)
    # tmp_best_score = tmp_test[np.max(tmp_test)]
    # plt.axvline(tmp_best_score, color='slategrey',
                # linestyle='--',
                # label="Best {} on test: {}".format(hyperparam, round(tmp_best_score, 3)))

def plot_decision_function(model, df, x1, x2, y, colorbar=True):
    """TODO: Docstring for plot_decision_function.

    :model: TODO
    :params: TODO
    :df: TODO
    :x1: TODO
    :x2: TODO
    :returns: TODO

    """
    tmp_y = df[y]
    tmp_y_names = np.unique(tmp_y)

    tmp_df = df.loc[:, [x1, x2]]
    tmp_x, tmp_y, joint_xy = generate_mesh_grid(tmp_df, x1, x2)
    tmp_complete_mat = df.loc[:, [x1, x2, y]]
    tmp_model = model.fit(tmp_df, df[y])
    tmp_z = tmp_model.predict(joint_xy).reshape(tmp_x.shape)
    custom_colormap = LinearSegmentedColormap.from_list('lista', color_palette_sequential )
    contour_values = plt.contourf(tmp_x, tmp_y, tmp_z, cmap='coolwarm')

    if colorbar is True:
        plt.colorbar(contour_values)
    else:
        pass

    if model is DecisionTreeClassifier():
        for i in tmp_complete_mat[y].unique():
            plt.scatter(tmp_complete_mat[tmp_complete_mat[y] == i][x1],
                        tmp_complete_mat[tmp_complete_mat[y] == i][x2],
                        alpha=.5, label="{}".format(str(i)), marker=markers[i],
                        color='grey')
    else:
        plt.scatter(tmp_complete_mat[x1], tmp_complete_mat[x2], alpha=.3, marker='.', color='slategrey')
    plt.xlabel(x1)
    plt.ylabel(x2)

def plot_decision_tree(model, df, x1, x2, y):
    """TODO: Docstring for plot_decision_tree.

    :model: TODO
    :df: TODO
    :x1: TODO
    :x2: TODO
    :y: TODO
    :returns: TODO

    """
    tmp_model = model.fit(df.loc[:, [x1, x2]], df[y])
    tmp_dot = export_graphviz(tmp_model, out_file=None)
    tmp_dot = pydotplus.graph_from_dot_data(tmp_dot)
    return tmp_dot

def demo_classfication_tree(X, y, y_labels, model=DecisionTreeClassifier):
    """TODO: Docstring for demo_classfication_tree.
    :returns: TODO

    """
    clf = model(criterion='entropy', max_depth=3).fit(X, y)
    dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns,
                               class_names = y_labels, filled=True, rounded=True,
                               impurity=False)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return graph



def plot_importance(fit_model, feat_names):
    """TODO: Docstring for plot_importance.

    :fit_model: TODO
    :: TODO
    :returns: TODO

    """
    tmp_importance = fit_model.feature_importances_
    sort_importance = np.argsort(tmp_importance)[::-1]
    names = [feat_names[i] for i in sort_importance]
    plt.title("Feature importance")
    plt.barh(range(len(feat_names)), tmp_importance[sort_importance])
    plt.yticks(range(len(feat_names)), names, rotation=0)
