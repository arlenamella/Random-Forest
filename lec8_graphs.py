#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: lec8_graphs.py
Author: Ignacio Soto Zamorano
Email: ignacio[dot]soto[dot]z[at]gmail[dot]com
Github: https://github.com/ignaciosotoz
Description: Ancilliary Files for Bagging and Random Forests Algorithms - adl
"""

import re
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score, accuracy_score, roc_auc_score
import seaborn as sns


markers = ['o', '^', '*','H', 'P', 'D', 'X', 'h', 'p', 'd', 'c']
model_list = ['RandomForestClassifier', 'RandomForestRegressor', 'BaggingRegressor', 'BaggingClassifier']
fetch_lims = lambda x: [np.floor(np.min(x)), np.ceil(np.max(x))]
count_valid_model_class = lambda x: [True if re.search(x, i, re.IGNORECASE) else False for i in model_list].count(True)


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
        np.linspace(np.min(tmp_X[x2]), np.max(tmp_X[x2]), num=100),
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
        param_spec = {hyperparam: i}
        tmp_model = model.set_params(**param_spec).fit(X_train, y_train)
        tmp_train.append(mean_squared_error(y_train, tmp_model.predict(X_train)))
        tmp_test.append(mean_squared_error(y_test, tmp_model.predict(X_test)))

    plt.plot(values, tmp_train, 'o-', color='dodgerblue', label='Train')
    plt.plot(values, tmp_train, 'o-', color='tomato', label='Test')
    plt.legend()
    plt.title(hyperparam)

def plot_decision_function(model, df, x1, x2, y, colorbar=True):
    """TODO: Docstring for plot_decision_function.

    :model: TODO
    :df: TODO
    :x1: TODO
    :x2: TODO
    :y: TODO
    :colorbar: TODO
    :returns: TODO

    """

    colors= ['dodgerblue', 'tomato']
    tmp_model = model.fit(df.loc[:, [x1, x2]], df[y])
     # tmp_get_x_1 = fetch_lims(df[x1])
     # tmp_get_x_2 = fetch_lims(df[x2])
    tmp_x_mesh, tmp_y_mesh, tmp_joint_xy = generate_mesh_grid(df, x1, x2)
     # randomforest no tiene el atributo decision_function, hay que hardcodearlo
    tmp_joint_xy = model.predict_proba(np.c_[tmp_x_mesh.ravel(),
                                              tmp_y_mesh.ravel()])[:, 1].reshape(tmp_x_mesh.shape)

     # for	i in df[y].unique():
     # plt.scatter(df[df[y] == i][x1],
     # df[df[y] == i][x2], 
     # marker='.', alpha=.3, color=colors[i]
     # label="{} = {}".format(y, i))
     # plt.legend()
    plt.contourf(tmp_x_mesh, tmp_y_mesh, tmp_joint_xy, cmap='coolwarm')

    plt.xlabel(x1)
    plt.ylabel(x2)


def plot_importance(fit_model, feat_names):
    """TODO: Docstring for plot_importance.

    :fit_model: TODO
    :feat_names: TODO
    :returns: TODO

    """
    tmp_importance = fit_model.feature_importances_
    sort_importances = np.argsort(tmp_importance)[::-1]
    names = [feat_names[i] for i in sort_importances]
    plt.title('Feature importance')
    plt.barh(range(len(feat_names)), tmp_importance[sort_importances])
    plt.yticks(range(len(feat_names)), names, rotation=0)


def plot_bootstrap(distribution=np.random.normal, n_sims=5000):
    """TODO: Docstring for plot_bootstrap.

    :distribution: TODO
    :n_sims: TODO
    :returns: TODO

    """
    x_dist = distribution(size=n_sims)
    x_min, x_max = fetch_lims(x_dist)
    x_axis = np.linspace(x_min, x_max, n_sims)
    population_density = stats.gaussian_kde(x_dist)
    population_density = np.reshape(population_density(x_axis).T, x_axis.shape)

    tmp_array = np.array(n_sims)
    for _ in range(n_sims):
        tmp_array = np.append(tmp_array,
                           np.random.choice(x_dist, size=len(x_dist) - 1, replace=True))

    bootstraped_density = stats.gaussian_kde(tmp_array)
    bootstraped_density = np.reshape(bootstraped_density(x_axis).T, x_axis.shape)

    plt.plot(x_axis, population_density, label="Densidad Poblacional", color='dodgerblue', lw=3)
    plt.plot(x_axis, bootstraped_density, label='Densidad Bootstrap', color='tomato', lw=3, linestyle='--')
    plt.title("Muestras realizadas: {}".format(n_sims))
    plt.legend()

def plot_bagging_behavior( scores, metric,  n_sims):
    tmp_x_range = ['RT: {}'.format(i) for i in n_sims]
    plt.plot(scores, 'o--', lw=1,color='dodgerblue', label=r'RegTree')
    plt.axhline(metric,color='tomato', label=r'Bagging')
    plt.xticks(range(len(tmp_x_range)),tmp_x_range, rotation=90)
    plt.legend();


def plot_between_trees_correlation(model, X_test):
    """TODO: Docstring for plot_between_trees_correlation.

    :model: TODO
    :returns: TODO

    """
    store_rho = []

    tmp_model_trees = model.estimators_

    for i in tmp_model_trees:
        for j in tmp_model_trees:
            store_rho.append(stats.pearsonr(i.predict(X_test),
                                            j.predict(X_test))[0])

    store_rho = np.array(store_rho).reshape(len(tmp_model_trees), len(tmp_model_trees))
    sns.heatmap(store_rho, cmap='coolwarm', annot=True)
