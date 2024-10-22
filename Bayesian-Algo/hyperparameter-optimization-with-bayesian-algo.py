import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import scipy.stats as st
import math
import random


def bayesian_parameter_optimization(objec_func, parameter_space):
    
    def gaussian_kernel (fea_vec1, fea_vec2, sigma=0.2, l=0.5):
        m,n = fea_vec1.shape[0], fea_vec2.shape[0]
        dist_matri = np.zeros((m,n), dtype=float)
        for i in range(m):
            for j in range(n):
                dist_matri[i][j]=((fea_vec1[i]-fea_vec2[j])**2)
        res = sigma**2*np.exp(-0.5/l**2*dist_matri)
        return res
    def GP_update(x, x_star):
        x =np.asarray(x)
        x_star=np.asarray(x_star)
        k_xstar_x = gaussian_kernel(x_star,x)
        k_x_x = gaussian_kernel(x,x)
        k_xstar_xstar = gaussian_kernel(x_star,x_star)
        k_x_xstar = k_xstar_x.T
        k_x_x_inv = np.linalg.inv(k_x_x + 1e-8*np.eye(len(x)))
        mu_star = k_xstar_x @ k_x_x_inv @ y
        cov_star = k_xstar_xstar - k_xstar_x @ k_x_x_inv@k_x_xstar
        return mu_star, cov_star
    
    def acquisition(x, x_star):
        mu, cov = GP_update(x,x_star)
        yhat = mu.ravel()
        std = 1.96*np.sqrt(np.diag(cov_star))
        upper=yhat+std
        max_at = np.argmax(upper)
        return x_star[max_at]
    initial_point_num=5
    x=np.array(random.sample(list(parameter_space),initial_point_num)).reshape(-1,1)
    y_li=[]
    for i in x.tolist():
        y=objec_func(i[0])
        y_li.append(y)
    y=np.array(y_li).reshape(-1,1)
    x_star = parameter_space.reshape(-1,1)
    mu_star , cov_star = GP_update(x,x_star)
    y_star = mu_star.ravel()
    uncertainty = 1.96*np.sqrt(np.diag(cov_star))
    i=1
    while True:
        new_x = acquisition(x, x_star)
        new_y = objec_func(new_x)
        i+=1
        if abs(new_y-y[-1][0])<=0.001:
            print(f'the optimal point is {new_x} and the optimal value is {new_y}',f'this is {i-1} interation')
            return (new_x,new_y)
        elif i>100:
            print(f'the optimal point is {new_x} and the optimal value is {new_y}',f'this is {i-1} interation')
            return (new_x,new_y)
        else:
            x = np.vstack((x, new_x))
            y = np.vstack((y, new_y))
            mu_star , cov_star = GP_update(x,x_star)
            y_star = mu_star.ravel()
            uncertainty = 1.96*np.sqrt(np.diag(cov_star))
