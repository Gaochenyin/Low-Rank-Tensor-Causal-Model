#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 21:09:14 2022

@author: chyga
"""

import numpy as np
import tensorly as tl
# from utils import P_Omega, truncate_vec
from sklearn.utils.extmath import randomized_svd
from tensorly import tucker_to_tensor
import random
import scipy
from scipy.special import expit
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels
from functools import partial
import matplotlib.pyplot as plt
import itertools
# handy functions 
def P_Omega(T, Omega):
    out = T.copy()
    # set the missing (unobserved) equals to zero
    if (~Omega).sum != 0:
        out[~Omega] = 0
    return out

# data generation for causal inference
def gen_data_potential_Y_observed(seed, N, T, d0 = 20, k = 5, case = 4):
    
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    K = 2**k
    X0_unscaled = np.random.normal(size = (N, d0))
    X0 = np.hstack((np.ones((N,1)), X0_unscaled))
    # generate the full potential outcome framework
    Y_full = gen_data_potential_Y_trucated(N = N, T = T, K = K,
                                           X0 = X0, case = case)
    # generate the observed data based on Y_full
    ## generate X, A and Y
    X = np.zeros((N ,T+k), dtype = 'float');
    A = np.zeros((N, T+k), dtype = 'int')
    Y_ts = np.zeros((N, T, K))
    Omega = np.zeros((N, T, K), dtype = 'int')
    # eta = 0.5 
    # for generate Y
    eta = 0.5
    for i in range(N):
        for t in range(T+k):
            # generate X
            if t == 0:
                ## stage 1
                X[i,t] = X0[i,].mean() #np.sum(X0[i,:]) #+ 0.5*np.random.normal(size = 1)
            elif t == 1:
                ## stage 2
                X[i,t] = X0[i,].mean() + A[i,t-1] * eta # + 0.5*np.random.normal(size = 1)
            elif t == 2:
                # stage 3
                X[i,t] = X0[i,].mean() + A[i,t-1] * eta + \
                    A[i,t-2] * eta/2 #+ 0.5*np.random.normal(size = 1)
            else:
                # stage > 3
                X[i,t] = X0[i,].mean() + A[i,t-1] * eta + \
                    A[i,t-2] * eta/2 +\
                    A[i,t-3] * eta/4 #+ 0.5*np.random.normal(size = 1)
            # generate A
            # tend to be assigned
            # add some confounder
            if case == 4 or case == 2: # more correlated treatment assignment
                A[i,t] = np.random.binomial(1, expit(2 * X[i,t] + np.array([0 if t-1<0 else 1]) * 2 * X[i, t-1]
                                                                     ))
            else: # less correlated treatment assignment
                A[i,t] = np.random.binomial(1, expit(X[i,t] + np.array([0 if t-1<0 else 1]) * X[i, t-1]
                                                                     ))
            # generate Yobs
            history = np.zeros(k, dtype = 'int')
            for idx in range(np.min([t+1, k])):
                history[k-idx-1] = A[i, t-idx]
            hist_conv = int(''.join(str(e) for e in history), 2)
            
            # only add low rank after t pass k
            if t > (k-1):
                Y_ts[i, t-k, hist_conv] = Y_full[i, t-k, hist_conv] + np.random.normal(0)
                Omega[i, t-k, hist_conv] = 1

    Omega = Omega.astype('bool')
    # truncate the first k time points
    X = X[:, k:]
    A = A[:, k:]
    XX = X[:, :, np.newaxis]

    # construct the observed data
    Y_obs = Y_ts * Omega
    
    return Y_obs, XX, A, Omega, X0, Y_full


def gen_data_potential_Y_trucated(N, T, K,
                                  X0 = None, case = 4):
    
    # d = 4
    # k = np.int(np.log2(B))
    # for generate X
    eta = 0.5 
    # for generate Y
    beta = 1
    # generate the potential outcomes under all treatment history
    Y_full = np.zeros((N, T, K), dtype = 'float')
    for i in range(N):
        for t in range(T):
            for l in range(K):
                # initialize flag
                # flag = 0
                # generate X[i,t, :]
                history_b = '{0:05b}'.format(l)
                # z =  np.random.normal(0, 1, 2)
                u0 = int(history_b[-2]) * eta + \
                    int(history_b[-3]) * eta/2 + \
                        int(history_b[-4]) * eta/4 #+z[0]
                u1 = int(history_b[-3]) * eta + \
                    int(history_b[-4]) * eta/2 + \
                        int(history_b[-5]) * eta/4 #+z[1]
                        
                # simple outcome
                if case == 1 or case == 2:
                    Y_full[i, t, l] =  X0[i, ].sum() * 4 + \
                        (u0 + X0[i,].mean()) * beta + \
                            (u1 + X0[i,].mean()) *beta + \
                                int(history_b[-1]) * 3 +\
                                int(history_b[-2]) * 3
                                
                # complex outcomes
                if case == 3 or case == 4:
                    # Legendre polynomial
                    def P(n, x): 
                        if(n == 0):
                            return 1 # P0 = 1
                        elif(n == 1):
                            return x # P1 = x
                        else:
                            return (((2 * n)-1)*x * P(n-1, x)-(n-1)*P(n-2, x))/float(n)
                    
                    # J = 5
                    Y_baseline = 1
                    for deg in range(1, 2):
                        Y_baseline += P(deg, X0[i, 1:]).mean() # delete the intercept column
                    
                    
                    Y_full[i, t, l] =  P(1, X0[i, 1:]).sum() * 4 * (1/2**t) + \
                        Y_baseline * 4 + \
                        X0[i, ].sum() * sum([int(h) for h in history_b]) * 3  + \
                            X0[i, ].sum() * sum([int(h) for h in history_b[1:]]) * (u0 + X0[i,].mean()) + \
                                X0[i, ].sum() * sum([int(h) for h in history_b[:-1]]) * (u1 + X0[i,].mean()) +\
                        X0[i, ].sum() * int(history_b[-1]) * 2 + X0[i, ].sum() * int(history_b[-2]) + \
                        X0[i, ].sum() * (u0 + X0[i,].mean()) * beta * 5 + X0[i, ].sum() * (u1 + X0[i,].mean()) * beta * 3 + \
                        (u0 + X0[i,].mean()) * beta + \
                            (u1 + X0[i,].mean())*beta + \
                                int(history_b[-1]) * 3 +\
                                int(history_b[-2]) * 3
    return Y_full

# traditional MSM
def TC_MSM(Aobs, Yobs, X0, X, k = 5, wts = None):
    N, T = Yobs.shape
    if X0 is None:
        X0 = np.eye(N)/np.sqrt(N)
    # estimate the inverse treatment probabilities
    if wts is None:
        wts = sw_estimation_univariate(Aobs = Aobs, X = X, k = k)
    reg = {}
    # for every t fit a logistic regression
    for t in range(T):
        reg[t] = linear_model.LinearRegression(fit_intercept=True)
        reg[t].fit(np.hstack((Aobs[:,max(0, t-k+1):t+1], X0)),
                Yobs[:,t], sample_weight = wts[:,t])
        
        # variance estimation by sandwitch formula
        X_mat = np.hstack((Aobs[:,max(0, t-k+1):t+1], X0))
        M_score = (wts[:,t] * (Yobs[:,t] - reg[t].predict(X = X_mat)))[:, None] * X_mat
        reg[t].cov_beta = np.linalg.pinv((wts[:,t, None] * X_mat).T @ (wts[:,t, None] * X_mat)/N) @ \
            (M_score.T @ M_score / N) @ \
            np.linalg.pinv((wts[:,t, None] * X_mat).T @ (wts[:,t, None] * X_mat)/N)
    # impute for each entries of the Y
    Y_hat_MSM = np.zeros((N, T, 2**k))
    # var_MSM = np.zeros((N, T, 2**k))
    for i in range(N):
        for t in range(T):
            for b in range(2**k):
                op = '{0:0'+str(k)+'b}'
                history_b = op.format(b)
                # flag = b < 2**(t+1)
                coef_temp = np.hstack(([int(history_b[-(min(t+1,k))+j]) for j in range(min(t+1,k))],
                           X0[i,:])).reshape((1,-1))
                Y_pred = reg[t].predict(coef_temp)
                # var_pred = coef_temp @ reg[t].cov_beta @ coef_temp.T
                # if the entry is flaggedb
                # if flag:
                Y_hat_MSM[i, t, b] = Y_pred
                # var_MSM[i, t, b] = var_pred
                # else:
                #     Y_hat_MSM[i, t, b] = np.nan
    return Y_hat_MSM, reg

def sw_estimation_univariate(Aobs, X, k = 5, pool = True):
    N, T = Aobs.shape
    
    if pool:
        glm_PS = sm.GLM(Aobs.ravel(order = 'F'),
               X.ravel(order = 'F'),
               family=sm.families.Binomial())
        res_A = glm_PS.fit()
        # predict the PS
        ps_hat = res_A.predict(X.ravel(order = 'F')).reshape((N,-1), order = 'F')
        wts_hat = 1./ps_hat
    else:
        res_A = {}
        # fit the logistic model for Aobs at t
        for t in range(T):
            # for A at time 0
            glm_PS = sm.GLM(Aobs[:, t], X[:, t],
                            family=sm.families.Binomial())
            try:
                res_A[t] = glm_PS.fit()
            except statsmodels.tools.sm_exceptions.PerfectSeparationError:
                pass
        # estimate the weights or the stablized weights
        wts_hat = np.ones((N, T), dtype = 'float')
        for i in range(N):
            for t in range(T):
                for p in range(min(t+1, k)):
                    prob_denom = res_A[t - p].predict(X[i, t - p])
                    if Aobs[i, t - p] == 1:  # should it be Aobs[i, t-p] == 1:
                        denom = prob_denom
                    else:
                        denom = 1 - prob_denom
                if i % 100 == 0 and t == 0:
                    print(f'{i}-th unit of sw estimation')
                wts_hat[i, t] *= (1 / denom)
    return wts_hat


def sw_estimation_calibration(Yobs, Aobs, XX, X0, Omega, k, penalty = False):
    N, T, d = XX.shape
    _, _, K = Yobs.shape
    import scipy.optimize
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import KFold
        # return np.array([10**(-6) if xi>np.log(np.finfo('float').max) else  np.exp(xi)/(1+np.exp(xi)) for xi in x])
        # if x>np.log(np.finfo('float').max):
        #     return 10**(-6)
        # else:
        #     return
    # define the calibration conditions
    def cal_contd_obj(coef,
                  XX_t, A_t,
                  regularization):
        # compute the weight
        linear_pred = np.dot(XX_t, coef)
        loss_CAL = A_t * np.exp(-linear_pred) + (1 - A_t) * linear_pred
        # loss_ML = np.log(1 + np.exp(linear_pred)) - A_t * linear_pred
        # not regularize on the intercept
        def SCAD_penalty(coef, regularization, a = 3.7):
            coef_abs = np.abs(coef)
            penalty = (coef_abs < regularization) * coef_abs * regularization +\
            ((coef_abs > regularization) & (coef_abs < a*regularization)) * \
                ((a*regularization*(coef_abs-regularization)-\
                 (coef_abs**2-regularization**2)/2)/(a-1) + regularization**2) +\
            (coef_abs > a*regularization)*((a-1)*regularization**2/2 + regularization**2)
            return penalty.sum()
        obj = loss_CAL.mean() - \
                SCAD_penalty(coef = coef[1:],
                              regularization = regularization,
                              a = 3.7) 
                    # regularization * np.linalg.norm(coef[1:], ord = 1) # lasso penalty 
        
        # XX_t_condt = np.multiply((A_t * weight_cal1)[:, None],
        #                    XX_t).sum(axis = 0)
        # XX_t_sum = XX_t.sum(axis = 0)
        ## add penalization
        # obj = np.sum(np.square(XX_t_sum - XX_t_condt)) + lam * np.linalg.norm(coef, ord = 1)
        
        return obj
    def cal_contd_grad(coef,
                  XX_t, A_t,
                  regularization):
        
        N = XX_t.shape[0]
        # compute the weight
        linear_pred = np.dot(XX_t, coef)
        obj_grad_CAL = (A_t * expit(linear_pred) -1).dot(XX_t)/N
        # obj_grad = (expit(linear_pred) - A_t).dot(XX_t)/N
        
        def l1_norm_grad(coef, regularization):
            return np.sign(coef) * regularization
        
        def SCAD_penalty_grad(coef, regularization, a = 3.7):
            coef_abs = np.abs(coef)
            penalty_grad = regularization * ((coef_abs < regularization) + \
            ((coef_abs > regularization) & (coef_abs < a*regularization)) * \
                (a*regularization - coef_abs)/((a-1)*regularization))
            return np.sign(coef) * penalty_grad
        penalty_grad = SCAD_penalty_grad(coef = coef[1:],
                          regularization = regularization,
                          a = 3.7) 
        # penalty_grad = l1_norm_grad(coef = coef[1:],
        #                   regularization = regularization)
        return (obj_grad_CAL - np.insert(penalty_grad, 0, 0))
        # cal0 = np.multiply(1-A_t[:, None], XX_t_cal).sum(axis = 0)
    
    def weight_contd_grad(coef,
                          XX_t, A_t):
        
        N = XX_t.shape[0]
        # compute the weight
        linear_pred = np.dot(XX_t, coef)
        # obj_grad_CAL = (A_t/expit(linear_pred) -1).dot(XX_t)/N
        obj_grad_CAL = (A_t / expit(linear_pred) -1).dot(XX_t)/N
        # obj_grad_CAL = (expit(linear_pred) - A_t).dot(XX_t)/N
        return obj_grad_CAL
    # def weight_contd_hess(coef,
    #                       XX_t, A_t):
    #     N = XX_t.shape[0]
    #     # compute the weight
    #     linear_pred = np.dot(XX_t, coef)
    #     # obj_hess_CAL = -np.matmul(A_t * (1-expit(linear_pred))/ (expit(linear_pred)) *\
    #     #                          XX_t.T, XX_t)/N
    #     obj_hess_CAL = -np.matmul(A_t / (np.exp(linear_pred)) *\
    #                               XX_t.T, XX_t)/N
    #     # by minorization–maximization algorithm
    #     return obj_hess_CAL #- penalty_grad*(coef/(10**(-6)+coef))
        
    def SCAD_penalty_grad(coef, regularization, a = 3.7):
        coef_abs = np.abs(coef)
        penalty_grad = regularization * ((coef_abs < regularization) + \
        ((coef_abs > regularization) & (coef_abs < a*regularization)) * \
            (a*regularization - coef_abs)/((a-1)*regularization))
        return penalty_grad #* np.sign(coef)# * penalty_grad
     

    # print(weights_coef_init, '\n')
    # cross-validation
    # lam_candidate = np.linspace(0, 5)
    # # for t in range(T):
    # kf = KFold(n_splits = 3)
    # for train_idx, test_idx in kf.split(range(N)):
    #     print(train_idx)
        
    def run_cal_contd(t):
        # combine all the information before
        XX_t_cal = XX[:, max(t - k + 1, 0):(t+1), :]
        XX_t_cal = XX_t_cal.reshape(N, -1)
        XX_t_cal = normalize(XX_t_cal, axis = 1, norm = 'l1')   
        # combine it with the baseline covariate 
        # including the intercept at first index (not penalized)
        if X0 is None:
            XX_t_cal = np.hstack((np.ones((N,1)), XX_t_cal)) 
        else:
            XX_t_cal = np.hstack((X0, XX_t_cal)) 
        A_t_pre = Aobs[:, max(t - k, 0):t]
        Y_t_pre = Yobs[:, max(t - k + 1, 0):t,:].sum(axis = 2)
        if t > 0:
            # if time is larger than 0, normalization
            pass
            # Y_t_pre = normalize(Y_t_pre, axis = 1, norm = 'l1')   
        # augment the intercept
        # XX_t_all = np.hstack([XX_t_cal, A_t_pre])
        XX_t_all = np.hstack([XX_t_cal, A_t_pre])
        # balancing group
        A_t = Aobs[:, t]
        # for coef 1
        # for cross validation
        lambdas = np.linspace(0.01, 0.1, num = 10)/N
        kf = KFold(n_splits = 5, shuffle = True)
        cv_scores = []
        maxit = 100
        for lam in lambdas:
            
            weights_coef_init = np.ones(XX_t_all.shape[1])/XX_t_all.shape[1]  
            kf.get_n_splits(range(N))
            
            k_fold_scores = []
            f = 1
            for train_index, test_index in kf.split(XX_t_all):
                # Training data
                CV_X = XX_t_all[train_index,:]
                CV_A = A_t[train_index]
                # print(train_index.shape)
                # Holdout data
                holdout_X = XX_t_all[test_index,:]
                holdout_Y = A_t[test_index]
                
                
                if penalty:
                    weights_coef = scipy.optimize.minimize(fun = cal_contd_obj,
                                  x0 = weights_coef_init,
                                  args = (CV_X,
                                          CV_A, 
                                          lam),
                                  jac = cal_contd_grad,
                                  method = 'CG',
                                  options = {"maxiter": 500}).x
                else:
                    weights_coef = scipy.optimize.root(fun = partial(weight_contd_grad,
                                                      XX_t = CV_X,
                                                      A_t = CV_A),
                                        x0 = weights_coef_init,
                                        # jac = partial(weight_contd_hess,
                                        #               XX_t = CV_X,
                                        #               A_t = CV_A),
                                        method = 'linearmixing').x
                
                    # print(weights_coef_init, '\n')
                
            
                # weights_coef_init = scipy.optimize.minimize(lambda x: cal_contd_obj(x, XX_t = CV_X,
                #                                       A_t = CV_A, regularization = lam),
                #                   weights_coef_init,
                #                   method = 'CG',
                #                   options = {"maxiter": 500}).x
                
                # Calculate holdout error
                fold_mape = weight_contd_grad(weights_coef, XX_t = holdout_X,
                                           A_t = holdout_Y)
                k_fold_scores.append(sum(fold_mape**2))
                # print("Fold: {}. Error: {}".format( f, fold_mape))
                f += 1
            lambda_scores = np.mean(k_fold_scores)
            # print("LAMBDA AVERAGE: {}".format(lambda_scores))
            cv_scores.append(lambda_scores)
        # print(cv_scores)
        # print(cv_scores)
        lambda_opt = lambdas[np.argmin(cv_scores)]
        
        # for coef 1
        
        if penalty:
            weights_coef0 = scipy.optimize.minimize(fun = cal_contd_obj,
                                  x0 = weights_coef,
                                  args = (XX_t_all,
                                          1 - A_t, 
                                          lambda_opt),
                                  jac = cal_contd_grad,
                                  method = 'CG',
                                  options = {"maxiter": 500}).x
            weights_coef1 = scipy.optimize.minimize(fun = cal_contd_obj,
                                  x0 = weights_coef,
                                  args = (XX_t_all,
                                          A_t, 
                                          lambda_opt),
                                  jac = cal_contd_grad,
                                  method = 'CG',
                                  options = {"maxiter": 500}).x
        else:
            weights_coef1 = scipy.optimize.root(fun = partial(weight_contd_grad,
                                              XX_t = XX_t_all,
                                              A_t = A_t),
                                x0 = weights_coef_init,
                                # jac = partial(weight_contd_hess,
                                #               XX_t = XX_t_all,
                                #               A_t = A_t),
                                method = 'linearmixing').x
            
            weights_coef0 = scipy.optimize.root(fun = partial(weight_contd_grad,
                                              XX_t = XX_t_all,
                                              A_t = 1 - A_t),
                                x0 = weights_coef_init,
                                # jac = partial(weight_contd_hess,
                                #               XX_t = XX_t_all,
                                #               A_t = 1 - A_t),
                                method = 'linearmixing').x
        
        # weight_coef1 = scipy.optimize.minimize(lambda x: cal_contd_obj(x, XX_t = XX_t_all,
        #                                               A_t = A_t, regularization = lambda_opt),
        #                           np.zeros(XX_t_all.shape[1]),
        #                           method = 'CG').x
        # for coef 0
        # coordinate descent algorithm (weight_contd_grad)
        # customize method?
        
        
        
        
        # weight_coef0 = scipy.optimize.minimize(lambda x: cal_contd_obj(x, XX_t = XX_t_all,
        #                                               A_t = 1 - A_t, regularization = lambda_opt),
        #                           np.zeros(XX_t_all.shape[1]),
        #                           method = 'CG').x
        
        # change XX_t_all to include all potential treatment regimes
        # K = 2**k
        # weights = np.zeros((N, K))
        # for b in range(K):
        #     history_b = '{0:05b}'.format(b)
        #     A_b = [int(e) for e in history_b]
            
        #     weights[:, b] = expit(np.dot(np.hstack([XX_t_cal, np.tile(A_b[-1+max(-k, -t):-1], (N, 1))]),
        #                  weights_coef1)) * np.repeat(A_b[-1], N) + \
        #         expit(np.dot(np.hstack([XX_t_cal, np.tile(A_b[-1+max(-k, -t):-1], (N, 1))]),
        #                  weights_coef0)) * np.repeat(1 - A_b[-1], N)
            
            
            
        # history = np.zeros(k, dtype='int')
        # for idx in range(np.min([k, t + 1])):
        #     history[k - idx - 1] = A_it[i, t - idx]
        # str1 = ''.join(str(int(e)) for e in history)
        # p = np.int(str1, 2)
        weights = expit(np.dot(XX_t_all, weights_coef1)) * A_t + \
            expit(np.dot(XX_t_all, weights_coef0))* (1-A_t)
            
        # weights = expit(np.dot(XX_t_all, weights_coef1)) * (1-A_t) + \
        #     expit(np.dot(XX_t_all, weights_coef0))* A_t
            
        return weights
    propensity_mat = np.vstack([run_cal_contd(t) for t in range(T)]).T
    # propensity_mat = np.stack([run_cal_contd(t) for t in range(T)], axis = 2)
    # propensity_mat = np.transpose(propensity_mat, (0, 2, 1))
    
    wts_hat = 1/propensity_mat
    # wts_hat = propensity_mat
    # ------change PS to tensor
    # # propensity_marginal = np.mean(Omega, axis = 0)
    # # estimate the weights
    # wts_hat = np.zeros_like(Omega, dtype = 'float')
    # N, T, _ = wts_hat.shape
    # for i in range(N):
    #     for t in range(T):
    #         # history = np.zeros(k, dtype = 'int')
    #         # for idx in range(np.min([t+1, k])):
    #         #     history[k-idx-1] = Aobs[i, t-idx]
    #         #     hist_conv = int(''.join(str(e) for e in history), 2)
    #             # only add low rank after t pass k
    #             # stablized weights
    #         hist_conv = np.nonzero(Omega[i,t,:])[0].astype('int')
    #         # print(hist_conv)
    #         wts_hat[i, t, hist_conv] = 1/propensity_mat[i, t]#\
    #             # propensity_marginal[t, hist_conv]/\
    #             # (np.product(propensity_mat[i, max((t-k+1),0):t]))
    
    # # estimate the weights
    # wts_hat = np.ones((N, T), dtype='float')
    # for i in range(N):
    #     for t in range(T):
    #         for p in range(min([t+1, k])):
    #             wts_hat[i, t] *= weights_mat[i, t-p]
    #         wts_hat[i, t] = np.array([1 if np.isinf(wts_hat[i, t]) or np.isnan(wts_hat[i, t])  else wts_hat[i, t]]) # in-line cap for large value
    # # normalization and inverse
    # wts_hat = normalize(wts_hat, axis = 1, norm = 'l1')    
    # wts_hat = wts_hat**(-1)
    
    # wts_hat = normalize(wts_hat, axis = 1, norm = 'l2') 
    # for i in range(N):
    #     for t in range(T):
    #         wts_hat[i, t] = np.array([0 if np.isinf(wts_hat[i, t]) or np.isnan(wts_hat[i, t])  else wts_hat[i, t]]) # in-line cap for large value
            
    return wts_hat

# proposed model
class TensorCompletionCovariateProjected():
    """
    Tensor Completion projected on the covariate space (similar to STEFA)
    """
    def __init__(self,
                 Y, X0, Omega,
                 W_tensor,
                 # optimization parameter
                 stepsize = 1e-10,
                 niters = 5000, tol = 1e-8,
                 r1_list = [4], r2_list = [2], r3_list = [8],
                 # split = False, 
                 trimm = False, tensorly = False, verbose = True):
        """
        stepsize: stepsize for projected gradient descent
        tau: the number of iterations
        """
        
        self.Y = Y
        N, _, _ = Y.shape
        # self.N = N; self.T = T; self.K = K
        if X0 is None:
            self.name = 'Tucker'
            X0 = np.eye(N)
        else: 
            self.name = 'Co-Tucker'
        self.X0 = X0
        self.Omega = Omega
        self.W_tensor = W_tensor
        self.stepsize = stepsize
        self.niters = niters
        self.tol = tol
        # self.split = split
        self.trimm = trimm
        self.tensorly = tensorly
        
        self.r1_list = r1_list
        self.r2_list = r2_list
        self.r3_list = r3_list
        
        self.verbose = verbose
    # BIC criterion
    def BIC(self, F_core, U_1, U_2, U_3):
        Y_hat = tl.tenalg.multi_mode_dot(F_core,
                                      [U_1,
                                       U_2, U_3])
        P1 = np.linalg.norm(self.Y * self.W_tensor - Y_hat)**2/np.product(self.Y.shape) # estimation error
        P2 = np.log(np.product(self.Y.shape))/np.product(self.Y.shape) * \
            (np.product(F_core.shape) + np.product(U_1.shape) + np.product(U_2.shape) + np.product(U_3.shape)-\
             -U_1.shape[1]**2 - U_2.shape[1] **2 - U_3.shape[1]**2)
        return P1+P2
    
    def _TC_Tucker_project(self, Y, A, W, X, 
                          r1, r2, r3,
                          stepsize = 1e-10,
                          niters = 1000):
        # trimm for weights larger than 1/0.1
        if self.trimm:
            A[W>1/.1] = 0
            W[W>1/.1] = 0
            
        Y_w = Y * A * W
        N, T, K = Y.shape
        

        
        P_X0 = X @ np.linalg.pinv(X.T@X) @ X.T
        # initialization
        U_1, _, _ = randomized_svd(tl.unfold(Y_w, mode = 0), 
                                n_components = r1,
                                random_state = None)
        U_1 = P_X0 @ U_1
        # U_1 =  np.linalg.pinv(X.T@X) @ X.T @ U_1
        U_2, _, _ = randomized_svd(tl.unfold(Y_w, mode = 1), 
                                n_components = r2,
                                random_state = None)
        U_3, _, _ = randomized_svd(tl.unfold(Y_w, mode = 2), 
                                n_components = r3,
                                random_state = None)
        # find the optimal G
        def optimal_G(U_1, U_2, U_3):
            num = np.zeros(shape = r1*r2*r3)
            den = np.zeros(shape = [r1*r2*r3, r1*r2*r3])
            for i in range(N):
                for t in range(T):
                    l_nonzero_idx = A[i,t,:].nonzero()
                    if len(l_nonzero_idx[0]) == 0:
                        pass
                    else:
                        l_it = l_nonzero_idx[0][0]
                        temp1 = np.einsum('i,j,k', U_1[i,:], U_2[t,:], U_3[l_it,:])
                        num += (Y_w[i,t,l_it] * temp1).flatten(order = 'C')
                        den += np.outer(W[i,t,l_it] * temp1.flatten(order = 'C'),
                                        temp1.flatten(order = 'C').T)
                    
            F_core = (np.linalg.inv(den) @ num).reshape([r1, r2, r3], order = 'C') 
            return F_core
        
        # # compute the mu_max for the factor matrices
        mu_1 = max(np.linalg.norm(U_1, axis = 1))**2*N/r1
        mu_2 = max(np.linalg.norm(U_2, axis = 1))**2*T/r2
        mu_3 = max(np.linalg.norm(U_3, axis = 1))**2*K/r3
        mu_max = max(mu_1, mu_2, mu_3)
        # # compute the L0
        L_max = np.max(Y_w)
        
        # F_core = optimal_G(U_1, U_2, U_3)
        F_core = tl.tenalg.multi_mode_dot(Y_w,
                              [(U_1).T, U_2.T, U_3.T],
                              [0, 1, 2])
        
        F_core, [U_1, U_2, U_3] = tl.decomposition.tucker(Y_w, n_iter_max = niters, 
                                                          rank = [r1, r2, r3],
                                                          # svd = 'truncated_svd',
                                                          )
        
        def P_Omega(T, Omega):
            T[Omega==0] = 0
            return T
        loss = []
        tol_temp = 1e3
        
        if self.tensorly:
            F_core, [U_1, U_2, U_3] = tl.decomposition.tucker(Y_w, n_iter_max = niters, 
                                                              rank = [r1, r2, r3],
                                                              # svd = 'truncated_svd',
                                                              verbose = self.verbose,
                                                              tol = 10e-8) 
            return (F_core, U_1, U_2, U_3), loss
        else:        
            for it in range(niters):
                Y_pre = tucker_to_tensor((F_core, [(U_1), 
                                                        U_2,
                                                        U_3]))
                loss_pre =  Y_pre - Y 
                loss_pre =  P_Omega(loss_pre, A) * W
                
                # else: # no split for gradient descent
                # grad for core F
                grad_f_X_unflod_mode1 = tl.unfold(loss_pre, mode = 0)
                grad_f_X_unflod_mode2 = tl.unfold(loss_pre, mode = 1)
                grad_f_X_unflod_mode3 = tl.unfold(loss_pre, mode = 2)
                
                
                grad_f_F_fold = tl.tenalg.multi_mode_dot(loss_pre,
                                       [(U_1).T, U_2.T, U_3.T],
                                      # [U_1.T, U_2.T, U_3.T],
                                      [0, 1, 2])
                # grad for U1
                # X.T @ 
                grad_U_1 = grad_f_X_unflod_mode1 @ \
                    tl.tenalg.kronecker([U_2, U_3]) @\
                        tl.unfold(F_core, 0).T
                # grad for U2
                grad_U_2 = grad_f_X_unflod_mode2 @ \
                tl.tenalg.kronecker([(U_1), U_3]) @\
                        tl.unfold(F_core, 1).T
                            # tl.tenalg.kronecker([U_1, U_3]) @\
                        # tl.tenalg.kronecker([(X @ U_1), U_3]) @\
                # grad for U3
                grad_U_3 = grad_f_X_unflod_mode3 @ \
                    tl.tenalg.kronecker([(U_1), U_2]) @\
                        tl.unfold(F_core, 2).T
                            # tl.tenalg.kronecker([(X @ U_1), U_2]) @\
                                # tl.tenalg.kronecker([U_1, U_2]) @\
                def otimes(A):
                    return A @ A.T
                
                #------------Gradient descent
                # update
                F_core = F_core - stepsize * grad_f_F_fold
                U_1 = U_1 - stepsize * grad_U_1
                U_1 =  X @ np.linalg.pinv(X.T@X) @ X.T @ U_1 # projected GD
                U_2 = U_2 - stepsize * grad_U_2
                U_3 = U_3 - stepsize * grad_U_3   
                # F_core = optimal_G(U_1, U_2, U_3)
                
                # project the factor matrices onto the restricted set
                U_1_2max = np.linalg.norm(U_1, axis = 1)**2*N/r1
                U_1[(U_1_2max > mu_max).nonzero()[0], :] = U_1[(U_1_2max > mu_max).nonzero()[0], :]/\
                    U_1_2max[(U_1_2max > mu_max).nonzero()[0], None] * mu_max
                
                U_2_2max = np.linalg.norm(U_2, axis = 1)**2*T/r2
                U_2[(U_2_2max > mu_max).nonzero()[0], :] = U_2[(U_2_2max > mu_max).nonzero()[0], :]/\
                    U_2_2max[(U_2_2max > mu_max).nonzero()[0], None] * mu_max
                    
                U_3_2max = np.linalg.norm(U_3, axis = 1)**2*K/r3
                U_3[(U_3_2max > mu_max).nonzero()[0], :] = U_3[(U_3_2max > mu_max).nonzero()[0], :]/\
                    U_3_2max[(U_3_2max > mu_max).nonzero()[0], None] * mu_max
                
                # project G onto the restricted set
                G_norm2_max = np.array([np.linalg.norm(tl.unfold(F_core, 0), ord = 2),
                          np.linalg.norm(tl.unfold(F_core, 1), ord = 2),
                          np.linalg.norm(tl.unfold(F_core, 2), ord = 2)]).max()
                G_norm2_limit = L_max * np.sqrt(N*T*K/(mu_max**3/2*(r1*r2*r3)**(1/2)))
                F_core = F_core/G_norm2_max*G_norm2_limit
                
    
                # after iteration
                Y_after = tucker_to_tensor((F_core, [(U_1), 
                                                        U_2,
                                                        U_3]))
                loss_after =  Y_after - Y 
                loss_after =  P_Omega(loss_after, A)*W
                loss.append(np.linalg.norm(loss_pre))
                
                tol_temp = np.abs(np.linalg.norm(loss_pre) - np.linalg.norm(loss_after))/\
                    np.linalg.norm(loss_pre)
                
                if not (it % 1000) and self.verbose:
                # if verbose:
                    print(f'({self.name}): {it}th iteration with loss: {round(loss[-1], 2)} and update {round(tol_temp, 2)}')
                    # print()
                if tol_temp < self.tol:
                    break
            # compute the covariate-independent loadings
            # Q_1 = tl.unfold(tl.tenalg.multi_mode_dot(F_core,
            #                           [U_2, U_3],
            #                           [1, 2]), mode = 0)
            # U_1 = tl.unfold(Y, mode = 0) @ Q_1.T @ np.linalg.pinv(Q_1 @ Q_1.T)
            # F_core = optimal_G(U_1, U_2, U_3)
            # multiply the permutation matrices to make U_1, U_2, U_3 orthogonal            
            U_1, R_1 = np.linalg.qr(U_1)
            U_2, R_2 = np.linalg.qr(U_2)
            U_3, R_3 = np.linalg.qr(U_3)
            F_core = tl.tenalg.multi_mode_dot(F_core, [R_1,R_2,R_3])
            return (F_core, U_1, U_2, U_3), loss
    # Sequentially tuning
    def _tuning_bic(self, r1, r2, r3, verbose = True):
        (F_core, U_1, U_2, U_3), loss  = \
            self._TC_Tucker_project(Y = self.Y, A = self.Omega, 
                                    W = self.W_tensor, X = self.X0, 
                                    r1 = r1, r2 = r2, r3 = r3,
                                    stepsize = self.stepsize,
                                    niters = self.niters)
        bic = self.BIC(F_core, U_1, U_2, U_3)
        return bic
    
    def SequentialTuning(self, out = True):
        # initialization
        r1_list = self.r1_list
        r2_list = self.r2_list
        r3_list = self.r3_list
        
        bic = 1e10
        r1_opt = np.random.choice(r1_list);
        r2_opt = np.random.choice(r2_list);
        r3_opt = np.random.choice(r3_list);
        
        
        # if length = 1
        if len(r1_list) == 1:
            pass
        else:
            # for rank1  
            bic_list = [self._tuning_bic(r1, r2_opt, r3_opt) for r1 in r1_list]
            if np.min(bic_list) < bic:
                bic = np.min(bic_list)
                r1_opt = r1_list[np.argmin(bic_list)]
            
            
        # if length = 1
        if len(r2_list) == 1:
            pass
        else:
            # for rank1  
            bic_list = [self._tuning_bic(r1_opt, r2, r3_opt) for r2 in r2_list]
            if np.min(bic_list) < bic:
                bic = np.min(bic_list)
                r2_opt = r2_list[np.argmin(bic_list)]
            
        
        # if length = 1
        if len(r3_list) == 1:
            pass
        else:
            # for rank1  
            bic_list = [self._tuning_bic(r1_opt, r2_opt, r3) for r3 in r3_list]
            if np.min(bic_list) < bic:
                bic = np.min(bic_list)
                r3_opt = r3_list[np.argmin(bic_list)]
            
        (F_core, U_1, U_2, U_3), loss = self._TC_Tucker_project(Y = self.Y, A = self.Omega, 
                                W = self.W_tensor, X = self.X0, 
                                r1 = r1_opt, r2 = r2_opt, r3 = r3_opt,
                                stepsize = self.stepsize,
                                niters = self.niters)
        self.bic = bic
        self.r1_opt = r1_opt
        self.r2_opt = r2_opt
        self.r3_opt = r3_opt
        self.Y_hat = tl.tenalg.multi_mode_dot(F_core,
                                              [(U_1), # already add the projection self.X0 @ 
                                               U_2, U_3])
        self.loss = loss
        
        # store the loading matrices
        self.F_core = F_core
        self.U_1 = U_1; self.U_2 = U_2; self.U_3 = U_3
        
        if out:
            return self.Y_hat 

# gradient descent for matrix completion
def MC_GradientDescent(Y_t, W_t, Omega_t, X0, rank = 5,
                step_size = 1e-8, niters = 500, tol = 1e-5):
    
    if step_size is None:
        flag = 1
    else: 
        flag = 0
    # init_step = step_size
    N,_ = Y_t.shape
    # initialize
    if X0 is None:
        X0 = np.eye(N)/np.sqrt(N)
    U_t, Sigma_t, V_t = randomized_svd((Y_t* W_t),
                   n_components = rank,
                   random_state = None)
    U_t = np.linalg.pinv(X0.T@X0) @ X0.T @ U_t
    Sigma_t = np.diag(Sigma_t)
    V_t = V_t.T
    # step_size = 1e-8
    # gradient descent
    for it in range(niters):
        # loss_pre = P_Omega(W_t*(Y_t - X0 @ U_t @ Sigma_t @ V_t.T), Omega_t)
        loss_pre = P_Omega((Y_t - X0 @ U_t @ Sigma_t @ V_t.T), Omega_t)
        # construct the loss function
        # loss_gradient = W_t * Omega_t * (X0 @ U_t @ Sigma_t @ V_t.T - Y_t)
        loss_gradient = Omega_t * (X0 @ U_t @ Sigma_t @ V_t.T - Y_t)
        
        U_gradient = X0.T @ loss_gradient @ V_t @ Sigma_t 
        V_gradient = loss_gradient.T @ X0 @ U_t @ Sigma_t
        S_gradient = U_t.T @ X0.T @ loss_gradient @ V_t
        

        if flag and not (it%100):
            lip_1 = np.linalg.norm(W_t * Omega_t) *\
                np.linalg.norm(U_t.T @ X0.T@ X0@ U_t) * np.linalg.norm(V_t.T @ V_t)
            lip_2 = np.linalg.norm(W_t * Omega_t) *\
                np.linalg.norm(X0 @ U_t @ Sigma_t @ V_t.T)**2
            step_size = min(1/lip_1, 1/lip_2)
            # print(step_size)
        
        # update
        U_t = U_t - step_size * U_gradient
        V_t = V_t - step_size * V_gradient
        Sigma_t = Sigma_t - step_size * S_gradient
        
        # compute the loss
        # loss_after = P_Omega(W_t * (Y_t - X0 @ U_t @ Sigma_t @ V_t.T), Omega_t)
        loss_after = P_Omega((Y_t - X0 @ U_t @ Sigma_t @ V_t.T), Omega_t)
        tol_temp = np.abs(np.linalg.norm(loss_pre) - np.linalg.norm(loss_after))/\
            np.linalg.norm(loss_pre)
        
        if not it % 1000:
            print(f'(MC-Gradient){it}-th iteration: loss {round(np.linalg.norm(loss_after), 2)} and update {round(tol_temp,2)}')
        if tol_temp < tol:
            break
    return X0 @ U_t @ Sigma_t @ V_t.T

# plot of the data in tensor formats
def plot3D_tensor(Y,
                  N, T, k, ax,
                  cmp = plt.get_cmap('bwr'),
                  title_main = None,
                  vmin = -400, vmax = 800):        
    # plot of potential outcomes
    x = range(1, (N+1)); y = range(1, (T+1)); z = range(1, (2**k+1))           
    points = []
    for element in itertools.product(x, y, z):
        points.append(element)
    xi, yi, zi = zip(*points)

    # select out exact zero entries (i.e. missing)
    xi_obs = [xi[i] for i in np.where((Y.flatten()!=np.nan))[0]]
    yi_obs = [yi[i] for i in np.where((Y.flatten()!=np.nan))[0]]
    zi_obs = [zi[i] for i in np.where((Y.flatten()!=np.nan))[0]]
    Y_obsi = [Y.flatten()[i] for i in np.where((Y.flatten()!=np.nan))[0]]
    
    # full
    p1 = ax.scatter3D(xi_obs, yi_obs, zi_obs, c = Y_obsi, 
               cmap = cmp,
               alpha = 0.5, s = 50,
               vmin = vmin, vmax = vmax)
    # fig.colorbar(p1, ax = ax, shrink = 0.5, aspect = 5)
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    # labels
    ax.set_xlabel('subject')
    ax.set_ylabel('time')
    ax.set_zlabel('treatment regime')
    ax.set_title(title_main)
    return p1
    # plt.show()