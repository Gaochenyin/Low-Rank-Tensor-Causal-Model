#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 19:45:13 2022

@author: hp
"""
import pandas as pd
import numpy as np
import tensorly as tl
from data_gen import gen_data_potential_Y_observed
from utils import TC_MSM, sw_estimation_calibration
from COSTCO import CovariateAssistedTensorCompletion
from MC_utils import MC_gradient
from utils_TC_cov import TensorCompletionCovariateProjected, P_Omega, trt_effect_avg
import sys

def simulation(seed, N, T, case, d0 = 20, k = 5):
    if case in [1, 2]: r1 = 4
    if case in [6, 7]: r1 = 6
    # generate the observed potential outcomes
    Y_obs, XX, A, Omega, X0, Y_full = gen_data_potential_Y_observed(seed = seed, N = N, T = T,
                                  d0 = d0, k = k, case = case)
    
    # W = sw_estimation_univariate(Aobs = A, X = X, k = k, pool = True)
    W = sw_estimation_calibration(Yobs = Y_obs, Aobs = A, XX = XX, X0 = None,
                                  Omega = Omega, k = 2)
    # W = W[:,:, np.newaxis] # algin with the dimension
    
    W = np.repeat(W[:,:, np.newaxis], 2**k, axis = 2) * Omega

    # Legendre polynomial
    # from sklearn.preprocessing import normalize
    def P(n, x): 
        if(n == 0):
            return 1 # P0 = 1
        elif(n == 1):
            return x # P1 = x
        else:
            return (((2 * n)-1)*x * P(n-1, x)-(n-1)*P(n-2, x))/float(n)
    # construct 10th order Legendre polynomial  basis function as Phi_X
    to_stack = []
    to_stack.append(
       np.ones(shape=(N, 1)))
    for deg in range(1, (6+1)):
        to_stack.append(
            np.apply_along_axis(lambda x: P(deg, x), 1, X0)
            )
    # change for different number of basis
    Phi_X = np.hstack(to_stack[:(2+1)])
    # OR decomposition
    Phi_X, _ = np.linalg.qr(Phi_X)
    # ---------------------------------
    # # matricization completion
    Y_MC_hat_unfold = MC_gradient(Y_t = tl.unfold(Y_obs, 0),
                                  W_t = tl.unfold(W, 0),
                                  Omega_t = tl.unfold(Omega, 0),  
                                  X0 = Phi_X,
                                  rank = 8,
                                  step_size = 1/np.linalg.norm(Y_obs)*1e-4, 
                                  niters = 5000, tol = 1e-8)
    
    Y_MC_hat_fold = tl.fold(Y_MC_hat_unfold, 0, Y_obs.shape)
    
    # loss_MC_fold.append(np.linalg.norm(P_Omega(Y_full- Y_MC_hat_fold, Omega))/\
    #                np.linalg.norm(P_Omega(Y_full, Omega)))
    # ---------------------------------
    
    # loss_MC_fold.append(np.linalg.norm(P_Omega(Y_full- Y_MC_hat_fold, ~Omega))/\
    #                 np.linalg.norm(P_Omega(Y_full, ~Omega)))

    # # matrix-wise completion
    # Y_hat_MC = np.zeros_like(Y_obs)
    # for t in range(T):
    #     Y_hat_sliced = MC_gradient(Y_obs[:,t,:],
    #                 W[:,t,:],
    #                 Omega[:,t,:], X0, rank = 5,
    #                 step_size = 1e-10, niters = 5000)
    #     Y_hat_MC[:,t,:] = Y_hat_sliced
    
    
    # Y_obs_reaxised = np.moveaxis(Y_obs, -1, 0)
    # for i, Y_mat in enumerate(Y_obs_reaxised):
    #     Y_mat[Y_mat==0] = np.nan
    #     Y_hat_MC[:, :, i] = MC_softimp(Y_mat, tau = 5)
        # MC_SNN(Y_mat)
    # loss_MC.append(np.linalg.norm(Y_full- Y_hat_MC)/np.linalg.norm(Y_full))
    # ---------------------------------
    # MSM tensor completion
    Y_hat_MSM, _ = TC_MSM(Aobs = A, Yobs = np.apply_along_axis(sum, 2, Y_obs),
            X0 = Phi_X, X = np.apply_along_axis(sum, 2, XX),
            wts = np.apply_along_axis(sum, 2, W * Omega))
    # loss_MSM.append(np.linalg.norm(P_Omega(Y_full- Y_hat_MSM, Omega))/\
    #                np.linalg.norm(P_Omega(Y_full, Omega)))
    # IPTWTC
    # Y_hat_IPTWTC,_ = TC_IPTWTC(Y_ts = Y_obs, Aobs = A, XX = X0, W = W)
    # ---------------------------------
    # COSTCO
    Omega_ones = np.ones(Y_obs.shape).astype('bool')
    COSTCO = CovariateAssistedTensorCompletion(Y = Y_obs, X = Phi_X, 
                                               A = Omega,
                                                Omega = Omega_ones,
                                                r_list =[8],
                                                s_list = [1],
                                                tau = 5000, tol = 1e-8,
                                                weight_external = W)
    Y_hat_COSTCO = COSTCO.SequentialTuning(verbose = False) 
    # loss_COSTCO.append(np.linalg.norm(P_Omega(Y_full- Y_hat_COSTCO, Omega))/\
    #                np.linalg.norm(P_Omega(Y_full, Omega)))
    # ---------------------------------
    # projection and approximation
    PTEFA = TensorCompletionCovariateProjected(Y = Y_obs, X0 = Phi_X, Omega = Omega,
                                        W_tensor = W, stepsize = 1/(np.linalg.norm(Y_obs)) * 1e-4,
                                        niters = 10000, tol = 1e-8,
					r1_list = [r1], 
					r2_list = [2],
					r3_list = [8])
    Y_hat_project_RDA = PTEFA.SequentialTuning()
    
    
    # projection and approximation without covariates
    PTEFA_eyes = TensorCompletionCovariateProjected(Y = Y_obs, X0 = None, Omega = Omega,
                                        W_tensor = W, stepsize = 1/(np.linalg.norm(Y_obs)) * 1e-4,
                                        niters = 10000, tol = 1e-8,
					r1_list = [r1], 
					r2_list = [2],
					r3_list = [8])
    Y_hat_project_RDA_eyes = PTEFA_eyes.SequentialTuning()
    # Y_hat_approx = TC_Tucker_approx(Y = Y_obs, A = Omega, W = W, X = X0)
    
    # # power iteration for Tucker
    # Y_hat_power, _ = TC_Tucker_power(Y = Y_obs, X = X0,
    #                 niters = 1000,
    #                 r_list = [10, 10, 10], 
    #                 verbose = True, tail = False)
    # np.linalg.norm(Y_full- Y_hat_power)/np.linalg.norm(Y_full)
    # # print
    # loss_Tucker_proj.append(np.linalg.norm(P_Omega(Y_full- Y_hat_project_RDA, Omega))/\
    #                np.linalg.norm(P_Omega(Y_full, Omega)))
    # ---------------------------------
    
    # loss_IPTWTC.append(np.linalg.norm(Y_full- Y_hat_IPTWTC)/np.linalg.norm(Y_full))
    # loss_COSTCO.append(np.linalg.norm(Y_full- Y_hat_COSTCO)/np.linalg.norm(Y_full))
    # loss_Tucker_approx.append(np.linalg.norm(Y_full - tl.tenalg.multi_mode_dot(Y_hat_approx[0],
    #                                                                  [X0 @ Y_hat_approx[1],
    #                                                                   Y_hat_approx[2],
    #                                                                   Y_hat_approx[3]]))/np.linalg.norm(Y_full))
    # print('Tucker_proj:', loss_Tucker_proj[-1], 'Tucker_approx:', loss_Tucker_approx[-1],
    #       'IPTWTC:', loss_IPTWTC[-1],
    #       'COSTCO:', loss_COSTCO[-1])
    
    
    # save the simulated results
    # np.save(f'./result/case{case}/Omega_{N}_{T}_{seed}.npy', Omega)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/Y_N{N}_T{T}_{seed}_full.npy', Y_full)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/TC_hat_N{N}_T{T}_{seed}_eyes.npy', Y_hat_project_RDA_eyes)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/TC_hat_N{N}_T{T}_{seed}.npy', Y_hat_project_RDA)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/MC_hat_N{N}_T{T}_{seed}.npy', Y_MC_hat_fold)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/MSM_hat_N{N}_T{T}_{seed}.npy', Y_hat_MSM)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/COSTCO_hat_N{N}_T{T}_{seed}.npy', Y_hat_COSTCO)

    ate_full = trt_effect_avg(Y_full)
    ate_MC = trt_effect_avg(Y_MC_hat_fold)
    ate_MSM = trt_effect_avg(Y_hat_MSM)
    ate_COSTCO = trt_effect_avg(Y_hat_COSTCO)
    ate_TC = trt_effect_avg(Y_hat_project_RDA)
    ate_TC_eyes = trt_effect_avg(Y_hat_project_RDA_eyes)

    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/ate_N{N}_T{T}_{seed}_full.npy', ate_full)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/ate_TC_hat_N{N}_T{T}_{seed}_eyes.npy', ate_TC_eyes)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/ate_TC_hat_N{N}_T{T}_{seed}.npy', ate_TC)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/ate_MC_hat_N{N}_T{T}_{seed}.npy', ate_MC)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/ate_MSM_hat_N{N}_T{T}_{seed}.npy', ate_MSM)
    np.save(f'/gpfs_common/share01/statistics/cgao6/tensor_causal/result/case{case}/ate_COSTCO_hat_N{N}_T{T}_{seed}.npy', ate_COSTCO)


if __name__ == '__main__':
    
    from functools import partial
    import multiprocessing


    # only the tensor is in low-rank, but the matrix is not
    # signal strength in Zhang and Xia (2018)
    N_list_1 = list(np.repeat(300, 6)); T_list_1 = [10, 20, 30, 40,50, 60]; 
    N_list_2 = [100, 150, 200, 250, 350]; T_list_2 = list(np.repeat(10, 5));
    N_list = N_list_1+ N_list_2; T_list = T_list_1 + T_list_2
    # N_list = [100, 150, 200, 250, 300, 350]; T_list = list(np.repeat(10, 6));
    case = int(sys.argv[1])
    niters = 100

    for N, T in zip(N_list, T_list):
        # construct a multi-processor
        with multiprocessing.Pool(8, initargs = ()) as pool:
        	main_NT = partial(simulation, N = N, T = T, case = case)
        	pool.map(main_NT, range(niters))
        	pool.close()
        	pool.join()   
         