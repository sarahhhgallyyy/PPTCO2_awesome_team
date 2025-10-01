# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:13:02 2025

@author: 20212148
"""
#add this after params_fit for standard deviation

# After fitting
params_fit = res.x

# Residual variance estimate
n = len(res.fun)
p = len(res.x)
dof = max(1, n - p)
residual_variance = np.sum(res.fun**2) / dof

# Covariance matrix and standard deviations
J = res.jac
cov = np.linalg.inv(J.T @ J) * residual_variance
param_std = np.sqrt(np.diag(cov))

print("Fitted parameters:", params_fit)
print("Standard deviations:", param_std)
