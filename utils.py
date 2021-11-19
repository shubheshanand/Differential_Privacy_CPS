import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random

## initializing the variables
n_homes=1405 ## number of houses
lambda_0 = 0.021
alpha = 0.1
p = 0.00436
q = 1.28
xcmax = 1.6            
K = 1  ## initial value of control parameter
T = 24 ## we need to find values for 24 hrs
v_mean = 0
v_var = 0.006
K=1
b_min=0.28
b_max=0.76
d_mean=7
d_var=3.5
psi_mean=-0.8
psi_var=0.1
omega_mean=0
omega_var=0.007

# 			     [xsT(k+1)	 ]	 [0		0		     p 	   ][xs(k)		]	   [		   q			    ]
# x(k+1) = 	 [xcT(k+1)	 ] = [0 		alpha 	 beta][xc(k)	  ] +  [omegaT(k) + bT(k)	]
# 			     [lambda(k+1)]	 [-K 	K 		   1	   ][lambda(k)]	   [		   KvT			  ]

def x_mat(n_tp=300, x_supply0=1500, DP=True, plot=True, K=1):
  x_0 = []
  for i in range(n_homes):
    x_0.append(random.random()*xcmax)

  x = []
  x.append(x_0)
  b, d, psi, beta, omega = get_b_d_psi_beta_omega()
  x_1 = np.array(b) + np.array(beta) + np.array(omega) + np.array(x_0)

  x_supply  = [0 for i in range(n_tp + 2)]
  x_cons = [0 for i in range(n_tp + 2)] 
  price = [0 for i in range(n_tp + 2)]
  x_supply[0] = x_supply0
  x_cons[0] = np.sum(x_1)
  price[0] = lambda_0

  if DP:
    x_supply_DP  = [0 for i in range(n_tp + 2)]
    x_cons_DP = [0 for i in range(n_tp + 2)] 
    price_DP = [0 for i in range(n_tp + 2)]
    x_supply_DP[0] = x_supply0
    x_cons_DP[0] = np.sum(x_1)
    price_DP[0] = lambda_0
    noise = calc_noise()

  for i in range(n_tp):
    b, d, psi, beta, omega = get_b_d_psi_beta_omega()
    x_supply[i+1] = 1000*(p*price[i] + q)
    x_cons[i+1] = alpha*x_cons[i] + np.sum(beta)*price[i] + np.sum(b) + np.sum(omega)
    price[i+1] = K*((x_cons[i]-x_supply[i])/1000+ np.sum(np.random.normal(omega_mean, omega_var, n_homes)) + price[i])
    if DP:
      x_supply_DP[i+1] = 1000*(p*price_DP[i] + q)
      x_cons_DP[i+1] = alpha*x_cons_DP[i] + np.sum(beta)*price_DP[i] + np.sum(b) + np.sum(omega) + np.sum(noise)
      price_DP[i+1] = K*((x_cons_DP[i]-x_supply_DP[i])/1000+ np.sum(np.random.normal(omega_mean, omega_var, n_homes)) + price_DP[i])
    
  
  if plot:
    plt.clf()
    error = []
    for i in range(n_tp+1):
      error.append(x_cons[i]-x_supply[i])
    if not DP:
      plt.plot(error)
      plt.savefig('results/sup_dem_mismatch.png')
    else:
      error_w_DP = []
      for i in range(n_tp+1):
        error_w_DP.append(x_cons_DP[i]-x_supply_DP[i])
      plt.plot(error_w_DP)
      plt.plot(error)
      plt.savefig('results/sup_dem_mismatch_DP.png')
  
  if not DP:
    return x_supply, x_cons, price
  
  if DP:
    return x_supply, x_cons, price, x_supply_DP, x_cons_DP, price_DP

def sigma_K(plot=True):
  error = []
  for K in range(16):
    x_supply, x_cons, price = x_mat(DP=False, plot=False, K=K/10)
    x_supply = np.array(x_supply)
    x_cons = np.array(x_cons)
    error_temp = x_supply-x_cons
    error.append(np.std(error_temp))
  if plot:
    plt.clf()
    plt.plot(error)
    plt.savefig('results/sigma_K.png')
  return error

def sigma_y(plot=True):
  x_supply = [0 for i in range(202)]
  x_cons = [0 for i in range(202)]
  price = [lambda_0 for i in range(n_homes+5)]
  x_supply[0] = 1500
  price[0] = lambda_0
  sigma = []
  x_0 = []
  K = 1
  x_s_1 = [0 for i in range(n_homes+5)]
  x_1 = x_0
  for i in range(n_homes):
    x_0.append(random.random()*xcmax)
  for i in range(100):
    b, d, psi, beta, omega = get_b_d_psi_beta_omega()
    x_supply[i+1] = 1000*(p*price[i] + q)
    error = []
    price = [lambda_0 for i in range(n_homes+5)]
    for k in range(len(x_1)):
      x_1[k] = np.array(b[k]) + np.array(beta[k])*price[i] + np.array(omega[k]) + 0.1*np.array(x_1[k])
      x_s_1[k] = 1000*(p*price[i]+q)
      error.append(x_1[k]-x_s_1[k] + np.random.normal(v_mean, v_var, 1))
    sigma.append(np.std(error))
    x_1 = np.array(b) + np.array(beta)*price[i] + np.array(omega) + np.array(x_0)
    x_s_1 = np.array([1000*(p*price[i]+q)/n_homes for l in range(n_homes)])
    error_f = x_1 - x_s_1
    x_cons[i+1] = alpha*x_cons[i] + np.sum(beta)*price[i] + np.sum(b) + np.sum(omega)
    price[i+1] = K*((x_cons[i]-x_supply[i])/1000+ np.sum(np.random.normal(omega_mean, omega_var, n_homes)) + price[i])
  if plot:
    plt.clf()
    plt.plot(sigma)
    plt.savefig('results/sigma.png')
  return sigma

def eps(plot=True):
  sigmay=[]
  epsilonT = []
  for m in range(0,16,1):
    noise = calc_noise()
    delta_o2 = 24*xcmax
    x_supply = [0 for i in range(202)]
    x_cons = [0 for i in range(202)]
    price = [lambda_0 for i in range(102)]
    x_supply[0] = 1500
    price[0] = lambda_0
    sigma = []
    x_0 = []
    x_s_1 = [0 for i in range(n_homes+5)]
    x_1 = x_0
    for i in range(n_homes):
      x_0.append(random.random()*xcmax)
    for i in range(100):
      b, d, psi, beta, omega = get_b_d_psi_beta_omega()
      x_supply[i+1] = 1000*(p*price[i] + q)
      error = []
      for k in range(len(x_1)):
        x_1[k] = np.array(b[k]) + np.array(beta[k])*price[i] + np.array(omega[k]) + 0.1*np.array(x_1[k]) + np.array(noise)
        x_s_1[k] = 1000*(p*price[i]+q)
        error.append(x_1[k]-x_s_1[k] + np.random.normal(v_mean, v_var,1))
      sigma.append(np.std(error))
      
      x_cons[i+1] = alpha*x_cons[i] + np.sum(beta)*price[i] + np.sum(b) + np.sum(omega)
      price[i+1] = m/10*((x_cons[i]-x_supply[i])/1000+ np.sum(np.random.normal(omega_mean, omega_var, n_homes)) + price[i])
    sigmay.append(sigma[98])
    
  for delta in range(1, 10):
    epsilon = []
    for sigma in sigmay:
      epsilon.append(np.sqrt(2*np.log(1.25/(delta/10)))*delta_o2/sigma/1000)
    epsilonT.append(epsilon)
  if plot:
    plt.clf()
    for i in range(1,9):
      plt.plot(epsilonT)
    plt.legend(range(1,9))
    plt.savefig('results/epsilon.png')
  return epsilonT

def calc_noise():
  v = np.random.normal(v_mean, v_var, (1, n_homes))
  rv = np.matmul(np.transpose(v), v)
  c = np.ones((n_homes, 1))
  ctrans = np.transpose(c)
  qk = 0.133
  qk = np.reshape(qk, (1, 1))
  qyk = np.matmul(c, np.matmul(qk, ctrans)) + rv
  rn = 0.354*0.354*np.eye(n_homes) - qk*qk*np.eye(n_homes) - rv
  rn_diag = np.diag(rn)
  noise = []
  for qq in range(n_homes):
    noise.append(np.random.normal(0, abs(rn_diag[qq]), 1))
  noise = np.array(noise)
  return noise

def get_b_d_psi_beta_omega():
  b = (np.random.rand(n_homes)*(b_max-b_min)) + b_min
  d = np.random.normal(d_mean, d_var, n_homes) ## generating random distributions
  psi = np.random.normal(psi_mean, psi_var, n_homes)
  beta = []
  for j in range(n_homes):
    beta.append((d[j]*psi[j]*math.pow(lambda_0, psi[j]-1))*lambda_0/1000)
  omega = np.random.normal(omega_mean, omega_var,n_homes)
  omega = np.reshape(omega, (n_homes))
  return b, d, psi, beta, omega


def main():
  x_mat()
  sigma_K()
  sigma_y()
  eps()

if __name__ == '__main__':
  main()