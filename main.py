# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:04:12 2017

@author: Rob Romijnders
For explanation, see robromijnders.github.io/EM
"""

from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from util import plot_cov_ellipse,plot_EM
from scipy.stats import multivariate_normal as mvn


#generate the parameters (collectively named theta)
mu_gen = [np.array([0., 1.5]),np.array([2.8, 1.5]),np.array([5.5, 3.0])]
sigma_gen = [np.array([[1.0, 0.9],[0.9,1.0]]),np.array([[1.0, -0.8],[-0.8,0.8]]),np.array([[1.0, 0.9],[0.9,1.0]])]
pi_gen = [0.5,0.15,0.35]

theta_gen = zip(mu_gen,sigma_gen,pi_gen)

mix_gen = [multivariate_normal(m,s) for m,s,_ in theta_gen]


colors = ['r','b','m']


# Now generate data
N = 1000
X = []

f, ax = plt.subplots(2, 2)  
for i,p in enumerate(pi_gen):
  num = int(p*N)
  data = mix_gen[i].rvs(num)
  for x in data:
    ax[0,0].plot(x[0],x[1],'.',c = colors[i],linewidth = 0.0)
  X.append(data)
X = np.concatenate(X,0)
ax[0,0].set_title('Scatter plot of data')
ax[0,0].set_xlabel('x1')
ax[0,0].set_ylabel('x2')
ax[0,0].set_ylim([-2,7])
ax[0,0].set_xlim([-2,7])


for m,s,_ in theta_gen:
  plot_cov_ellipse(s, m, ax=ax[0,0])
  



class fitGMM(plot_EM):
  def __init__(self,X,K = 3,max_iter = 200,ax=None):
    self.K = K
    self.X = X
    self.max_iter = max_iter
    
    self.N, self.D = X.shape
    assert self.N > self.D
    
    #Random initialization
    self.phi = np.random.rand(K)
    self.phi /= np.sum(self.phi)
    
    #Randomly initialize means as existing samples
    self.mu = self.X[np.random.choice(self.N,self.K,replace=False)]  
#    self.mu = np.random.rand(self.K,self.D)*5.0-1.0
    self.sigma = np.zeros((self.K,self.D,self.D))
    for k in range(self.K):
      A = np.random.randn(self.D,self.D)
      self.sigma[k] = np.dot(A.T,A)
      assert np.all(np.linalg.eigvals(self.sigma[k]) > 0)
    
    self.expLL = np.zeros((max_iter,))
    
    self.Post = np.zeros((self.N,self.K))/self.K  #posteriors Bishop eq.9.13
    
    self.i = 0
    
    self.plot=False
    if ax is not None:
      assert np.sum(ax.shape) > 2, 'Expected an array of axes'
      self.ax = ax
      self.plot=True
    
  def fit(self):
    while self.i < self.max_iter:
      self.E_step()
      self.M_step()
      if (self.expLL[self.i] - self.expLL[self.i-1])**2 < 1E-1:
        if self.i>2:
          break
      print('Step %3.0f expLL %10.2f'%(self.i, self.expLL[self.i]))
      if self.plot:
        
        self.plot_LL((0,0),(1,0))
        self.plot_LL((2,0),(1,1))
        self.plot_GMM()
      self.i += 1
      
    self.theta = zip(list(self.mu),list(self.sigma),list(self.phi))
    return self.expLL
    
  def E_step(self):
    #Implements Bishop eq.9.23
    for n in range(self.N):
      for k in range(self.K):
        llike = mvn.logpdf(self.X[n],self.mu[k],self.sigma[k])
        #Update the expected log likelihood
        self.expLL[self.i] += self.Post[n,k]*llike
        #update the posterior
        self.Post[n,k] = self.phi[k]*np.exp(llike)
      
    #and normalize
    self.Post /= np.expand_dims(np.sum(self.Post,axis=1),1)
    return
    
    
  def M_step(self):
    Nk = np.sum(self.Post,axis=0) #Bishop eq.9.27
    self.mu = (np.dot(self.X.T,self.Post)/Nk).T #Calculates Bishop eq.9.24
    for k in range(self.K):  #Calculates Bishop eq.9.25
      Xm = self.X - self.mu[k]
      self.sigma[k] = 1/Nk[k]*np.dot(Xm.T*self.Post[:,k],Xm)
      assert np.all(np.linalg.eigvals(self.sigma[k]) > 0)
    self.phi = Nk/np.sum(Nk)  #Calculates Bishop eq.9.26
    return

    

fitgmm = fitGMM(X,ax=ax)    
expLL = fitgmm.fit()

#Now go to the directory and run  (after install ImageMagick)
#  convert -delay 10 -loop 0 *.png EM.gif

