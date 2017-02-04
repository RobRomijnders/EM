# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:24:14 2017

@author: rob
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
    """

    import numpy as np
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    ax.add_artist(ellip)  
    
    
class plot_EM():
  def plot_GMM(self):
    self.ax[0,1].cla()
    self.theta = zip(list(self.mu),list(self.sigma),list(self.phi))
    for m,s,_ in self.theta:
      plot_cov_ellipse(s, m, ax=self.ax[0,1])
    self.ax[0,1].set_xlim([-2,7]) 
    self.ax[0,1].set_ylim([-2,7])
    self.ax[0,1].set_title('Current exp ll %8.1f'%self.expLL[self.i])
    plt.savefig('im/step%05d.png'%self.i)
  def plot_LL(self,x = (0,0), ax_tup = (1,0)):
    self.ax[ax_tup].cla()
    mu_store = self.mu[x].copy()
    for mx1 in np.linspace(-2,7,30):
      self.mu[x] = mx1
      ll = self.calc_LL()
      ell = self.calc_ELL()
      self.ax[ax_tup].plot(mx1,ll,'.',linewidth = 0.0,c='r')
      self.ax[ax_tup].plot(mx1,ell,'.',linewidth = 0.0,c='b')
    self.ax[ax_tup].set_xlim([-2 , 7])
    self.ax[ax_tup].set_ylim([-15000,-1500])
    self.ax[ax_tup].set_xlabel('Changing x%.0f of mu%.0f'%(x[1],x[0]))
    self.mu[x] = mu_store.copy()

  def calc_LL(self):
    #calculates bishop eq.9.14
    prob = np.zeros((self.N,))
    for k in range(self.K):
      like = mvn.pdf(self.X,self.mu[k],self.sigma[k]) 
      prob += like*self.phi[k]
    log_prob = np.log(prob)
    return np.sum(log_prob)
  def calc_ELL(self):
    #Calculates Bishop eq.9.40
    ell = 0.0
    for k in range(self.K):
      llike = mvn.logpdf(self.X,self.mu[k],self.sigma[k])+np.log(self.phi[k])
      ell += np.dot(self.Post[:,k],llike.T)
    return ell
    
    
"""Old rubbish code"""
#def loglikelihood(X,pi,mu,sigma):
#  """Calculates the log-likelihood for the data X given the parameters
#  This function implements eq.9.14
#  input
#  - X: data with each sample a column
#  - pi,mu,sigma: lists of length K containing the parameters"""
#  assert isinstance(pi,list) and isinstance(mu,list) and isinstance(sigma,list)
#  assert len(pi) == len(mu) == len(sigma)
#  assert len(X.shape) == 2
#  assert X.shape[0]>X.shape[1]
#  
#  theta = zip(mu,sigma,pi)
#  mix = [multivariate_normal(m,s) for m,s,_ in theta]
#  prob = []
#  for i,p in enumerate(pi):
#    prob_k = mix[i].pdf(X)
#    prob.append(prob_k*p)
#  prob = np.sum(prob,axis=0)
#  log_prob = np.log(prob)
#  return np.sum(log_prob)
#
##print('Log likelihood of the data under the true parameters is %.3f'%loglikelihood(X,pi_gen,mu_gen,sigma_gen))

      
    