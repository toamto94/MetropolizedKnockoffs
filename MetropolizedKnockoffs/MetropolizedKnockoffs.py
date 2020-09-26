import numpy as np
import pandas as pd
from scipy import random
from scipy.stats import multivariate_normal


class Ising_Knockoff_Sampler:
    def __init__(self, Z, Theta):
        self.Z = Z
        self.Theta = Theta
        
    def joint_energy(self, Z, Zj, knockoffs):
        index = len(knockoffs)
        Z[index] = Zj
        energy = Z.T.dot(self.Theta).dot(Z)
        for i in range(index):
            for k in range(self.Theta.shape[0]):
                if i != k:
                    energy += knockoffs[i] * Z[k] * self.Theta[i, k]
        energy += knockoffs.T.dot(self.Theta[:index, :index].dot(knockoffs))
        return(np.exp(energy))
    
    def probas(self, Zj, knockoffs):
        joint_energy_plus = self.joint_energy(Zj, 1 ,knockoffs)
        joint_energy_minus = self.joint_energy(Zj, -1, knockoffs)
        p = joint_energy_plus/(joint_energy_plus + joint_energy_minus)
        return(p)
    
    def sample_point(self, p):
        u = np.random.uniform()
        if u <= p:
            return(1)
        else:
            return(-1)
    
    def sample_row(self, j, knockoffs):
        Zj = self.Z[j, :]
        if len(knockoffs) == len(Zj):
            return(knockoffs)
        else:
            p = self.probas(Zj, knockoffs)
            knockoffs = np.hstack((knockoffs, self.sample_point(p)))
            return(self.sample_row(j, knockoffs))
        
    def sample(self):
        Z_tilde = np.zeros_like(self.Z)
        for i in range(self.Z.shape[0]):
            Z_tilde[i, :] = self.sample_row(i, np.array([]))
        return(Z_tilde)
        
        
        
        
class Gaussian_Knockoff_Sampler:
    def __init__(self, Y, mu=None, cov=None):
        self.Y = Y
        self.mu = mu
        self.cov = cov
        eigv = np.linalg.eig(Y.T.dot(Y))[0]
        s = np.min(cov.diagonal()) - np.random.uniform(0,0.001)
        s = np.array([s] * Y.shape[1])
        self.mu_c = np.hstack((mu, mu))
        self.G = np.block([[cov, cov-np.diag(s)], [cov-np.diag(s), cov]])
        
    def conditional_gaussian(self, i, knockoffs):
        k = len(knockoffs)
        Yi = self.Y[i, :]
        YX = np.hstack((Yi, knockoffs))
        p = self.Y.shape[1]
        mu_k = self.mu_c[k+p:]
        muX = self.mu_c[:k+p]
        cov11 = self.G[k+p:, k+p:]
        cov12 = self.G[:k+p, k+p:]
        cov22 = self.G[:k+p, :k+p]
        
        mu_tilde = mu_k + cov12.T.dot(np.linalg.pinv(cov22)).dot(YX - muX)
        G_tilde = np.linalg.pinv(np.linalg.pinv(self.G)[k+p:, k+p:])
        return(mu_tilde[0], G_tilde[0, 0])
    
    def propose(self, i, knockoffs):
        mu, cov = self.conditional_gaussian(i, knockoffs)
        return(mu, cov, np.random.normal(mu, cov, 1))
    
    def evaluate_probability(self, i, knockoffs):
        mu_tilde, cov_tilde, y = self.propose(i, knockoffs)
        k = len(knockoffs)
        p = self.Y.shape[1]
        Yi = self.Y[i, :]
        YX = np.hstack((Yi, knockoffs))
        YX_proposal = YX.copy()
        YX_proposal[k] = y
        mu = self.mu_c[:k+p]
        G = self.G[:k+p, :k+p]
        rv_joint = multivariate_normal(mu, G)
        rv_cond = multivariate_normal(mu_tilde, cov_tilde)
        return((rv_cond.pdf(YX[k]) / rv_cond.pdf(y)) * (rv_joint.pdf(YX_proposal) / rv_joint.pdf(YX)), y, YX[k])
    
    def sample_knockoff(self, i, knockoffs):
        prob, proposal, original = self.evaluate_probability(i, knockoffs)
        if prob >= 1:
            knockoffs = np.hstack((knockoffs, proposal))
            return(knockoffs)
        else:
            u = np.random.uniform()
            if u <= prob:
                knockoffs = np.hstack((knockoffs, proposal))
                return(knockoffs)
            else:
                knockoffs = np.hstack((knockoffs, original))
                return(knockoffs)
            
    def sample_row(self, i, knockoffs):
        if len(knockoffs) == self.Y.shape[1]:
            return(knockoffs)
        else:
            knockoffs = self.sample_knockoff(i, knockoffs)
            return(self.sample_row(i, knockoffs))
        
    def sample(self):
        Y_tilde = np.zeros_like(self.Y)
        for i in range(Y_tilde.shape[0]):
            Y_tilde[i, :] = self.sample_row(i, np.array([]))
        return(Y_tilde)