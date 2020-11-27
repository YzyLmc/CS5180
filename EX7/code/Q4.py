#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:24:04 2020

@author: ziyi
"""


import gym
from tiles3 import tiles, IHT
import numpy as np
import random

class mntCar():
    
    def __init__(self, t_num = 4096):
        self.aSpace = [2,0,1]
        self.w = np.zeros(t_num)
        self.env = gym.make('MountainCar-v0')
        self.env._max_episode_steps = 600
        self.iht = IHT(t_num)
        
    
    def totile(self,state, a):
        #print(state,a)
        return tiles(self.iht, 8, [8*state[0]/1.7, 8*state[1]/0.14], [a])
    
    def oneEps(self, epsilon = 0, alpha = 0.5/8, gamma = 0.95):
        self.env.reset()
        done = False
        state = self.env.state

        qLs = []
        for a in self.aSpace:
            agg = self.totile(state, a)
            qi = 0
            for tile in agg:
                qi += self.w[tile]
            #print(self.w, agg)
            qLs.append(qi)
        qmax = max(qLs)
        actLs = []
        for a in self.aSpace:
            agg = self.totile(state, a)
            qi = 0
            for tile in agg:
                qi += self.w[tile]
            if qi == qmax:
                actLs.append(a)
        #print(qi,qmax)
        #print(actLs)
        act = random.choice(actLs)
            
        itr = 0
        while done == False:               
            agg = self.totile(state, act)
            q = 0
            for tile in agg:
                q += self.w[tile]
            
            #print(agg,self.w)
            #print(q)
            
            newState, reward, done, emtDict = self.env.step(act)
            
            if done == True:
                #print(self.w)
                for tile in agg:
                    self.w[tile] += alpha*(reward - q)
                
            
            else:
                
                qLs = []
                for a in self.aSpace:
                    agg = self.totile(newState, a)
                    qi = 0
                    for tile in agg:
                        qi += self.w[tile]
                    qLs.append(qi)
                
                qmax = max(qLs)
                actLs = []
                for a in self.aSpace:
                    agg = self.totile(newState, a)
                    qi = 0
                    for tile in agg:
                        qi += self.w[tile]

                    if qi == qmax:
                        actLs.append(a)
                act1 = random.choice(actLs)
                agg1 = self.totile(newState, act1)
               
                q1 = 0
                for tile in agg1:
                    q1 += self.w[tile]

                for tile in agg:
                    self.w[tile] += alpha*(reward + gamma*q1 - q)
            state = newState
            act = act1
            itr += 1
        return itr
                
    def sarsa(self, eps = 500):
        self.env.reset()
        e = 0
        epsLs = []
        while e < eps:
            n_step = self.oneEps()
            epsLs.append(n_step)
            e += 1
        
        return epsLs
        

if __name__ == '__main__':
    mc = mntCar()
    epsLs = mc.sarsa()
    avgLs = [0] * len(epsLs)
    for i in range(len(epsLs)):
        stIdx = max(0,i-100)
        avg = sum(epsLs[stIdx:i+1])/(i + 1 -stIdx)
        avgLs[i] = avg
    import matplotlib.pyplot as plt
    
    fig,ax = plt.subplots()

    ax.set_xlabel('episodes')
    ax.set_ylabel('steps per episode')
    ax.plot(avgLs)
    plt.show()
#%%    
from mpl_toolkits.mplot3d import Axes3D
def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)
estimate = np.zeros([100,100])
stepi = 1.7/100
stepj = 0.14/100
for i in range(100):
    for j in range(100):
        emax = -1000
        for a in mc.aSpace:                  
            agg = mc.totile([stepi*i,stepj*j],a)
            q1 = 0
            for tile in agg:
                q1 += mc.w[tile]
            emax = max(emax,-q1)
        estimate[i,j] = emax
(fig, ax, surf) = surface_plot(estimate, cmap=plt.cm.coolwarm)

fig.colorbar(surf)

ax.set_xlabel('n_Cars(A)')
ax.set_ylabel('n_Cars(B)')
ax.set_zlabel('values')

plt.show()  
    
        
        
        