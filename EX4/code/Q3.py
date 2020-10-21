#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:01:50 2020

@author: ziyi
"""

import gym
import numpy as np
import random

class blackj():
    
    def __init__(self):
        self.env = gym.make('Blackjack-v0')
        self.aSpace = [0,1]
        self.aMap = np.ones([21,10,2])
        self.returnMap = [[[[] for i in range(2)]for i in range(10)]for i in range(21)]
        self.qMap = np.zeros([21,10,2,2])
        self.aCount = np.zeros([21,10,2,2])
        self.vMap = np.zeros([21,10,2])
        
    def newGame(self):
        state = self.env.reset() #my card, dealer's card, usable ace
        state = [int(state[0]-1),int(state[1]-1), int(state[2])]
        return state
    
    def play(self,act): #act = [hit, stand]
        state, reward, done, ept_dict = self.env.step(int(act))
        state = [int(state[0]-1),int(state[1]-1), int(state[2])]
        return state, reward, done
    
    def oneEps(self):
        stateLs = []
        state = self.newGame()
        stateLs.append(state)
        done = False
        while done == False:
            act = self.aMap[state[0],state[1],state[2]]
            state, nxtReward, done = self.play(act)
            stateLs.append(state)
            
        return stateLs, nxtReward
    
    def oneEpsCtrl(self):
        stateLs = []
        actLs = []
        state = self.newGame()
        stateLs.append(state)
        act = random.choice(self.aSpace)
        actLs.append(act)
        done = False
        
        while done == False:
            
            state, nxtReward, done = self.play(act)
            stateLs.append(state)
            try:
                act = self.aMap[state[0],state[1],state[2]]
                actLs.append(int(act))
            except:
                act = '0'
                actLs.append(act)
        return actLs, stateLs, nxtReward
    
    def policy1(self):
        self.aMap [19:,:,:] = 0
    
    def mcPredict(self, eps = 500000):
        e = 0
        while e < eps:
            stateLs, nxtReward = self.oneEps()
            for i in range(len(stateLs)):
                state = stateLs[-(i+1)]
                if state[0] > 20:
                    continue

                self.returnMap[state[0]][state[1]][state[2]].append(nxtReward)
                self.vMap[state[0],state[1],state[2]] = np.mean(self.returnMap[state[0]][state[1]][int(state[2])])
                
            e += 1
                
        return self.vMap
    
    def mcControl(self,eps = 10000):
        e = 0
        while e < eps:
            actLs, stateLs, nxtReward = self.oneEpsCtrl()
            for i in range(len(stateLs)):
                state = stateLs[-(i+1)]
                act = actLs[-(i+1)]
                if state[0] > 20:
                    continue
                self.aCount[state[0], state[1], state[2], act] += 1
                self.qMap[state[0], state[1], state[2], act] += (nxtReward - self.qMap[state[0]][state[1]][state[2]][act])/self.aCount[state[0]][state[1]][state[2]][act]
                maxq = max(self.qMap[state[0]][state[1]][state[2]])
                maxa = []
                for a in self.aSpace:
                    if self.qMap[state[0]][state[1]][state[2]][a] == maxq:
                        maxa.append(a)
                self.aMap[state[0], state[1], state[2]] = random.choice(maxa)
                
            e += 1
            
        return self.aMap
        
#%%                
if __name__ == '__main__':
    bj = blackj()
    bj.policy1()
    vMap = bj.mcPredict()
    print(vMap)
    #%% (a)
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    
    vMap_noA = vMap[11:,:,0]
    vMap_A = vMap[11:,:,1]
    def surface_plot (matrix, **kwargs):
        # acquire the cartesian coordinate matrices from the matrix
        # x is cols, y is rows
        (x, y) = np.meshgrid(np.arange(12,22), np.arange(1,11))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, matrix, **kwargs)
        return (fig, ax, surf)

    (fig, ax, surf) = surface_plot(vMap_noA, cmap=plt.cm.coolwarm)
    
    fig.colorbar(surf)
    
    ax.set_xlabel('My sum')
    ax.set_ylabel('dealer sum')
    ax.set_zlabel('values')
    
    plt.show()  
#%%
    bj1 = blackj()
    policy = bj1.mcControl(eps=3000000)  
    #%%
    fig, ax = plt.subplots()
    im = ax.imshow(bj1.aMap[10:,:,1],extent = [0,10,21,10])
    
    plt.show() 
#%%
    for i in range(21):
        for j in range(10):
            for k in range(2):
                bj1.vMap[i,j,k] = max(bj1.qMap[i,j,k])
    (fig, ax, surf) = surface_plot(bj1.vMap[11:,:,0], cmap=plt.cm.coolwarm)
    
    fig.colorbar(surf)
    
    ax.set_xlabel('dealer sum')
    ax.set_ylabel('player sum')
    ax.set_zlabel('values')
    
    plt.show()  