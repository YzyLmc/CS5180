#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:38:39 2020

@author: ziyi
"""


import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D

class gridworld():
    
    def __init__(self, randomGoal = False):
        self.map = np.array([[0,0,0,0,0,-1,0,0,0,0,0],  ##map flipped in matrix form
                            [0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,-1,-1,0,-1,-1],
                            [-1,0,-1,-1,-1,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            ])
        self.rMap = np.zeros([11,11])
        self.agentPos = np.array([0,0])
        self.aSpace = ['u','d','l','r']
        self.actionList = {'u':np.array([0,1]),'d':np.array([0,-1]),
                           'l':np.array([-1,0]),'r':np.array([1,0])}
        self.n_steps = 0

        self.aMap = [[ random.choice(self.aSpace) for i in range(11)] for i in range(11)]
        self.aCount = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        self.qMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        self.vMap = np.zeros([11,11])
        #self.goalPos = np.array([0,0])
        
        
        if randomGoal:
            self.randomGoal()
            print(self.goalPos)
        else:
            self.goalPos = np.array([10,10])
        self.rMap[self.goalPos[0], self.goalPos[1]] = 1
                
    def randomGoal(self):
        notDone = True
        while notDone:
            g = np.array([random.randrange(0,11)] + [random.randrange(0,11)])
            if self.__isValid(g) and repr(g) != repr(self.agentPos):
                self.goalPos = g
                notDone = False
                print('random goalPos:',self.goalPos)
            

    def setAgentPos(self,pos):
        if self.__isvalid(pos):
            self.agentPos = pos  ##pos is an array as [x,y]
        else:
            raise Exception("input pos invalid")
            
                                
    def __isValid(self,pos): ##if a pos = [x,y] is valid, x&y is integer        
        
        if pos[0] > 10 or pos[0] < 0 or pos[1] >10 or pos[1] < 0:
            return False
        elif self.map[pos[0],pos[1]] == -1:
            return False
        else:
            return True
        
    def legalActions(self,pos):
        legalActs = []
        if self.__isValid(pos + np.array([0,1])):
            legalActs.append('u')
        if self.__isValid(pos + np.array([0,-1])):
            legalActs.append('d')
        if self.__isValid(pos + np.array([1,0])):
            legalActs.append('r')
        if self.__isValid(pos + np.array([-1,0])):
            legalActs.append('l')
        
        return legalActs
    
    def nextPos(self,action):
        if action not in self.actionList:
            print("action not in list")
        elif self.rMap[self.agentPos[0],self.agentPos[1]] == 1:
            #print('New episode!')
            self.resetAgent()
        else:
            legalActs = self.legalActions(self.agentPos)
            key = random.random()
            if action == 'u':
                if key < 0.8:
                    finalAction = 'u'
                elif key < 0.9:
                    finalAction = 'l'
                else:
                    finalAction = 'r'
            elif action == 'l':
                if key < 0.8:
                    finalAction = 'l'
                elif key < 0.9:
                    finalAction = 'u'
                else:
                    finalAction = 'd'
            elif action == 'd':
                if key < 0.8:
                    finalAction = 'd'
                elif key < 0.9:
                    finalAction = 'r'
                else:
                    finalAction = 'l'
            elif action == 'r':
                if key < 0.8:
                    finalAction = 'r'
                elif key < 0.9:
                    finalAction = 'd'
                else:
                    finalAction = 'u'
            
            if finalAction in legalActs:
                nextPos = self.agentPos + self.actionList[finalAction]
                self.agentPos = nextPos
            else:
                self.agentPos = self.agentPos
            
        return self.agentPos, self.rMap[self.agentPos[0],self.agentPos[1]]
    
    def getPiAS(self,Pos, action):
        dstr = {}
        if action not in self.actionList:
            print("action not in list")
        elif self.rMap[Pos[0],Pos[1]] == 1:
            #print('New episode!')
            self.resetAgent()
        else:
            legalActs = self.legalActions(Pos)
            if action == 'u':
                    Action = 'u'
                    sideAction1 = 'l'
                    sideAction2 = 'r'
            elif action == 'l':
                    Action = 'l'
                    sideAction1 = 'u'
                    sideAction2 = 'd'
            elif action == 'd':
                    Action = 'd'
                    sideAction1 = 'l'
                    sideAction2 = 'r'
            elif action == 'r':
                    Action = 'r'
                    sideAction1 = 'u'
                    sideAction2 = 'd'
                                
            if Action in legalActs:
                nextPos = Pos + self.actionList[Action]
                try:
                    dstr[repr(nextPos)] += 0.8 
                except: dstr[repr(nextPos)] = 0.8 
            else:
                nextPos = Pos
                try:
                    dstr[repr(nextPos)] += 0.8 
                except: dstr[repr(nextPos)] = 0.8 
                
            if sideAction1 in legalActs:
                nextPos = Pos + self.actionList[sideAction1]
                try:
                    dstr[repr(nextPos)] += 0.1
                except: dstr[repr(nextPos)] = 0.1
            else:
                nextPos = Pos
                try:
                    dstr[repr(nextPos)] += 0.1
                except: dstr[repr(nextPos)] = 0.1
                
            if sideAction2 in legalActs:
                nextPos = Pos + self.actionList[sideAction2]
                try:
                    dstr[repr(nextPos)] += 0.1
                except: dstr[repr(nextPos)] = 0.1
            else:
                nextPos = Pos
                try:
                    dstr[repr(nextPos)] += 0.1
                except: dstr[repr(nextPos)] = 0.1
                
        return dstr
                
    
    def resetAgent(self):
        self.agentPos = np.array([0,0])
        self.n_steps = 0
        
    def resetQA(self):
        self.aMap = [[ random.choice(self.aSpace) for i in range(11)] for i in range(11)]
        self.aCount = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        self.qMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        
        
    def oneEpsCtrl(self,e = 0.1, timeout = 800):
        self.resetAgent()
        stateLs = []
        stateLs.append(self.agentPos)
        actLs = []
        itr = 0
        while repr(self.agentPos) != repr(self.goalPos) and itr < timeout:
            coin = random.random()
            if coin < e:
                act = random.choice(self.aSpace)
            else:
                act = self.aMap[self.agentPos[0]][self.agentPos[1]]
            pos, reward = self.nextPos(act)
            stateLs.append(pos)
            actLs.append(act)
            itr += 1
        if reward == 1:
            print(len(stateLs),reward)    
        return stateLs, actLs, reward
    
    def mcControl(self,eps = 10000,gamma = 0.99):
        e = 0
        rewardLs = []
        aMapLs = []
        epLs = []
        actLsLs = []
        aMapLs.append(self.aMap)
        #rCum = 0
        while e < eps:
            stateLs, actLs, reward = self.oneEpsCtrl()
            #rCum += reward
            #rewardLs.append(rCum)
            rewardLs.append(reward)
            stateDic = findIndex(stateLs)
            for state, index in stateDic.items():
                state = state[7:-2].split(',')
                state = [int(state[0]),int(state[1])]
                rewardi = pow(reward*0.99,len(stateLs)-index-1)
                try:
                    act = actLs[index]
                except:
                    continue
                self.aCount[state[0]][state[1]][act] += 1
                self.qMap[state[0]][state[1]][act] += ( rewardi- self.qMap[state[0]][state[1]][act])/self.aCount[state[0]][state[1]][act]
                maxq = max(self.qMap[state[0]][state[1]].values())
                maxa = []
                #print(self.qMap[state[0]][state[1]],maxq)
                for a in self.aSpace:
                    if self.qMap[state[0]][state[1]][a] == maxq:
                        maxa.append(a)
                self.aMap[state[0]][state[1]] = random.choice(maxa)
                
            epLs.append(stateLs)
            actLsLs.append(actLs)
            aMapLs.append(self.aMap)  
            e += 1
            #print('onpolicy',e)
        return rewardLs,aMapLs, epLs, actLsLs
    
    def onpolicy(self,rewardLs,aMapLs,epLs,actLsLs):
        qMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        cMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        for i in range(len(epLs)):
            stateLs = epLs[i][:-1]
            actLs = actLsLs[i]
            rewardi = rewardLs[i]
        
            for j in range(len(stateLs)):
                state = stateLs[-j-1]
                reward = pow(rewardi*0.99,j+1)
                act = actLs[-j-1]
                cMap[state[0]][state[1]][act] += 1
                qMap[state[0]][state[1]][act] += (reward - qMap[state[0]][state[1]][act])/cMap[state[0]][state[1]][act]

                
                print(reward)
                    
        return qMap
    
    def offpolicy(self,rewardLs,aMapLs,epLs,actLsLs):
        qMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        pi_map = self.aMap
        cMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        for i in range(len(epLs)):
            stateLs = epLs[i][:-1]
            actLs = actLsLs[i]
            aMap = aMapLs[i]
            rewardi = rewardLs[i]
            w = 1
            for j in range(len(stateLs)):
                state = stateLs[-j-1]
                reward = pow(rewardi*0.99,j+1)
                act = actLs[-j-1]
                cMap[state[0]][state[1]][act] += w
                qMap[state[0]][state[1]][act] += w*(reward - qMap[state[0]][state[1]][act])/cMap[state[0]][state[1]][act]
                if act == pi_map[state[0]][state[1]]:
                    if act == aMap[state[0]][state[1]]:
                        w = w*(1/0.925)
                    else:
                        w = w*(1/0.025)
                elif act != pi_map[state[0]][state[1]]:
                    break
                
                print(reward)
                    
        return qMap,pi_map,cMap
    
    def dp(self, policy, delta = 0.001):
        d = 1
        
        while d > delta:
            d = 0
            for i in range(11):
                for j in range(11):
                    if self.__isValid(np.array([i,j])):
                        act = policy[i][j]
                        stateDic = self.getPiAS(np.array([i,j]),act)
                        newV = 0
                        for state, prob in stateDic.items():
                            state = state[7:-2].split(',')
                            state = [int(state[0]),int(state[1])]
                            newV += prob*(self.rMap[state[0],state[1]] + 0.99*self.vMap[state[0],state[1]]) 
                            
                        d = max(d, abs(newV - self.vMap[i,j]))
                        
                        self.vMap[i,j] = newV
                    else:
                        continue
            
        return self.vMap
    
    def valueIter(self, delta = 0.001):
        d = 1
        vMap = np.zeros([11,11])
        aMap = [[ random.choice(self.aSpace) for i in range(11)] for i in range(11)]
        while d > delta:
            print(d)
            d = 0
            for i in range(11):
                for j in range(11):
                    if self.__isValid(np.array([i,j])):
                        vLs = []
                        for act in self.aSpace:
                            stateDic = self.getPiAS(np.array([i,j]),act)
                            newV = 0
                            for state, prob in stateDic.items():
                                state = state[7:-2].split(',')
                                state = [int(state[0]),int(state[1])]
                                newV += prob*(self.rMap[state[0],state[1]] + 0.99*vMap[state[0],state[1]]) 
                            vLs.append(newV)
                        d = max(d, abs(vMap[i,j] - max(vLs)))
                        
                        vMap[i,j] = max(vLs)
                    else:
                        continue
        
        for i in range(11):
            for j in range(11):
                if self.__isValid(np.array([i,j])):
                        vLs = []
                        for act in self.aSpace:
                            stateDic = self.getPiAS(np.array([i,j]),act)
                            newV = 0
                            for state, prob in stateDic.items():
                                state = state[7:-2].split(',')
                                state = [int(state[0]),int(state[1])]
                                newV += prob*(self.rMap[state[0],state[1]] + 0.99*vMap[state[0],state[1]]) 
                            vLs.append(newV)
                        
                        aLs = []
                        maxv = max(vLs)
                        for i in range(len(self.aSpace)):
                            for j in range(len(self.aSpace)):
                                if vLs[i] == maxv:
                                    aLs.append(self.aSpace[i])
                                    aMap[i][j] = random.choice(aLs)
                                    
        return vMap, aMap
        

def findIndex(stateLs):
    stateDic = {}
    length = len(stateLs)
    for i in range(length):
        state = stateLs[-i-1]
        stateDic[repr(state)] = length - i - 1
    
    return stateDic


if __name__ == '__main__':
    gw = gridworld()
    rewardLs,aMapLs, epLs, actLsLs = gw.mcControl()
    qMap_on = gw.onpolicy(rewardLs,aMapLs, epLs, actLsLs)    
    qMap,pi_map,cMap = gw.offpolicy(rewardLs,aMapLs, epLs, actLsLs)

 #%%   offpolicy estimation of Vpi 
    on_map = qMap_on
    
    vMap_off = np.zeros([11,11])
    for i in range(len(qMap)):
        for j in range(len(qMap[0])):
            vMap_off[i,j] = max(qMap[i][j].values())
            
    vMap_on = np.zeros([11,11])
    for i in range(len(qMap)):
        for j in range(len(qMap[0])):
            vMap_on[i,j] = max(on_map[i][j].values())
    
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    

    def surface_plot (matrix, **kwargs):
        # acquire the cartesian coordinate matrices from the matrix
        # x is cols, y is rows
        (x, y) = np.meshgrid(np.arange(0,11), np.arange(0,11))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, matrix, **kwargs)
        return (fig, ax, surf)

    (fig, ax, surf) = surface_plot(vMap_on, cmap=plt.cm.coolwarm)
    
    fig.colorbar(surf)
    
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('values')
    
    (fig1, ax1, surf1) = surface_plot(vMap_off, cmap=plt.cm.coolwarm)
    
    fig1.colorbar(surf1)
    
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')
    ax1.set_zlabel('values')
    
    plt.show()  
    
    #%% Vpi using DP
    vMap = gw.dp(gw.aMap)
    
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    def surface_plot (matrix, **kwargs):
        # acquire the cartesian coordinate matrices from the matrix
        # x is cols, y is rows
        (x, y) = np.meshgrid(np.arange(0,11), np.arange(0,11))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, matrix, **kwargs)
        return (fig, ax, surf)

    (fig2, ax2, surf2) = surface_plot(vMap, cmap=plt.cm.coolwarm)
    
    fig2.colorbar(surf2)
    
    ax2.set_xlabel('Y')
    ax2.set_ylabel('X')
    ax2.set_zlabel('values')
    
    plt.show()  
#%%  V*, pi* value iteration
    v_star, pi_star = gw.valueIter() 

    (fig3, ax3, surf3) = surface_plot(v_star, cmap=plt.cm.coolwarm)
    
    fig3.colorbar(surf3)
    
    ax2.set_xlabel('Y')
    ax2.set_ylabel('X')
    ax2.set_zlabel('values')
    
    plt.show()         