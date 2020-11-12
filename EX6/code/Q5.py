#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:33:09 2020

@author: ziyi
"""


import numpy as np
import random
from copy import copy

class gridworld():
    
    def __init__(self, randomGoal = False):
        self.map = np.array([[0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,1,1,1,1,1,1,1,1],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0]])
        self.agentPos = np.array([0,3])
        self.actionList = {'u':np.array([1,0]),'d':np.array([-1,0]),
                           'l':np.array([0,-1]),'r':np.array([0,1])}
        self.aSpace = ['u','d','l','r']
        self.goalPos = np.array([5,8])
        self.aMap = [[ [random.choice(['u','d','l','r'])] for i in range(9)] for i in range(6)]
        self.qMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(9)] for i in range(6)]
        self.model = {}
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                self.model[repr(np.array([i,j]))] = {'u':0,'d':0,'l':0,'r':0}

        
    def resetAgent(self):
        self.agentPos = np.array([0,3])
        
    def __isValid(self,pos): ##if a pos = [x,y] is valid, x&y is integer        
        
        if pos[0] > 5 or pos[0] < 0 or pos[1] > 8 or pos[1] < 0:            
            return False
        else:
            if self.map[pos[0],pos[1]] == 0:
                return True
            else: return False
        
    def legalActions(self,pos):
        legalActs = []
        if self.__isValid(pos + np.array([0,1])):
            legalActs.append('r')
        if self.__isValid(pos + np.array([0,-1])):
            legalActs.append('l')
        if self.__isValid(pos + np.array([1,0])):
            legalActs.append('u')
        if self.__isValid(pos + np.array([-1,0])):
            legalActs.append('d')
            
        return legalActs
    
    def nextPos(self,action):
        
        if repr(self.agentPos) == repr(self.goalPos):
            self.resetAgent()
            reward = 0
            return self.agentPos, reward
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
            else:
                nextPos = self.agentPos
            
            reward = 0
            if repr(nextPos) == repr(self.goalPos):
                reward = 1
            self.agentPos = nextPos
            return self.agentPos, reward
        
    def dynaP(self, e=0.2, alpha=0.5, maxStep = 6000, n = 250, gamma = 0.95, k = 0.001):
        aCount = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(9)] for i in range(6)]
        rLs = []
        step = 0
        obs = []
        rCum = 0
        while step < 3000:
            step += 1
            pos = self.agentPos
            coin = random.random()
            if coin < e:
                act = random.choice(self.aSpace)
            else:
                qDict = self.qMap[pos[0]][pos[1]]
                maxq = max(qDict.values())
                aLs = []
                for a in qDict.keys():
                    if qDict[a] == maxq:
                        aLs.append(a)
                act = random.choice(aLs)
            #print(act)
            #print(self.agentPos)
            nextPos, reward = self.nextPos(act)
            aCount[pos[0]][pos[1]][act] = 0
            if repr(nextPos) == repr(self.goalPos):
                rCum += 1
                print(step)
                self.resetAgent()
            rLs.append(rCum)
            qDict2 = self.qMap[nextPos[0]][nextPos[1]]
            maxq2 = max(qDict2.values())
            self.qMap[pos[0]][pos[1]][act] += alpha*(reward + gamma*maxq2 - self.qMap[pos[0]][pos[1]][act])
            self.model[repr(pos)][act] = nextPos
            ts = (pos, act, nextPos)
            obs.append(ts)
            for i in range(n):
                ts = random.choice(obs)
                state = ts[0]
                act = ts[1]
                nextState = ts[2]
                
                if repr(nextState) == repr(self.goalPos):
                    reward = 1
                else: reward = 0
                qDict2 = self.qMap[nextState[0]][nextState[1]]
                maxq2 = max(qDict2.values())
                bonus = k*np.sqrt(aCount[state[0]][state[1]][act])
                self.qMap[state[0]][state[1]][act] += alpha*(reward + bonus + gamma*maxq2 - self.qMap[state[0]][state[1]][act])
                

            for i in range(6):
                for j in range(9):
                    for key in aCount[i][j].keys():
                        aCount[i][j][key] += 1
                    
                    
        #self.map[2,-1] = 0 ## open wall
        #self.map[2,0] = 0
        
        while step < 6000:
            step += 1
            pos = self.agentPos
            coin = random.random()
            if coin < e:
                act = random.choice(self.aSpace)
            else:
                qDict = self.qMap[pos[0]][pos[1]]
                maxq = max(qDict.values())
                aLs = []
                for a in qDict.keys():
                    if qDict[a] == maxq:
                        aLs.append(a)
                act = random.choice(aLs)
            #print(act)
            #print(self.agentPos)
            nextPos, reward = self.nextPos(act)
            aCount[pos[0]][pos[1]][act] = 0
            if repr(nextPos) == repr(self.goalPos):
                rCum += 1
                print(step)
                self.resetAgent()
            rLs.append(rCum)
            qDict2 = self.qMap[nextPos[0]][nextPos[1]]
            maxq2 = max(qDict2.values())
            self.qMap[pos[0]][pos[1]][act] += alpha*(reward + gamma*maxq2 - self.qMap[pos[0]][pos[1]][act])
            self.model[repr(pos)][act] = nextPos
            ts = (pos, act, nextPos)
            obs.append(ts)
            for i in range(n):
                ts = random.choice(obs)
                state = ts[0]
                act = ts[1]
                nextState = ts[2]
                
                if repr(nextState) == repr(self.goalPos):
                    reward = 1
                else: reward = 0
                qDict2 = self.qMap[nextState[0]][nextState[1]]
                maxq2 = max(qDict2.values())
                bonus = k*np.sqrt(aCount[state[0]][state[1]][act])
                self.qMap[state[0]][state[1]][act] += alpha*(reward + bonus + gamma*maxq2 - self.qMap[state[0]][state[1]][act])
                

            for i in range(6):
                for j in range(9):
                    for key in aCount[i][j].keys():
                        aCount[i][j][key] += 1
                        
        return rLs 
    
    def dynaP2(self, e=0.25, alpha=0.5, maxStep = 6000, n = 250, gamma = 0.95, k = 0.001):
        aCount = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(9)] for i in range(6)]
        rLs = []
        step = 0
        obs = []
        rCum = 0
        while step < 3000:
            step += 1
            pos = self.agentPos
            coin = random.random()
            if coin < e:
                act = random.choice(self.aSpace)
            else:
                qDict = self.qMap[pos[0]][pos[1]]
                maxq = max(qDict.values())
                aLs = []
                for a in qDict.keys():
                    if qDict[a] == maxq:
                        aLs.append(a)
                act = random.choice(aLs)
            #print(act)
            #print(self.agentPos)
            nextPos, reward = self.nextPos(act)
            aCount[pos[0]][pos[1]][act] = 0
            if repr(nextPos) == repr(self.goalPos):
                rCum += 1
                print(step)
                self.resetAgent()
            rLs.append(rCum)
            qDict2 = self.qMap[nextPos[0]][nextPos[1]]
            maxq2 = max(qDict2.values())
            self.qMap[pos[0]][pos[1]][act] += alpha*(reward + gamma*maxq2 - self.qMap[pos[0]][pos[1]][act])
            self.model[repr(pos)][act] = nextPos
            ts = (pos, act, nextPos)
            obs.append(ts)
            if len(obs) > 800:
                obs = obs[-800:]
            for i in range(n):
                ts = random.choice(obs)
                state = ts[0]
                act = ts[1]
                nextState = ts[2]
                
                if repr(nextState) == repr(self.goalPos):
                    reward = 1
                else: reward = 0
                qDict2 = self.qMap[nextState[0]][nextState[1]]
                maxq2 = max(qDict2.values())
                bonus = k*np.sqrt(aCount[state[0]][state[1]][act])
                self.qMap[state[0]][state[1]][act] += alpha*(reward + bonus + gamma*maxq2 - self.qMap[state[0]][state[1]][act])
                

            for i in range(6):
                for j in range(9):
                    for key in aCount[i][j].keys():
                        aCount[i][j][key] += 1
                    
                    
        self.map[2,-1] = 0 ## open wall
        #self.map[2,0] = 0
        
        while step < 6000:
            step += 1
            pos = self.agentPos
            coin = random.random()
            if coin < e:
                act = random.choice(self.aSpace)
            else:
                qDict = self.qMap[pos[0]][pos[1]]
                maxq = max(qDict.values())
                aLs = []
                for a in qDict.keys():
                    if qDict[a] == maxq:
                        aLs.append(a)
                act = random.choice(aLs)
            #print(act)
            #print(self.agentPos)
            nextPos, reward = self.nextPos(act)
            aCount[pos[0]][pos[1]][act] = 0
            if repr(nextPos) == repr(self.goalPos):
                rCum += 1
                print(step)
                self.resetAgent()
            rLs.append(rCum)
            qDict2 = self.qMap[nextPos[0]][nextPos[1]]
            maxq2 = max(qDict2.values())
            self.qMap[pos[0]][pos[1]][act] += alpha*(reward + gamma*maxq2 - self.qMap[pos[0]][pos[1]][act])
            self.model[repr(pos)][act] = nextPos
            ts = (pos, act, nextPos)
            obs.append(ts)
            if len(obs) > 800:
                obs = obs[-800:]
            for i in range(n):
                ts = random.choice(obs)
                state = ts[0]
                act = ts[1]
                nextState = ts[2]
                
                if repr(nextState) == repr(self.goalPos):
                    reward = 1
                else: reward = 0
                qDict2 = self.qMap[nextState[0]][nextState[1]]
                maxq2 = max(qDict2.values())
                bonus = k*np.sqrt(aCount[state[0]][state[1]][act])
                self.qMap[state[0]][state[1]][act] += alpha*(reward + bonus + gamma*maxq2 - self.qMap[state[0]][state[1]][act])
                

            for i in range(6):
                for j in range(9):
                    for key in aCount[i][j].keys():
                        aCount[i][j][key] += 1
                        
        return rLs 
if __name__ == '__main__':
    tenItr = []

    for i in range(10): 
        gw = gridworld()
        rLs = gw.dynaP()  
        tenItr.append(rLs)
    #%%
    import matplotlib.pyplot as plt
    rAvg = np.zeros(len(tenItr[0]))
    rStd = np.zeros(len(tenItr[0]))
    for i in range(len(tenItr[0])):
        rAvg[i] = np.mean([tenItr[j][i] for j in range(len(tenItr))])
        rStd[i] = np.std([tenItr[j][i] for j in range(len(tenItr))])
    fig,ax = plt.subplots()
    ax.plot(rAvg)
    x = np.linspace(0,len(rAvg)-1,len(rAvg))
    ax.fill_between(x, rAvg-1.96*rStd/np.sqrt(len(tenItr)), rAvg+1.96*rStd/np.sqrt(len(tenItr)), alpha = 0.5)
    ax.set_title('Dyna-Q stochastic (n=250, k=0.001)')
    ax.set_ylabel('Cumulative reward')
    ax.set_xlabel('steps')

    plt.show()
    #%%
    q = np.zeros([6,9])
    
    for i in range(6):
        for j in range(9):
            q[i,j] = max(gw.qMap[i][j].values())
            
    print(q.round(2))