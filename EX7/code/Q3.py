#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:26:41 2020

@author: ziyi
"""


import numpy as np
import random
from copy import copy

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
        self.agentPos = np.array([0,0])
        self.aSpace = ['u','d','l','r']
        self.actionList = {'u':np.array([0,1]),'d':np.array([0,-1]),
                           'l':np.array([-1,0]),'r':np.array([1,0])}
        self.n_steps = 0

        self.aMap = [[ random.choice(self.aSpace) for i in range(11)] for i in range(11)]
        self.aCount = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        self.qMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        self.w = np.zeros([121, 4])
        self.w2 = np.zeros([4,3])
        #self.w2 = np.zeros([4,3])
        #self.goalPos = np.array([0,0])
        
        
        if randomGoal:
            self.randomGoal()
            print(self.goalPos)
        else:
            self.goalPos = np.array([10,10])
        self.map[self.goalPos[0], self.goalPos[1]] = 1
                
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
        elif self.map[self.agentPos[0],self.agentPos[1]] == 1:
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
            
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]]
    
    def resetAgent(self):
        self.agentPos = np.array([0,0])
        self.n_steps = 0
        
    def tabular(self,state):
        return state[0] + state[1]
    
    def agg1(self,state):
        if state[1] < 5:
            if state[0] < 5:
                idx = 0
            elif state[0] > 5:
                idx = 1
            elif state[0] == 5:
                idx = 2
        elif state[1] > 5:
            if state[0] < 4:
                idx = 3
            elif state[0] > 4:
                idx = 4
            elif state[0] == 4:
                idx = 5
                
        elif state[1] == 5:
            if state[0] == 1:
                idx = 6
            elif state[0] == 8:
                idx = 7
        
        return idx
        
    def agg2(self,state):
        if state[1] < 3:
            if state[0] < 3:
                idx = 0
            elif state[0] > 2 and state[0] < 5:
                idx = 1
            elif state[0] == 5:
                idx = 2
            elif state[0] > 5 and state[0] < 8:
                idx = 3
            elif state[0] > 7:
                idx = 4
        elif state[1] > 2 and state[1] < 5:
            if state[0] < 3:
                idx = 5
            elif state[0] > 2 and state[0] < 5:
                idx = 6
            elif state[0] == 5:
                idx = 7
            elif state[0] > 5 and state[0] < 8:
                idx = 8
            elif state[0] > 7:
                idx = 9
        elif state[1] > 5 and state[1] < 8:
            if state[0] < 3:
                idx = 10
            elif state[0] > 2 and state[0] < 4:
                idx = 11
            elif state[0] > 4 and state[0] < 7:
                idx = 12
            elif state[0] > 6:
                idx = 13
            elif state[0] == 4:
                idx = 14
                
        elif state[1] > 7:
            if state[0] < 3:
                idx = 15
            elif state[0] > 2 and state[0] < 4:
                idx = 16
            elif state[0] > 4 and state[0] < 7:
                idx = 17
            elif state[0] > 6:
                idx = 18
            elif state[0] == 4:
                idx = 19
                
        elif state[1] == 5:
            if state[0] == 1:
                idx = 20
            if state[0] == 8:
                idx = 21
        
        return idx
        
    def agg3(self,state):
        if state[1] < 5:
            idx = 0
        elif state[1] > 5:
            idx = 1
                
        elif state[1] == 5:
            if state[0] == 1:
                idx = 2
            elif state[0] == 8:
                idx = 3
        
        return idx
        
    def oneEps(self,e = 0.1,alpha = 0.1, timeout = 800): ## 0:up, 1: down, 2:left, 3:right
        self.resetAgent()
        coin = random.random()
        if coin < e:
            act = random.choice([0,1,2,3])
        else:
            stateIdx = self.tabular(self.agentPos)
            qLs = []
            for i in range(4):
                qi = self.w[stateIdx,i] * 1
                qLs.append(qi)
            qmax = max(qLs)
            actLs = []
            for i in range(4):
                qi = self.w[stateIdx,i] * 1
                if qi == qmax:
                    actLs.append(i)
            act = random.choice(actLs)
        itr = 0
        while repr(self.agentPos) != repr(self.goalPos) and itr < timeout:

            stateIdx = self.tabular(self.agentPos)   
            pos, reward = self.nextPos(self.aSpace[int(act)])
            newIdx = self.tabular(self.agentPos)
            coin = random.random()
            if coin < e:
                act1 = random.choice([0,1,2,3])
            else:
                qLs = []
                for i in range(4):
                    qi = self.w[stateIdx,i] * 1
                    qLs.append(qi)
                qmax = max(qLs)
                actLs = []
                for i in range(4):
                    qi = self.w[stateIdx,i] * 1
                    if qi == qmax:
                        actLs.append(i)
                act1 = random.choice(actLs)
            #print(newIdx,act1)    
            self.w[stateIdx, act] += alpha*(reward + self.w[newIdx,act1] - self.w[stateIdx,act])
            act = act1    
            itr += 1   
        return itr 
    
    def oneEpsLF(self,e = 0.1, alpha = 0.05, gamma = 0.95, timeout = 800):
        self.resetAgent()
        coin = random.random()
        if coin < e:
            act = random.choice([0,1,2,3])
        else:
            
            qLs = []
            for i in range(4):
                vi = [self.agentPos[0], self.agentPos[1],1]/np.linalg.norm([self.agentPos[0], self.agentPos[1],1])
                qi = np.dot(self.w2[i],vi)
                qLs.append(qi)
            
            qmax = max(qLs)
            actLs = []
            for i in range(4):
                vi = [self.agentPos[0],self.agentPos[1],1]/np.linalg.norm([self.agentPos[0], self.agentPos[1],1])
                qi = np.dot(self.w2[i],vi)
                #print(qi,i)
                if qi == qmax:
                    actLs.append(i)
            #print(actLs,qLs,qmax)
            act = random.choice(actLs)
        itr = 0
        while repr(self.agentPos) != repr(self.goalPos) and itr < timeout:

            stateVec = [copy(self.agentPos[0]),copy(self.agentPos[1]),1]/np.linalg.norm([copy(self.agentPos[0]), copy(self.agentPos[1]),1])
            newState, reward = self.nextPos(self.aSpace[int(act)])
            coin = random.random()
            if coin < e:
                act1 = random.choice([0,1,2,3])
            else:
                qLs = []
                for i in range(4):
                    vi = [copy(self.agentPos[0]), copy(self.agentPos[1]),1]/np.linalg.norm([self.agentPos[0], self.agentPos[1],1])
                    qi = np.dot(self.w2[i],vi)
                    qLs.append(qi)
                qmax = max(qLs)
                actLs = []
                for i in range(4):
                    vi = [self.agentPos[0], self.agentPos[1],1]/np.linalg.norm([self.agentPos[0], self.agentPos[1],1])
                    qi = np.dot(self.w2[i],vi)
                    #print(qi, qmax, qi ==qmax)
                    if qi == qmax:
                        #print('yes')
                        actLs.append(i)
                #print(qLs,qmax,actLs)
                act1 = random.choice(actLs)
                
            #print(act1)  
            newVec =[copy(self.agentPos[0]),copy(self.agentPos[1]),1]/np.linalg.norm([self.agentPos[0], self.agentPos[1],1])
            q = np.dot(self.w2[act],stateVec)
            q1 = np.dot(self.w2[act1],newVec)
            if repr(self.agentPos) == repr(self.goalPos):
                for i in range(len(self.w2[act])):
                    self.w2[act,i] += alpha* (reward - q) * stateVec[i]
            else:
                for i in range(len(self.w2[act])):
                    self.w2[act,i] += alpha* (reward +gamma * q1 - q)*stateVec[i]
            #self.w[stateIdx, act] += alpha*(reward + self.w[newIdx,act1] - self.w[stateIdx,act])
            act = act1    
            itr += 1  
            #print(newVec,stateVec)
        return itr 
    
    def sarsa(self, eps = 100):
        e = 0
        epsLs = []
        while e < eps:
            n_step = self.oneEps()
            epsLs.append(n_step)
            e += 1
        
        return epsLs
    
if __name__ == '__main__':
    tenItr = []

    for i in range(10): 
        gw = gridworld()
        epsLs = gw.sarsa()  
        avgLs = [0] * len(epsLs)
        for i in range(len(epsLs)):
            stIdx = max(0,i-100)
            avg = sum(epsLs[stIdx:i+1])/(i + 1 -stIdx)
            avgLs[i] = avg

        tenItr.append(avgLs)
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
    ax.set_title('tabular equivalent')
    ax.set_xlabel('episodes')
    ax.set_ylabel('steps per episode')
    #ax.hlines(pow(0.99,20),0,len(rAvg), color = 'black', linestyles = 'dashed')
    plt.show()
        