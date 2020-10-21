#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:05:15 2020

@author: ziyi
"""


import numpy as np
import random

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
        
    def resetQA(self):
        self.aMap = [[ random.choice(self.aSpace) for i in range(11)] for i in range(11)]
        self.aCount = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        self.qMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(11)] for i in range(11)]
        
        
    def oneEpsCtrl(self,e = 0.01, timeout = 800):
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
        print(len(stateLs),reward)    
        return stateLs, actLs, reward
    
    def mcControl(self,eps = 10000,gamma = 0.99):
        e = 0
        rewardLs = []
        #rCum = 0
        while e < eps:
            stateLs, actLs, reward = self.oneEpsCtrl()
            #rCum += reward
            #rewardLs.append(rCum)
            rewardLs.append((pow(reward*gamma,len(stateLs)-1)))
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
                
            e += 1
            #print(reward,e)
        return rewardLs

def findIndex(stateLs):
    stateDic = {}
    length = len(stateLs)
    for i in range(length):
        state = stateLs[-i-1]
        stateDic[repr(state)] = length - i - 1
    
    return stateDic
    
if __name__ == '__main__':

    tenItr = []

    for i in range(10): 
        gw = gridworld()
        rLs = gw.mcControl()  
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
    ax.set_title('Discounted reward e = 0.01')
    ax.set_xlabel('episodes')
    ax.hlines(pow(0.99,20),0,len(rAvg), color = 'black', linestyles = 'dashed')
    plt.show()
    
            
            
            
            
            
            
            
            