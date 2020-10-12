#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:55:51 2020

@author: ziyi
"""


import numpy as np

class gridworld():
    
    def __init__(self):
        self.mapState = np.zeros([5,5])
        self.mapReward = np.array([[0,10,0,5,0],
                              [0,0,0,0,0],
                              [0,0,0,0,0],
                              [0,0,0,0,0],
                              [0,0,0,0,0]])
        self.actionList = {'u':np.array([-1,0]),'d':np.array([1,0]),
                           'l':np.array([0,-1]),'r':np.array([0,1])}
        self.n_steps = 0
        self.init_policy = {'[0, 0]': ['u', 'd', 'l', 'r'], '[0, 1]': ['u', 'd', 'l', 'r'], '[0, 2]': ['u', 'd', 'l', 'r'], '[0, 3]': ['u', 'd', 'l', 'r'], '[0, 4]': ['u', 'd', 'l', 'r'], '[1, 0]': ['u', 'd', 'l', 'r'], '[1, 1]': ['u', 'd', 'l', 'r'], '[1, 2]': ['u', 'd', 'l', 'r'], '[1, 3]': ['u', 'd', 'l', 'r'], '[1, 4]': ['u', 'd', 'l', 'r'], '[2, 0]': ['u', 'd', 'l', 'r'], '[2, 1]': ['u', 'd', 'l', 'r'], '[2, 2]': ['u', 'd', 'l', 'r'], '[2, 3]': ['u', 'd', 'l', 'r'], '[2, 4]': ['u', 'd', 'l', 'r'], '[3, 0]': ['u', 'd', 'l', 'r'], '[3, 1]': ['u', 'd', 'l', 'r'], '[3, 2]': ['u', 'd', 'l', 'r'], '[3, 3]': ['u', 'd', 'l', 'r'], '[3, 4]': ['u', 'd', 'l', 'r'], '[4, 0]': ['u', 'd', 'l', 'r'], '[4, 1]': ['u', 'd', 'l', 'r'], '[4, 2]': ['u', 'd', 'l', 'r'], '[4, 3]': ['u', 'd', 'l', 'r'], '[4, 4]': ['u', 'd', 'l', 'r']}
        
    def getNextState(self,state,action):
        d = self.actionList[action]
        s = [state[0]+d[0],state[1]+d[1]]
        if s[0] > -1 and s[0] < 5:
            if s[1] > -1 and s[1] < 5:           
                nextstate = s
            else: nextstate = state  
        else:
            nextstate = state   
            
        if state == [0,1]:           
            nextstate = [4,1]            
        elif state == [0,3]:            
            nextstate = [2,3]        
        return nextstate

    def getVvalue(self,s):
        if s[0] > -1 and s[0] < 5:
            if s[1] > -1 and s[1] < 5: 
                V = self.mapState[s[0],s[1]]
            else:V = 0
        else:
            V = 0
        return V
    
    def setVvalue(self,s,val):
        self.mapState[s[0],s[1]] = val
    
    def getRvalue(self,s):
        
        if s[0] > -1 and s[0] < 5:
            if s[1] > -1 and s[1] < 5:           
                reward = 0
            else: reward = -1    
        else:
            reward = -1        
        return reward
        
    
    def getNeighbors(self,s):
        
        nbrState = [[s[0]-1,s[1]],
                    [s[0]+1,s[1]],
                    [s[0],s[1]+1],
                    [s[0],s[1]-1]]    
                
        return nbrState
    
    def policyevl(self, delta = 0.0001, gamma = 0.9):
        self.mapState = np.zeros([5,5])
        n_action = 4
        d = delta + 1
        while d > delta:
            d = 0
            newMap = np.zeros([5,5])
            for i in range(len(self.mapState[0])):
                
                for j in range(len(self.mapState[1])):
                    
                    old_v = self.mapState[i][j]
                    if [i,j] != [0,1] and [i,j] != [0,3]:
                        
                        nbrStates = self.getNeighbors([i,j])
                        new_v = 0
                        for s in nbrStates: 
                            if s[0] > -1 and s[0] < 5:
                                if s[1] > -1 and s[1] < 5: 
                                    new_v += (self.getRvalue(s) + gamma*self.getVvalue(s))/n_action
                                else:
                                    new_v += (self.getRvalue(s) + gamma*self.getVvalue([i,j]))/n_action
                            else: new_v += (self.getRvalue(s) + gamma*self.getVvalue([i,j]))/n_action
                            
                    elif [i,j] == [0,1]:
                        
                        nextState = [4,1]
                        new_v = 10 + gamma*self.getVvalue(nextState)
                        
                    elif [i,j] == [0,3]:
                        
                        nextState = [2,3]
                        new_v = 5 + gamma*self.getVvalue(nextState)
                    
                    newMap[i][j] = new_v
                    d = max(d,abs(new_v-old_v))    
            self.mapState = newMap
        self.mapState = self.mapState.round(2)   
            
        return self.mapState  
    def valIter(self, delta = 0.0001, gamma = 0.9):
        self.mapState = np.zeros([5,5])
        d = delta + 1
        while d > delta:
            d = 0
            newMap = np.zeros([5,5])
            for i in range(len(self.mapState[0])):
                
                for j in range(len(self.mapState[1])):
                    
                    old_v = self.mapState[i][j]
                    if [i,j] != [0,1] and [i,j] != [0,3]:
                        
                        evlDict = {}
                        for a in self.actionList:
                            nextState = self.getNextState([i,j],a)
                            evlDict[a] = self.getRvalue(nextState) + gamma*self.getVvalue(nextState)
                        
                        new_v = max(evlDict.values())
                    elif [i,j] == [0,1]:
                        
                        nextState = [4,1]
                        new_v = 10 + gamma*self.getVvalue(nextState)
                        
                    elif [i,j] == [0,3]:
                        
                        nextState = [2,3]
                        new_v = 5 + gamma*self.getVvalue(nextState)
                    
                    newMap[i][j] = new_v
                    d = max(d,abs(new_v-old_v))    
            self.mapState = newMap
        self.mapState = self.mapState.round(2)
        
        actionMap = {}
        for i in range(len(self.mapState[0])):
                
                for j in range(len(self.mapState[1])):
                    
                    evlDict = {}
                    for a in self.actionList:
                        nextState = self.getNextState([i,j],a)
                        evlDict[a] = self.getRvalue(nextState) + gamma*self.getVvalue(nextState)
                    maxV = max(evlDict.values())
                    actionMap[repr([i,j])] = []
                    for action, value in evlDict.items():

                        if value == maxV:
                            actionMap[repr([i,j])].append(action)
                    
                    
        return self.mapState, actionMap
    
    def policyIter(self,delta = 0.0001, gamma = 0.9):
        self.mapState = np.zeros([5,5])
        actionMap = self.init_policy
        policy_stable = False
        while policy_stable == False:

            d = delta + 1
            while d > delta:
                d = 0
                newMap = np.zeros([5,5])
                for i in range(len(self.mapState[0])):
                    
                    for j in range(len(self.mapState[1])):
                        
                        old_v = self.mapState[i][j]
                        if [i,j] != [0,1] and [i,j] != [0,3]:
                            n_action = len(actionMap[repr([i,j])])
                            new_v = 0
                            for a in actionMap[repr([i,j])]:
                                s = self.getNextState([i,j],a)
    
                                if s[0] > -1 and s[0] < 5:
                                    if s[1] > -1 and s[1] < 5: 
                                        new_v += (self.getRvalue(s) + gamma*self.getVvalue(s))/n_action
                                    else:
                                        new_v += (self.getRvalue(s) + gamma*self.getVvalue([i,j]))/n_action
                                else: new_v += (self.getRvalue(s) + gamma*self.getVvalue([i,j]))/n_action
                                
                        elif [i,j] == [0,1]:
                            
                            nextState = [4,1]
                            new_v = 10 + gamma*self.getVvalue(nextState)
                            
                        elif [i,j] == [0,3]:
                            
                            nextState = [2,3]
                            new_v = 5 + gamma*self.getVvalue(nextState)
                        
                        newMap[i][j] = new_v
                        d = max(d,abs(new_v-old_v))    
                self.mapState = newMap
            self.mapState = self.mapState.round(2)   
            
            newactionMap = {}
            for i in range(len(self.mapState[0])):
                    
                    for j in range(len(self.mapState[1])):
                        
                        evlDict = {}
                        for a in self.actionList:
                            nextState = self.getNextState([i,j],a)
                            evlDict[a] = self.getRvalue(nextState) + gamma*self.getVvalue(nextState)
                        maxV = max(evlDict.values())
                        newactionMap[repr([i,j])] = []
                        for action, value in evlDict.items():
    
                            if value == maxV:
                                newactionMap[repr([i,j])].append(action)
            if newactionMap == actionMap:
                policy_stable = True
            else:
                actionMap = newactionMap
        
        return self.mapState, actionMap
        
        
#%%(a)        

if __name__ == '__main__':
    
    gw = gridworld()
    evlMap = gw.policyevl()
    print(evlMap)  
#%%(b)
    
if __name__ == '__main__':
    
    gw = gridworld()
    evlMap, actMap = gw.valIter()
    print(evlMap) 
    print(actMap)      
#%%(c)
    
if __name__ == '__main__':
    
    gw = gridworld()
    evlMap, actMap = gw.policyIter()
    print(evlMap) 
    print(actMap)                  
                        
                    
                    
