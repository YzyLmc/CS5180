#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 23:34:10 2020

@author: ziyi
"""

import numpy as np
import random
from copy import copy

class gridworld():
    
    def __init__(self, randomGoal = False):
        self.map = np.zeros([7,10])
        self.agentPos = np.array([3,0])
        self.actionList = {'u':np.array([1,0]),'d':np.array([-1,0]),
                           'l':np.array([0,-1]),'r':np.array([0,1]),
                           'ul':np.array([1,-1]), 'ur':np.array([1,1]),
                           'dl':np.array([-1,-1]), 'dr':np.array([-1,1])}
        self.aSpace = ['u','d','l','r']
        self.goalPos = np.array([3,7])
        self.wind = np.array([[0,0,0,1,1,1,2,2,1,0],
                              [0,0,0,1,1,1,2,2,1,0],
                              [0,0,0,1,1,1,2,2,1,0],
                              [0,0,0,1,1,1,2,2,1,0],
                              [0,0,0,1,1,1,2,2,1,0],
                              [0,0,0,1,1,1,2,2,1,0],
                              [0,0,0,1,1,1,2,2,1,0],
                              ])
        self.aMap = [[ [random.choice(['u','d','l','r'])] for i in range(10)] for i in range(7)]
        self.aCount = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(10)] for i in range(7)]
        self.qMap = [[ {'u':0,'d':0,'l':0,'r':0} for i in range(10)] for i in range(7)]
    def resetAgent(self):
        self.agentPos = np.array([3,0])
        
    def __isValid(self,pos): ##if a pos = [x,y] is valid, x&y is integer        
        
        if pos[0] > 6 or pos[0] < 0 or pos[1] >9 or pos[1] < 0:
            return False
        else:
            return True
        
    def legalActions(self,pos, king = False):
        legalActs = []
        if self.__isValid(pos + np.array([0,1])):
            legalActs.append('r')
        if self.__isValid(pos + np.array([0,-1])):
            legalActs.append('l')
        if self.__isValid(pos + np.array([1,0])):
            legalActs.append('u')
        if self.__isValid(pos + np.array([-1,0])):
            legalActs.append('d')
        
        if king:
            if self.__isValid(pos + np.array([1,1])):
                legalActs.append('ur')
            if self.__isValid(pos + np.array([1,-1])):
                legalActs.append('ul')
            if self.__isValid(pos + np.array([-1,1])):
                legalActs.append('dr')
            if self.__isValid(pos + np.array([-1,-1])):
                legalActs.append('dl')
        
        return legalActs
    
    def nextPos(self,action, king = False):
        
        if repr(self.agentPos) == repr(self.goalPos):
            self.resetAgent()
            reward = 0
            return self.agentPos, reward
        else:
            legalActs = self.legalActions(self.agentPos, king)   
            if action in legalActs:
                nextPos = self.agentPos + self.actionList[action]
            else:
                nextPos = self.agentPos
            finPos = np.array([nextPos[0] + self.wind[nextPos[0],nextPos[1]],nextPos[1]])
            finPos[0] = min(6,finPos[0])
            self.agentPos = finPos
            reward = -1
            if repr(self.agentPos) == repr(self.goalPos):
                reward = -1
            #print(self.agentPos)
            return self.agentPos, reward
    
    def oneEpsCtrl(self,e = 0.2,timeout = 1000):
        self.resetAgent()
        stateLs = []
        stateLs.append(self.agentPos)
        actLs = []
        cumR = 0
        itr = 0
        while repr(self.agentPos) != repr(self.goalPos) and itr < timeout:
            coin = random.random()
            if coin < e:
                act = random.choice(self.aSpace)
            else:
                act = random.choice(self.aMap[self.agentPos[0]][self.agentPos[1]])
            #print(act)
            pos, reward = self.nextPos(act)
            cumR += reward
            stateLs.append(pos)
            actLs.append(act)
            itr += 1
        #print(len(stateLs),cumR) 

        return stateLs, actLs, cumR
    
    def mcControl(self,eps = 1000,gamma = 1):
        e = 0
        eps_num = 0
        epsLs = []
        #rCum = 0
        while e < eps:
            stateLs, actLs, reward = self.oneEpsCtrl()
            #rCum += reward
            #rewardLs.append(rCum)
            epsLs += [eps_num] * len(stateLs)
            eps_num += 1

            stateDic = findIndex(stateLs)
            for state, index in stateDic.items():
                state = state[7:-2].split(',')
                state = [int(state[0]),int(state[1])]
                rewardi = -(len(stateLs)-index-1)
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
                self.aMap[state[0]][state[1]] = [random.choice(maxa)]
                 
            e += 1
        return epsLs
    
    def sarsa(self,e = 0.1, alpha = 0.5, maxStep = 10000):
        self.resetAgent()
        step = 0
        eps_num = 0
        epsLs = []
        #rCum = 0
        while step < maxStep:
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
            nextPos, reward = self.nextPos(act)
            coin2 = random.random()
            if coin2 < e:
                act2 = random.choice(self.aSpace)
            else:
                qDict2 = self.qMap[nextPos[0]][nextPos[1]]
                maxq2 = max(qDict2.values())
                aLs2 = []
                for a in qDict2.keys():
                    if qDict2[a] == maxq2:
                        aLs2.append(a)
                act2 = random.choice(aLs2)
            #if repr(nextPos) == repr(self.goalPos):
            #print(pos, act)    
            self.qMap[pos[0]][pos[1]][act] += alpha*(reward + self.qMap[nextPos[0]][nextPos[1]][act2] - self.qMap[pos[0]][pos[1]][act])
            if repr(nextPos) == repr(self.goalPos):
                self.resetAgent()
                eps_num += 1
                
            epsLs.append(eps_num)
            step+=1
            
        return epsLs
    
    def sarsaEx(self,e = 0.1, alpha = 0.5, maxStep = 10000):
        self.resetAgent()
        step = 0
        eps_num = 0
        epsLs = []
        #rCum = 0
        while step < maxStep:
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
            nextPos, reward = self.nextPos(act)
            #coin2 = random.random()
            
                #act2 = random.choice(self.aSpace)
            
            qDict2 = self.qMap[nextPos[0]][nextPos[1]]
            maxq2 = max(qDict2.values())
            aLs2 = []
            for a in qDict2.keys():
                if qDict2[a] == maxq2:
                        aLs2.append(a)
                #act2 = random.choice(aLs2)
            #if repr(nextPos) == repr(self.goalPos):
            #print(pos, act)
            q2 = 0
            for a in self.aSpace:
                q2 += e*self.qMap[nextPos[0]][nextPos[1]][a]/4
            for a in aLs2:               
                q2 += (1-e)*self.qMap[nextPos[0]][nextPos[1]][a]/len(aLs2)
            self.qMap[pos[0]][pos[1]][act] += alpha*(reward + q2 - self.qMap[pos[0]][pos[1]][act])
            if repr(nextPos) == repr(self.goalPos):
                self.resetAgent()
                eps_num += 1
                
            epsLs.append(eps_num)
            step+=1
            
        return epsLs
    
    def qLearning(self,e = 0.1, alpha = 0.5, maxStep = 10000):
        self.resetAgent()
        step = 0
        eps_num = 0
        epsLs = []
        #rCum = 0
        while step < maxStep:
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
            nextPos, reward = self.nextPos(act)
            qDict2 = self.qMap[nextPos[0]][nextPos[1]]
            maxq2 = max(qDict2.values())
            #if repr(nextPos) == repr(self.goalPos):
            #print(pos, act)    
            self.qMap[pos[0]][pos[1]][act] += alpha*(reward + maxq2 - self.qMap[pos[0]][pos[1]][act])
            if repr(nextPos) == repr(self.goalPos):
                self.resetAgent()
                eps_num += 1
                
            epsLs.append(eps_num)
            step+=1
            
        return epsLs
    
    def sarsaN(self,n = 4, e = 0.1, alpha = 0.5, maxStep = 10000):
        eps = 0
        epsLs = []
        totalStep = 0
        while totalStep < maxStep:
            self.resetAgent()
            T = maxStep
            step = 0
            stateLs = [self.agentPos]
            rLs = []
            actLs = []
            while step < T:
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
                nextPos, reward = self.nextPos(act)
                rLs.append(reward)
                stateLs.append(nextPos)
                if repr(nextPos) == repr(self.goalPos):
                    T = step + 1
                    eps += 1
                else: 
                    qDict2 = self.qMap[nextPos[0]][nextPos[1]]
                    maxq2 = max(qDict2.values())
                    aLs2 = []
                    for a in qDict2.keys():
                        if qDict2[a] == maxq2:
                                aLs2.append(a)
                    act2 = random.choice(aLs2)
                    
                actLs.append(act)
                
                tao = step -n + 1
                if tao > -1 :
                    state = stateLs[tao]
    
                    a = actLs[tao]
                    g = 0
                    h = min(n, T - tao)
                    staten = stateLs[tao + h]
                    for i in range(h-1):
                        g += rLs[tao+i]
                    if tao + n < T:
                        g += self.qMap[staten[0]][staten[1]][act2]  
                        
                    self.qMap[state[0]][state[1]][a] += alpha*(g-self.qMap[state[0]][state[1]][a])
                
                step += 1
                epsLs.append(eps)
            totalStep += step
        return epsLs
    
    def train(self,qMap, N, e = 0.1):
        epsLs = []
        aLsLs = []
        
        for itr in range(N):
            self.resetAgent()
            eps = []
            eps.append(self.agentPos)
            while repr(self.agentPos) != repr(self.goalPos):
                pos = self.agentPos
                coin = random.random()
                if coin < e:
                    act = random.choice(self.aSpace)
                else:
                    qDict = qMap[pos[0]][pos[1]]
                    maxq = max(qDict.values())
                    aLs = []
                    for a in qDict.keys():
                        if qDict[a] == maxq:
                            aLs.append(a)
                    act = random.choice(aLs)
                nextPos,reward = self.nextPos(act)
                
                eps.append(nextPos)
                aLs.append(act)
            
            epsLs.append(eps)
            aLsLs.append(aLs)
            
        return epsLs, aLsLs              
    
    def tdPredict(self,epsLs, alpha = 0.5):
        vMap = np.zeros([7,10])
        
        for i in range(len(epsLs)):
            eps = epsLs[i]
            for n in range(len(eps)-1):
                state = eps[n]
                nxtState = eps[n+1]
                vMap[state[0],state[1]] += alpha*(-1 + vMap[nxtState[0],nxtState[1]]-vMap[state[0],state[1]])
                
        return vMap
    
    def mcPredict(self, epsLs):
        vMap = np.zeros([7,10])
        vCount = np.zeros([7,10])
        for i in range(len(epsLs)):
            eps = epsLs[i]
            stateDic = findIndex(eps)
            for state, index in stateDic.items():
                state = state = state[7:-2].split(',')
                state = [int(state[0]),int(state[1])]
                vCount[state[0]][state[1]] += 1
                vMap[state[0]][state[1]] += (-(len(eps)-index-1) - vMap[state[0]][state[1]])/vCount[state[0]][state[1]]
                
        return vMap        

    def ntdPredict(self,epsLs, nStep = 4, alpha = 0.5):
        vMap = np.zeros([7,10])
        
        for i in range(len(epsLs)):
            eps = epsLs[i]
            for n in range(len(eps) + nStep-1):
                tao = n - nStep + 1
                if tao >-1:
                    state = eps[tao]
                    h = min(len(eps)-1, nStep + tao)    
                    nxtState = eps[h]
                    
                    g = -(h-tao)
                    if tao + nStep < len(eps):
                        g += vMap[nxtState[0], nxtState[1]]
                    vMap[state[0],state[1]] += alpha*(g - vMap[state[0],state[1]])
        
        return vMap
    
    def tdPredicte(self,epsLs, alpha = 0.5, num = 100):
        vMap = np.zeros([7,10])
        d = 0
        while d < num :
            for i in range(len(epsLs)):
                eps = epsLs[i]
                for n in range(len(eps)-1):
                    state = eps[n]
                    nxtState = eps[n+1]
                    #g = alpha*(-1 + vMap[nxtState[0],nxtState[1]]-vMap[state[0],state[1]])
                    #print('abc', alpha*(-1 + vMap[nxtState[0],nxtState[1]]-vMap[state[0],state[1]]))
                    vMap[state[0],state[1]] += alpha*(-1 + vMap[nxtState[0],nxtState[1]]-vMap[state[0],state[1]])

            d += 1
            #print(vMap)
        return vMap
    
    def mcPredicte(self, epsLs, delta = 0.01, num = 100):
        vMap = np.zeros([7,10])
        vCount = np.zeros([7,10])
        d = 0
        while d < num:
            for i in range(len(epsLs)):
                eps = epsLs[i]
                stateDic = findIndex(eps)
                for state, index in stateDic.items():
                    state = state = state[7:-2].split(',')
                    state = [int(state[0]),int(state[1])]
                    vCount[state[0]][state[1]] += 1
                    vMap[state[0]][state[1]] += (-(len(eps)-index-1) - vMap[state[0]][state[1]])/vCount[state[0]][state[1]]
            d += 1
        return vMap        

    def ntdPredicte(self,epsLs, nStep = 4, alpha = 0.5, num = 100):
        vMap = np.zeros([7,10])
        d = 0
        while d < num:
            for i in range(len(epsLs)):
                eps = epsLs[i]
                for n in range(len(eps) + nStep-1):
                    tao = n - nStep + 1
                    if tao >-1:
                        state = eps[tao]
                        h = min(len(eps)-1, nStep + tao)    
                        nxtState = eps[h]
                        
                        g = -(h-tao)
                        if tao + nStep < len(eps):
                            g += vMap[nxtState[0], nxtState[1]]
                        vMap[state[0],state[1]] += alpha*(g - vMap[state[0],state[1]])
            d += 1
            print(d)
        return vMap
    
    def tdEvl(self,epsLs, vMap):
        targetLs = []
        for i in range(len(epsLs)):
            eps = epsLs[i]
            for n in range(len(eps)-1):
                state = eps[n]
                if repr(state) == repr(np.array([3,0])):
                    nxtState = eps[n+1]
                    target = -1 + vMap[nxtState[0],nxtState[1]]
                    targetLs.append(target)
                    
        return targetLs
    
    def mcEvl(self, epsLs):
        targetLs = []
        for i in range(len(epsLs)):
            eps = epsLs[i]
            targetLs.append(-len(eps)-1)
                    
        return targetLs
    
    def ntdEvl(self,epsLs, vMap, nStep = 4):
        targetLs = []
        for i in range(len(epsLs)):
            eps = epsLs[i]
            for n in range(len(eps) + nStep-1):
                tao = n - nStep + 1
                if tao == 0:
                    h = min(len(eps)-1, nStep + tao)    
                    nxtState = eps[h]
                    
                    g = -(h-tao)
                    if tao + nStep < len(eps):
                        g += vMap[nxtState[0], nxtState[1]]
                    targetLs.append(g)
        
        return targetLs
    
    def dp(self, qMap, delta = 0.01):
        vMap = np.zeros([7,10])
        d = 1
        while d > delta:
            d = 0
            newMap = np.zeros([7,10])
            for i in range(7):
                for j in range(10):
                    self.agentPos = np.array([i,j])
                    qDict = qMap[i][j]
                    maxq = max(qDict.values())
                    aLs = []
                    for a in qDict.keys():
                        if qDict[a] == maxq:
                            aLs.append(a)
                            
                    target = 0
                    
                    for act in self.aSpace:
                        self.agentPos = np.array([i,j])
                        nextPos, reward = self.nextPos(act)
                        target += 0.1*(reward + vMap[nextPos[0],nextPos[1]])/4
                        
                    for act in aLs:
                        self.agentPos = np.array([i,j])
                        nextPos, reward = self.nextPos(act)
                        target += 0.9*(reward + vMap[nextPos[0],nextPos[1]])/len(aLs)
            

                    if i==3 and j ==7:
                        target = 0
                    newMap[i,j] = target
                    d = max(d,abs(target-vMap[i,j]))
                    print(d)
                    
            vMap = newMap
                    
        return vMap
                    
   
def findIndex(stateLs):
    stateDic = {}
    length = len(stateLs)
    for i in range(length):
        state = stateLs[-i-1]
        stateDic[repr(state)] = length - i - 1
    
    return stateDic

if __name__ == '__main__':
    gw = gridworld()
    rLs = gw.qLearning(maxStep = 30000)
    qMap = gw.qMap
    #vMap = gw.dp(qMap)
    #trueVal =  vMap[3,0]
    #%%
    #epsLs,aLsLs = gw.train(qMap,100)
    #np.save('epsLs100.npy',epsLs)
    epsLs_train = np.load('epsLs10.npy', allow_pickle = True)
    vMap = gw.tdPredicte(epsLs_train)
    epsLs_evl = np.load('epsLs100.npy', allow_pickle = True)
    tgtLs = gw.mcEvl(epsLs_evl)
    #tgtTd = gw.tdEvl(epsLs,vMap)
    #tgtNtd = gw.ntdEvl(epsLs,vMap)
    
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    plt.hist(tgtLs)
    ax.set_xlabel('G')
    ax.set_title('Monte-Carlo: N=1 (converged)')
    ax.vlines(trueVal,0,110, color = 'black', linestyles = 'dashed')
    plt.show()
