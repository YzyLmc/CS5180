#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:08:20 2020
@author: ziyi
"""


import numpy as np
from scipy.stats import poisson
from copy import copy

class carRental():
    def __init__(self):
        self.state = [21,21]
        self.vMap = np.zeros([21,21])
        self.aMap = np.zeros([21,21])
        self.actionSpace = [i for i in range(-5,6)]
        try:
            self.rTrans1 = np.load('rTrans1.npy',allow_pickle=True).item()
        except:                    
            self.rTrans1 = {}
            
        try:
            self.rTrans2 = np.load('rTrans2.npy',allow_pickle=True).item()
        except:                    
            self.rTrans2 = {}
            
        try:
            self.sTrans = np.load('sTrans.npy',allow_pickle=True).item()
        except:      
            self.sTrans = {}
    
    def rentA(self):
        prob_dict ={}
        x = np.arange(poisson.ppf(0.0001, 3),
              poisson.ppf(0.9999, 3))
        for i in x:
            prob_dict[i] = poisson.pmf(i, 3)
            
        return prob_dict
    def returnA(self):
        prob_dict ={}
        x = np.arange(poisson.ppf(0.0001, 3),
              poisson.ppf(0.9999, 3))
        for i in x:
            prob_dict[i] = poisson.pmf(i, 3)
            
        return prob_dict
    
    def rentB(self):
        prob_dict ={}
        x = np.arange(poisson.ppf(0.0001, 4),poisson.ppf(0.9999, 4))
        for i in x:
            prob_dict[i] = poisson.pmf(i, 4)
            
        return prob_dict
    def returnB(self):
        prob_dict ={}
        x = np.arange(poisson.ppf(0.0001, 2),
              poisson.ppf(0.9999, 2))
        for i in x:
            prob_dict[i] = poisson.pmf(i, 2)
            
        return prob_dict
    
    def reward1(self,n_rent,n_move):
        return 10*n_rent - 2* n_move
    
    def reward2(self,n_rent,n_move):
        return 10*n_rent - 2*(n_move-1)
    
    def reward3(self,n_rent,n_move,n_return,n_cur):
        if n_cur + n_return - n_rent > 10:
            park = 4
        else:
            park = 0
            
        return 10*n_rent - 2* max((abs(n_move) -1),0) - park
    
    
    def getLegalAction(self,s):
        i = s[0]
        j = s[1]
        if i<=5 and j<=5:
            actualAspace = [i for i in range(-j,i+1)]
        elif i<= 5 and j >= 5:
            actualAspace = [i for i in range(-5,i+1)]
        elif i>=5 and j <= 5:
            actualAspace = [i for i in range(-j,6)]
        else:
            actualAspace = self.actionSpace
        
        return actualAspace
    
    def sTransCal(self,cur_state,n_move):
        if not repr(cur_state) in self.sTrans.keys(): 
            self.sTrans[repr(cur_state)] = {}
            old_state = copy(cur_state)
            new_state = [old_state[0]-n_move, old_state[1]+n_move]
            rentA = self.rentA()
            returnA = self.returnA()
            rentB = self.rentB()
            returnB = self.returnB()
            actualRentA = {}
            for num,prob in rentA.items():
                if num <= new_state[0]:
                    actualRentA[num] = prob
                else:
                    try:
                        actualRentA[new_state[0]] += prob
                    except:
                        actualRentA[new_state[0]] = prob   
                        
            actualRentB={}        
            for num,prob in rentB.items():
                if num <= new_state[1]:
                    actualRentB[num] = prob
                else:
                    try:
                        actualRentB[new_state[1]] += prob
                    except:
                        actualRentB[new_state[1]] = prob    
                    
            n_carsA = {}
            for num1,prob1 in actualRentA.items():
                for num2, prob2 in returnA.items():
                    closeA = int(new_state[0] - num1 + num2)
                    closeAprob = prob1*prob2
                    try:
                        n_carsA[min(closeA,20)] += closeAprob
                    except:
                        n_carsA[min(closeA,20)] = closeAprob
                               
                        
            n_carsB = {}
            for num1,prob1 in actualRentB.items():
                for num2, prob2 in returnB.items():
                    closeB = int(new_state[0] - num1 + num2)
                    closeBprob = prob1*prob2
                    try:
                        n_carsB[min(closeB,20)] += closeBprob
                    except:
                        n_carsB[min(closeB,20)] = closeBprob                    

            nextState = {}            
            for num1, prob1 in n_carsA.items():
                for num2, prob2 in n_carsB.items():
                    nextState[repr([num1,num2])] = prob1 * prob2
        
        
            self.sTrans[repr(old_state)][n_move] = nextState
            with open('sTrans.npy','wb') as f:
                np.save(f,self.sTrans)
            
        elif not n_move in self.sTrans[repr(cur_state)].keys():
            old_state = copy(cur_state)
            new_state = [old_state[0]-n_move, old_state[1]+n_move]
            rentA = self.rentA()
            returnA = self.returnA()
            rentB = self.rentB()
            returnB = self.returnB()
            actualRentA = {}
            for num,prob in rentA.items():
                if num <= new_state[0]:
                    actualRentA[num] = prob
                else:
                    try:
                        actualRentA[new_state[0]] += prob
                    except:
                        actualRentA[new_state[0]] = prob   
                        
            actualRentB={}        
            for num,prob in rentB.items():
                if num <= new_state[1]:
                    actualRentB[num] = prob
                else:
                    try:
                        actualRentB[new_state[1]] += prob
                    except:
                        actualRentB[new_state[1]] = prob    
                    
            n_carsA = {}
            for num1,prob1 in actualRentA.items():
                for num2, prob2 in returnA.items():
                    closeA = int(new_state[0] - num1 + num2)
                    closeAprob = prob1*prob2
                    try:
                        n_carsA[min(closeA,20)] += closeAprob
                    except:
                        n_carsA[min(closeA,20)] = closeAprob
                               
                        
            n_carsB = {}
            for num1,prob1 in actualRentB.items():
                for num2, prob2 in returnB.items():
                    closeB = int(new_state[0] - num1 + num2)
                    closeBprob = prob1*prob2
                    try:
                        n_carsB[min(closeB,20)] += closeBprob
                    except:
                        n_carsB[min(closeB,20)] = closeBprob                    

            nextState = {}            
            for num1, prob1 in n_carsA.items():
                for num2, prob2 in n_carsB.items():
                    nextState[repr([num1,num2])] = prob1 * prob2
        
        
            self.sTrans[repr(old_state)][n_move] = nextState
            with open('sTrans.npy','wb') as f:
                np.save(f,self.sTrans)
        else: 
            nextState = self.sTrans[repr(cur_state)][n_move]
            
        
        if not repr(cur_state[0]) in self.rTrans1.keys(): #rewardA
            self.rTrans1[repr(cur_state[0])] = {}
            old_state = copy(cur_state)
            new_state = [old_state[0]-n_move, old_state[1]+n_move]
            rentA = self.rentA()
            returnA = self.returnA()
            actualRentA = {}
            for num,prob in rentA.items():
                if num <= new_state[0]:
                    actualRentA[num] = prob
                else:
                    try:
                        actualRentA[new_state[0]] += prob
                    except:
                        actualRentA[new_state[0]] = prob  
            
            rewardA = {}                       
            for num1,prob1 in actualRentA.items():
                for num2, prob2 in returnA.items():
                    closeAprob = prob1*prob2
                    reward = num1 * 10 - 2* max((abs(n_move) -1),0)
                    try:
                        rewardA[reward] += closeAprob
                    except:
                        rewardA[reward] = closeAprob 
                        
            self.rTrans1[repr(cur_state[0])][n_move] = rewardA
            
            with open('rTrans1.npy','wb') as f:
                np.save(f,self.rTrans1)
            
        elif not n_move in self.rTrans1[repr(cur_state[0])].keys():
            old_state = copy(cur_state)
            new_state = [old_state[0]-n_move, old_state[1]+n_move]
            rentA = self.rentA()
            returnA = self.returnA()
            actualRentA = {}
            for num,prob in rentA.items():
                if num <= new_state[0]:
                    actualRentA[num] = prob
                else:
                    try:
                        actualRentA[new_state[0]] += prob
                    except:
                        actualRentA[new_state[0]] = prob  
                        
            rewardA = {}                        
            for num1,prob1 in actualRentA.items():
                for num2, prob2 in returnA.items():
                    closeAprob = prob1*prob2
                    reward = num1 * 10 - 2* max((abs(n_move) -1),0)
                    try:
                        rewardA[reward] += closeAprob
                    except:
                        rewardA[reward] = closeAprob 
                        
            self.rTrans1[repr(cur_state[0])][n_move] = rewardA
            with open('rTrans1.npy','wb') as f:
                np.save(f,self.rTrans1)
            
        else:
            rewardA = self.rTrans1[repr(cur_state[0])][n_move]
            
            
        if not repr(cur_state[1]) in self.rTrans2.keys(): #rewardB
            self.rTrans2[repr(cur_state[1])] = {}
            old_state = copy(cur_state)
            new_state = [old_state[0]-n_move, old_state[1]+n_move]
            rentB = self.rentB()
            returnB = self.returnB()
            actualRentB = {}
            for num,prob in rentB.items():
                if num <= new_state[1]:
                    actualRentB[num] = prob
                else:
                    try:
                        actualRentB[new_state[1]] += prob
                    except:
                        actualRentB[new_state[1]] = prob  
                        
            rewardB = {}                        
            for num1,prob1 in actualRentB.items():
                for num2, prob2 in returnB.items():
                    closeBprob = prob1*prob2
                    reward = num1 * 10 - 2* max((abs(n_move) -1),0)
                    try:
                        rewardB[reward] += closeBprob
                    except:
                        rewardB[reward] = closeBprob 
                        
            self.rTrans2[repr(cur_state[1])][n_move] = rewardB
            with open('rTrans2.npy','wb') as f:
                np.save(f,self.rTrans2)
            
        elif not n_move in self.rTrans2[repr(cur_state[1])].keys():
            old_state = copy(cur_state)
            new_state = [old_state[0]-n_move, old_state[1]+n_move]
            rentB = self.rentB()
            returnB = self.returnB()
            actualRentB = {}
            for num,prob in rentB.items():
                if num <= new_state[1]:
                    actualRentB[num] = prob
                else:
                    try:
                        actualRentB[new_state[1]] += prob
                    except:
                        actualRentB[new_state[1]] = prob  
                        
            rewardB = {}                        
            for num1,prob1 in actualRentB.items():
                for num2, prob2 in returnB.items():
                    closeBprob = prob1*prob2
                    reward = num1 * 10 - 2* max((abs(n_move) -1),0)
                    try:
                        rewardB[reward] += closeBprob
                    except:
                        rewardB[reward] = closeBprob 
                        
            self.rTrans2[repr(cur_state[1])][n_move] = rewardB
            with open('rTrans2.npy','wb') as f:
                np.save(f,self.rTrans2)
            
        else:
            rewardB = self.rTrans2[repr(cur_state[1])][n_move] 
            
            
        return nextState, rewardA, rewardB
    
    def DPone(self, cur_state, n_move, gamma = 0.9): #n_move is a policy
        
        nextState, rewardA,rewardB = self.sTransCal(cur_state,n_move)
        newstateV = 0
        for state, prob in nextState.items():
            state = state[1:-1].split(',')
            a_state = [int(float(state[0])),int(float(state[1]))]
                
            newstateV += prob*(gamma*self.vMap[a_state[0],a_state[1]])
            
        r1 = 0
        for rewardAi,probRAi in rewardA.items():
            r1 += rewardAi * probRAi
            
        r2 = 0
        for rewardBi, probRBi in rewardB.items():
            r2 += rewardBi * probRBi
        
        newstateV += r1 + r2 
        
        return newstateV
    
    
    def policyEvl(self,delta = 0.001):
        d = delta + 1
        while d > delta:
            d = 0
            newVmap = np.zeros([21,21])
            for i in range(len(self.vMap)):
                for j in range(len(self.vMap)):                       
                    a = self.aMap[i][j]                       
                    old_v = self.vMap[i][j]
                    new_v = self.DPone([i,j],a)
                    d = max(d,abs(old_v-new_v))
                    newVmap[i][j] = new_v
            
            self.vMap = newVmap
            
        self.vMap = self.vMap.round(2)
        
        return self.vMap
        
    
    
    def policyImprove(self):
        newAmap = np.zeros([21,21])
        for i in range(len(self.vMap)):
            for j in range(len(self.vMap)):
                aSpace = self.getLegalAction([i,j])
                vList = np.zeros(len(aSpace))
                for num in range(len(aSpace)):
                    a = aSpace[num]
                    nextState, rewardA,rewardB = self.sTransCal([i,j],a)
                    newstateV = 0
                    for state, prob in nextState.items():
                        state = state[1:-1].split(',')
                        a_state = [int(float(state[0])),int(float(state[1]))]
                            
                        newstateV += prob*(0.9*self.vMap[a_state[0],a_state[1]])                        
                    r1 = 0
                    for rewardAi,probRAi in rewardA.items():
                        r1 += rewardAi * probRAi                        
                    r2 = 0
                    for rewardBi, probRBi in rewardB.items():
                        r2 += rewardBi * probRBi                    
                    newstateV += r1 + r2
                    vList[num] = newstateV
                idx = max(aSpace, key = lambda  num:vList[num])
                a_pi = aSpace[idx]
                newAmap[i][j] = a_pi
                
        self.aMap = newAmap
        
        return self.aMap
                
        
if __name__ == '__main__':
    jack = carRental()
    vMap = jack.policyEvl()
    print(vMap)
    newAmap = jack.policyImprove()
    print(newAmap)
    #%%
    vMap1 = jack.policyEvl()
    print(vMap1)
    #%%
    newAmap2 = jack.policyImprove()
    print(newAmap2)
    #%%
    vMap2 = jack.policyEvl()
    print(vMap2)
    #%%
    newAmap3 = jack.policyImprove()
    print(newAmap3)
    #%%
    vMap3 = jack.policyEvl()
    print(vMap3)
    #%%
    newAmap4 = jack.policyImprove()
    print(newAmap4)
    #%%
    vMap4 = jack.policyEvl()
    print(vMap4)
    #%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
im = ax.imshow(newAmap4)
    
plt.show() 
#%%

def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)

(fig, ax, surf) = surface_plot(vMap4-60, cmap=plt.cm.coolwarm)

fig.colorbar(surf)

ax.set_xlabel('n_Cars(A)')
ax.set_ylabel('n_Cars(B)')
ax.set_zlabel('values')

plt.show()  