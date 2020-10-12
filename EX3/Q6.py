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
    
    def rentA(self):
        prob_dict ={}
        x = np.arange(poisson.ppf(0.001, 3),
              poisson.ppf(0.999, 3))
        for i in x:
            prob_dict[i] = poisson.pmf(i, 3)
            
        return prob_dict
    def returnA(self):
        prob_dict ={}
        x = np.arange(poisson.ppf(0.001, 3),
              poisson.ppf(0.999, 3))
        for i in x:
            prob_dict[i] = poisson.pmf(i, 3)
            
        return prob_dict
    
    def rentB(self):
        prob_dict ={}
        x = np.arange(poisson.ppf(0.001, 4),poisson.ppf(0.999, 4))
        for i in x:
            prob_dict[i] = poisson.pmf(i, 4)
            
        return prob_dict
    def returnB(self):
        prob_dict ={}
        x = np.arange(poisson.ppf(0.001, 2),
              poisson.ppf(0.999, 2))
        for i in x:
            prob_dict[i] = poisson.pmf(i, 2)
            
        return prob_dict
    
    def DPone(self, cur_state, n_move, gamma = 0.9): #n_move is a policy
                
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
                    actualRentA[new_state[1]] += prob
                except:
                    actualRentA[new_state[1]] = prob   
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
        rewardA = {}
        for num1,prob1 in actualRentA.items():
            for num2, prob2 in returnA.items():
                closeA = new_state[0] - num1 + num2
                closeAprob = num1*num2
                if not closeA in rewardA.keys():
                    rewardA[closeA] = {}
                reward = num1 * 10 - 2* abs(n_move) 
                try:
                    n_carsA[min(closeA,20)] += closeAprob
                    rewardA[closeA][reward] += closeAprob
                except:
                    n_carsA[min(closeA,20)] = closeAprob
                    rewardA[closeA][reward] = closeAprob
                    
        n_carsB = {}
        rewardB = {}
        for num1,prob1 in actualRentA.items():
            for num2, prob2 in returnB.items():
                closeB = new_state[0] - num1 + num2
                closeBprob = num1*num2
                if not closeB in rewardB.keys():
                    rewardB[closeA] = {}
                reward = num1 * 10 - 2* abs(n_move) 
                try:
                    n_carsB[min(closeA,20)] += closeBprob
                    rewardB[min(closeA,20)][reward] += closeBprob
                except:
                    n_carsB[min(closeA,20)] = closeBprob
                    rewardB[min(closeA,20)][reward] = closeBprob
        
        nextState = {}            
        for num1, prob1 in n_carsA.items():
            for num2, prob2 in n_carsB.items():
                nextState[repr([num1,num2])] = prob1 * prob2
                
        newstateV = 0
        for state, prob in nextState.items():
            state = state[1:-1].split(',')
            a_state = [int(float(state[0])),int(float(state[1]))]
            print(rewardA[0])
            r1 = 0
            for rewardA,probRA in rewardA[0].items():
                r1 += rewardA * probRA
                
            r2 = 0
            for rewardB, probRB in rewardB[a_state[1]].items():
                r2 += rewardB * probRB
                
            newstateV += r1 + r2 + gamma*self.vMap[a_state[0],a_state[1]]
          
        return newstateV
                    
        
    def policyEvl(self,delta = 0.001):
        d = delta + 1
        while d > delta:
            d = 0
            newVmap = np.zeros([21,21])
            for i in range(len(self.vMap)):
                for j in range(len(self.vMap)):
                    '''
                    if i<5 and j<5:
                        actualAspace = [i for i in range(-j,i+1)]
                    elif i< 5 and j > 5:
                        actualAspace = [i for i in range(-5,i+1)]
                    elif i>5 and j < 5:
                        actualAspace = [i for i in range(-j,6)]
                    else:
                        actualAspace = self.actionSpace
                    '''                        
                    a = self.aMap[i][j]                       
                    old_v = self.vMap[i][j]
                    new_v = self.DPone([i,j],a)
                    d = max(d,abs(old_v-new_v))
                    newVmap[i][j] = new_v
            
            self.vMap = newVmap
            
        self.vMap = self.vMap.round(2)
        
        return self.vMap
        
if __name__ == '__main__':
    jack = carRental()
    vMap = jack.policyEvl()
    print(vMap)
                        

                