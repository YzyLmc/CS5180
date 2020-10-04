#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:09:28 2020

@author: ziyi
"""
import numpy as np
from copy import copy

class tenArm():
    def __init__(self, randomWalk = False, n_times = 1000, trials = 200, shift = 0):
        self.arm = 10
        self.mean = 0
        self.var = 1  
        self.n_times = n_times
        self.trials = trials
        self.q_list = np.random.normal(self.mean, self.var, self.arm)
        self.q_list += shift
        if randomWalk == True:
            qLs = np.repeat(np.random.normal(self.mean, self.var), self.arm)
            q_his= []
            q_his.append(qLs)
            for i in range(n_times):
                dev = np.random.normal(0, 0.01, self.arm)
                qLs += dev
                q_his.append(copy(qLs))
            self.q_his = q_his
        
        
    def pull(self, armNum):
              
        q_pi = self.q_list[armNum - 1]
        return np.random.normal(q_pi,1)
    
    
    def manyPull(self, armNum, n_times = 10000):
        
        rewardLs = []
        for i in range(n_times):
            rewardLs.append(self.pull(armNum))
            
        return rewardLs
    
    def eGreedySA(self, epsilon = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.zeros([1,n_times])
        optMat = np.zeros([n_times,trials])
        optfin = np.zeros([1,n_times])
        optLs = []
        rewardMat = np.zeros([n_times,trials])
        stdMat = np.zeros([1,n_times])
        for t in range(trials):
            Q_list = [0] * self.arm                 
            opcount = 0
            opcountLs = []
            acts_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
            for i in range(n_times):
                coin = np.random.random()
                if coin < epsilon:
                    act = np.random.choice(list(range(10)))
                else:
                    acts = [idx for idx, j in enumerate(Q_list) if j == max(Q_list)]
                    act = np.random.choice(acts)   
                if act == np.argmax(self.q_list):
                    optMat[i,t] = 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])/acts_count[act]
                rewardMat[i,t] = reward
                opcountLs.append(opcount/(i+1))
                
            optLs.append(max(self.q_list))
        for i in range(len(stdMat[0,:])):
            stdMat[0,i] = np.std(rewardMat[i,:])
            rewardfin[0,i] = np.mean(rewardMat[i,:])
            optfin[0,i] =   np.mean(optMat[i,:])
        return rewardfin[0,:], optfin[0,:], np.mean(optLs), stdMat[0,:]
    
    def eGreedySA7(self, epsilon = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.zeros([1,n_times])
        optMat = np.zeros([n_times,trials])
        optfin = np.zeros([1,n_times])
        optLs = np.zeros([1,n_times])
        rewardMat = np.zeros([n_times,trials])
        stdMat = np.zeros([1,n_times])
        for t in range(trials):
            Q_list = [0] * self.arm                 
            opcount = 0
            opcountLs = []
            acts_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
            for i in range(n_times):
                self.q_list = self.q_his[i]
                coin = np.random.random()
                if coin < epsilon:
                    act = np.random.choice(list(range(10)))
                else:
                    acts = [idx for idx, j in enumerate(Q_list) if j == max(Q_list)]
                    act = np.random.choice(acts)   
               # print(act)
                if act == np.argmax(self.q_list):
                    optMat[i,t] = 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])/acts_count[act]
                rewardMat[i,t] = reward
                opcountLs.append(opcount/(i+1))
                
                                            
        for i in range(len(stdMat[0,:])):
            stdMat[0,i] = np.std(rewardMat[i,:])
            rewardfin[0,i] = np.mean(rewardMat[i,:])
            optLs[0,i] = max(self.q_his[i])
            optfin[0,i] =   np.mean(optMat[i,:])
        return rewardfin[0,:], optfin[0,:], optLs[0,:], stdMat[0,:]
    
    def eGreedyCS7(self, epsilon = 0.1, alpha = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.zeros([1,n_times])
        optMat = np.zeros([n_times,trials])
        optfin = np.zeros([1,n_times])
        optLs = np.zeros([1,n_times])
        rewardMat = np.zeros([n_times,trials])
        stdMat = np.zeros([1,n_times])
        for t in range(trials):
            Q_list = [0] * self.arm                 
            opcount = 0
            opcountLs = []
            acts_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
            for i in range(n_times):
                self.q_list = self.q_his[i]
                coin = np.random.random()
                if coin < epsilon:
                    act = np.random.choice(list(range(10)))
                else:
                    acts = [idx for idx, j in enumerate(Q_list) if j == max(Q_list)]
                    act = np.random.choice(acts)   
               # print(act)
                if act == np.argmax(self.q_list):
                    optMat[i,t] = 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                rewardMat[i,t] = reward
                Q_list[act] = Q_list[act] + (reward - Q_list[act])*alpha
                opcountLs.append(opcount/(i+1))
                

        for i in range(len(stdMat[0,:])):
            stdMat[0,i] = np.std(rewardMat[i,:])
            rewardfin[0,i] = np.mean(rewardMat[i,:])
            optLs[0,i] = max(self.q_his[i])
            optfin[0,i] =   np.mean(optMat[i,:])
        return rewardfin[0,:], optfin[0,:], optLs[0,:], stdMat[0,:]
    
    def optGreedy(self,q1 = 0, epsilon = 0.1, alpha = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.zeros([1,n_times])
        optMat = np.zeros([n_times,trials])
        optfin = np.zeros([1,n_times])
        optLs = []
        optMat = np.zeros([n_times,trials])
        rewardMat = np.zeros([n_times,trials])
        stdMat = np.zeros([1,n_times])
        for t in range(trials):
            Q_list = [q1] * self.arm                 
            acts_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
            for i in range(n_times):
                coin = np.random.random()
                if coin < epsilon:
                    act = np.random.choice(list(range(10)))
                else:
                    act = max(acts_count.keys(), key=lambda act: Q_list[act])   
                if act == np.argmax(self.q_list):
                    optMat[i,t] = 1
                    
                #if i < 30:
                #    print(np.around(Q_list,3))
                    
                acts_count[act] += 1
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])*alpha
                rewardMat[i,t] = reward
                
            optLs.append(max(self.q_list))
            
        for i in range(len(stdMat[0,:])):
            stdMat[0,i] = np.std(rewardMat[i,:])
            rewardfin[0,i] = np.mean(rewardMat[i,:])
            optfin[0,i] =   np.mean(optMat[i,:])
            
        return rewardfin[0,:], optfin[0,:], np.mean(optLs), stdMat[0,:]
    
    
    def UCB(self, c = 2, alpha = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.zeros([1,n_times])
        optMat = np.zeros([n_times,trials])
        optfin = np.zeros([1,n_times])
        optLs = []
        rewardMat = np.zeros([n_times,trials])
        stdMat = np.zeros([1,n_times])
        for t in range(trials):
            Q_list = [0] * self.arm                 
            opcount = 0
            opcountLs = []
            acts_count = {0:10e-10,1:10e-10,2:10e-10,3:10e-10,4:10e-10,5:10e-10,6:10e-10,7:10e-10,8:10e-10,9:10e-10}
            for i in range(n_times):
                act = max(acts_count.keys(), key=lambda act: Q_list[act]+c*np.sqrt(np.log(i)/acts_count[act]))
                if act == np.argmax(self.q_list):
                     optMat[i,t] = 1
                acts_count[act] += 1
                                
                #if i < 30:
                #    print(np.around(Q_list,3))
                    
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])/acts_count[act]
                rewardMat[i,t] = reward
                opcountLs.append(opcount/(i+1))
                

            optLs.append(max(self.q_list))
        
        for i in range(len(stdMat[0,:])):
            stdMat[0,i] = np.std(rewardMat[i,:])
            rewardfin[0,i] = np.mean(rewardMat[i,:])
            optfin[0,i] =   np.mean(optMat[i,:])
        return rewardfin[0,:], optfin[0,:], np.mean(optLs), stdMat[0,:]
    
    def gradientBandit(self, alpha=0.1, baseline = True):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.zeros([1,n_times])
        optMat = np.zeros([n_times,trials])
        optfin = np.zeros([1,n_times])
        optLs = []
        rewardMat = np.zeros([n_times,trials])
        stdMat = np.zeros([1,n_times])
        for t in range(trials):
            act_p = np.repeat(1/self.arm,self.arm)
            H_list = [0] * self.arm                 
            rewardLs = []
            acts_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
            for i in range(n_times):
                act = int(np.random.choice(np.linspace(0,9,10),p = act_p))
                if act == np.argmax(self.q_list):
                    optMat[i,t] = 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                rewardLs.append(reward)
                rewardMat[i,t] = reward
                
                if baseline == True:
                    for a in range(self.arm):
                        if a == act:
                            H_list[a] = H_list[a] + alpha*(reward-np.mean(rewardLs))*(1-act_p[a])
                        else:
                            H_list[a] = H_list[a] - alpha*(reward-np.mean(rewardLs))*act_p[a]
                else:
                    for a in range(self.arm):
                        if a == act:
                            H_list[a] = H_list[a] + alpha*reward*(1-act_p[a])
                        else:
                            H_list[a] = H_list[a] - alpha*reward*act_p[a]
                act_p = np.exp(H_list)/sum(np.exp(H_list))
                
            #print(Q_list,self.q_list)
            #print(max(Q_list),max(self.q_list))   
            #print(np.argmax(np.array(Q_list)),np.argmax(np.array(self.q_list)))  
                
            optLs.append(max(self.q_list))
            
        for i in range(len(stdMat[0,:])):
            stdMat[0,i] = np.std(rewardMat[i,:])
            rewardfin[0,i] = np.mean(rewardMat[i,:])
            optfin[0,i] =   np.mean(optMat[i,:])
        return rewardfin[0,:], optfin[0,:], np.mean(optLs), stdMat[0,:]
        
    
            
        
        


