#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 18:09:28 2020

@author: ziyi
"""
import numpy as np
from copy import copy

class tenArm():
    def __init__(self, randomWalk = False, n_times = 10000, trials = 20):
        self.arm = 10
        self.mean = 0
        self.var = 1  
        self.n_times = n_times
        self.trials = trials
        self.q_list = np.random.normal(self.mean, self.var, self.arm)
        
        if randomWalk == True:
            self.q_list = np.repeat(np.random.normal(self.mean, self.var), self.arm)
            qLs = self.q_list
            q_his= []
            q_his.append(qLs)
            for i in range(n_times):
                dev = np.random.normal(0, 0.01, self.arm)
                #print(dev)
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
        rewardfin = np.array([0]*n_times)
        avgcount = np.array([0]*n_times)
        optLs = []
        for t in range(trials):
            Q_list = [0] * self.arm                 
            rewardLs = []
            avgrewardLs = []
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
               # print(act)
                if act == np.argmax(self.q_list):
                    opcount += 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])/acts_count[act]
                rewardLs.append(reward)
                avgrewardLs.append(sum(rewardLs)/(i + 1))
                opcountLs.append(opcount/(i+1))
                
                
            #print(Q_list,self.q_list)
            #print(max(Q_list),max(self.q_list))   
            #print(np.argmax(np.array(Q_list)),np.argmax(np.array(self.q_list)))  
                
            rewardfin = rewardfin + (avgrewardLs - rewardfin)/(t+1)
            avgcount = avgcount + (opcountLs - avgcount)/(t+1)
            optLs.append(max(avgrewardLs))
        
        return rewardfin, avgcount, np.mean(optLs)
    
    def eGreedySA7(self, epsilon = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.array([0]*n_times)
        avgcount = np.array([0]*n_times)
        optLs = []
        for t in range(trials):
            Q_list = [0] * self.arm                 
            rewardLs = []
            avgrewardLs = []
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
                    opcount += 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])/acts_count[act]
                rewardLs.append(reward)
                avgrewardLs.append(sum(rewardLs)/(i + 1))
                opcountLs.append(opcount/(i+1))
                
                
            #print(Q_list,self.q_list)
            #print(max(Q_list),max(self.q_list))   
            #print(np.argmax(np.array(Q_list)),np.argmax(np.array(self.q_list)))  
                
            rewardfin = rewardfin + (avgrewardLs - rewardfin)/(t+1)
            avgcount = avgcount + (opcountLs - avgcount)/(t+1)
            optLs.append(max(avgrewardLs))
            
        return rewardfin, avgcount, np.mean(optLs)
    
    def eGreedyCS7(self, epsilon = 0.1, alpha = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.array([0]*n_times)
        avgcount = np.array([0]*n_times)
        optLs = []
        for t in range(trials):
            Q_list = [0] * self.arm                 
            rewardLs = []
            avgrewardLs = []
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
                    opcount += 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])*alpha
                rewardLs.append(reward)
                avgrewardLs.append(sum(rewardLs)/(i + 1))
                opcountLs.append(opcount/(i+1))
                
                
            #print(Q_list,self.q_list)
            #print(max(Q_list),max(self.q_list))   
            #print(np.argmax(np.array(Q_list)),np.argmax(np.array(self.q_list)))  
                
            rewardfin = rewardfin + (avgrewardLs - rewardfin)/(t+1)
            avgcount = avgcount + (opcountLs - avgcount)/(t+1)
            optLs.append(max(avgrewardLs))
            
        return rewardfin, avgcount, np.mean(optLs)
    
    def optGreedy(self,q1 = 0, epsilon = 0.1, alpha = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.array([0]*n_times)
        avgcount = np.array([0]*n_times)
        optLs = []
        for t in range(trials):
            Q_list = [q1] * self.arm                 
            rewardLs = []
            avgrewardLs = []
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
               # print(act)
                if act == np.argmax(self.q_list):
                    opcount += 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])*alpha
                rewardLs.append(reward)
                avgrewardLs.append(sum(rewardLs)/(i + 1))
                opcountLs.append(opcount/(i+1))
                
                
            #print(Q_list,self.q_list)
            #print(max(Q_list),max(self.q_list))   
            #print(np.argmax(np.array(Q_list)),np.argmax(np.array(self.q_list)))  
                
            rewardfin = rewardfin + (avgrewardLs - rewardfin)/(t+1)
            avgcount = avgcount + (opcountLs - avgcount)/(t+1)
            optLs.append(max(avgrewardLs))
            
        return rewardfin, avgcount, np.mean(optLs)
    
    
    def UCB(self, c = 2, alpha = 0.1):
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.array([0]*n_times)
        avgcount = np.array([0]*n_times)
        optLs = []
        for t in range(trials):
            Q_list = [0] * self.arm                 
            rewardLs = []
            avgrewardLs = []
            opcount = 0
            opcountLs = []
            acts_count = {0:10e-8,1:10e-8,2:10e-8,3:10e-8,4:10e-8,5:10e-8,6:10e-8,7:10e-8,8:10e-8,9:10e-8}
            for i in range(n_times):
                act = max(acts_count.keys(), key=lambda act: Q_list[act]+c*np.sqrt(np.log(i)/acts_count[act]))
                if act == np.argmax(self.q_list):
                    opcount += 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                Q_list[act] = Q_list[act] + (reward - Q_list[act])/acts_count[act]
                rewardLs.append(reward)
                avgrewardLs.append(sum(rewardLs)/(i + 1))
                opcountLs.append(opcount/(i+1))
                
                
            #print(Q_list,self.q_list)
            #print(max(Q_list),max(self.q_list))   
            #print(np.argmax(np.array(Q_list)),np.argmax(np.array(self.q_list)))  
                
            rewardfin = rewardfin + (avgrewardLs - rewardfin)/(t+1)
            avgcount = avgcount + (opcountLs - avgcount)/(t+1)
            optLs.append(max(avgrewardLs))
        
        return rewardfin, avgcount, np.mean(optLs)
    
    def gradientBandit(self, alpha=0.1):
                
        n_times = self.n_times
        trials = self.trials
        rewardfin = np.array([0]*n_times)
        avgcount = np.array([0]*n_times)
        optLs = []
        for t in range(trials):
            act_p = np.repeat(1/self.arm,self.arm)
            H_list = [0] * self.arm                 
            rewardLs = []
            avgrewardLs = []
            opcount = 0
            opcountLs = []
            acts_count = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
            for i in range(n_times):
                act = int(np.random.choice(np.linspace(0,9,10),p = act_p))
                if act == np.argmax(self.q_list):
                    opcount += 1
                acts_count[act] += 1
                reward = self.pull(act+1)
                rewardLs.append(reward)
                avgrewardLs.append(sum(rewardLs)/(i + 1))
                opcountLs.append(opcount/(i+1))
                
                for a in range(self.arm):
                    if a == act:
                        H_list[a] = H_list[a] + alpha*(reward-np.mean(rewardLs))*(1-act_p[a])
                    else:
                        H_list[a] = H_list[a] - alpha*(reward-np.mean(rewardLs))*act_p[act]
                act_p = np.exp(H_list)/sum(np.exp(H_list))
                
            #print(Q_list,self.q_list)
            #print(max(Q_list),max(self.q_list))   
            #print(np.argmax(np.array(Q_list)),np.argmax(np.array(self.q_list)))  
                
            rewardfin = rewardfin + (avgrewardLs - rewardfin)/(t+1)
            avgcount = avgcount + (opcountLs - avgcount)/(t+1)
            optLs.append(max(avgrewardLs))
        
        return rewardfin, avgcount, np.mean(optLs)
        
    
            
        
        


