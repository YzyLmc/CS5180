#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 16:57:35 2020

@author: ziyi
"""


from gridworld import gridworld
import matplotlib.pyplot as plt
import numpy as np

def evaluate(agentName, trail = 10, steps = 10000, randomGoal = False):
    gw = gridworld(randomGoal = randomGoal)
    reward_avg_plot = []
    fig, ax = plt.subplots()
    if agentName == 'random':
        for t in range(trail):
            
            i = 0
            reward = 0
            reward_plot= []
            while i < steps:
                p, r , s = gw.randomAgent()
                reward += r
                reward_plot.append(reward)
                i += 1
            ax.plot(reward_plot, alpha = 0.15)
            reward_avg_plot.append(np.array(reward_plot))
            gw.resetQtable()
    
    elif agentName == 'better':
        for t in range(trail):        
            i = 0
            reward = 0
            reward_plot= []
            while i < steps:
                p, r , s = gw.betterAgent()
                reward += r
                reward_plot.append(reward)
                i += 1
            ax.plot(reward_plot, alpha = 0.15)
            reward_avg_plot.append(np.array(reward_plot))
            gw.resetQtable()
            
    elif agentName == 'worse':
        for t in range(trail):        
            i = 0
            reward = 0
            reward_plot= []
            while i < steps:
                p, r , s = gw.worseAgent()
                reward += r
                reward_plot.append(reward)
                i += 1
            ax.plot(reward_plot, alpha = 0.15)
            reward_avg_plot.append(np.array(reward_plot))
            gw.resetQtable()

    elif agentName == 'q':
        for t in range(trail):        
            i = 0
            reward = 0
            reward_plot= []
            while i < steps:
                p, r , s = gw.qAgent()
                reward += r
                reward_plot.append(reward)
                i += 1
            ax.plot(reward_plot, alpha = 0.15)
            reward_avg_plot.append(np.array(reward_plot))
            gw.resetQtable()    
        
    else:
        raise Exception('wrong agent name')
    mean_reward_plot = np.mean(reward_avg_plot, axis = 0)
    ax.plot(mean_reward_plot, color = 'black')
    ax.set_ylabel('Cumulative Reward')
    ax.set_xlabel('Steps')
    ax.set_title(agentName+' policy')
    plt.show()
    
def allInOne(trail = 10, steps = 10000, randomGoal = False):
    gw = gridworld(randomGoal = randomGoal)
    reward_avg_plot = []
    fig, ax = plt.subplots()
    
    for t in range(trail):  ## random policy
        
        i = 0
        reward = 0
        reward_plot= []
        while i < steps:
            p, r , s = gw.randomAgent()
            reward += r
            reward_plot.append(reward)
            i += 1
        ax.plot(reward_plot, alpha = 0.15)
        reward_avg_plot.append(np.array(reward_plot))
        gw.resetQtable()
    mean_reward_plot = np.mean(reward_avg_plot, axis = 0)
    ax.plot(mean_reward_plot, color = 'r', label = 'Random policy')
    reward_avg_plot = []
    
    for t in range(trail):        ##better policy
        i = 0
        reward = 0
        reward_plot= []
        while i < steps:
            p, r , s = gw.betterAgent()
            reward += r
            reward_plot.append(reward)
            i += 1
        ax.plot(reward_plot, alpha = 0.15)
        reward_avg_plot.append(np.array(reward_plot))
        gw.resetQtable()
    mean_reward_plot = np.mean(reward_avg_plot, axis = 0)
    ax.plot(mean_reward_plot, color = 'g', label = 'Better policy')
    reward_avg_plot = []

    for t in range(trail):        ##worse policy
        i = 0
        reward = 0
        reward_plot= []
        while i < steps:
            p, r , s = gw.worseAgent()
            reward += r
            reward_plot.append(reward)
            i += 1
        ax.plot(reward_plot, alpha = 0.15)
        reward_avg_plot.append(np.array(reward_plot))
        gw.resetQtable()
    mean_reward_plot = np.mean(reward_avg_plot, axis = 0)
    ax.plot(mean_reward_plot, color = 'b', label = 'Worse policy')
    reward_avg_plot = []
    
    for t in range(trail):      ##Q learning agent    
        i = 0
        reward = 0
        reward_plot= []
        while i < steps:
            p, r , s = gw.qAgent()
            reward += r
            reward_plot.append(reward)
            i += 1
        ax.plot(reward_plot, alpha = 0.15)
        reward_avg_plot.append(np.array(reward_plot))
        gw.resetQtable()    
        
    mean_reward_plot = np.mean(reward_avg_plot, axis = 0)
    ax.plot(mean_reward_plot, color = 'black', label = 'learning policy')
    
    ax.set_ylabel('Cumulative Reward')
    ax.set_xlabel('Steps')
    ax.set_title('Policy Comparison')
    ax.legend(loc="upper left")
    plt.show()
