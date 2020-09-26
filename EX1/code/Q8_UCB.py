#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 00:56:56 2020

@author: ziyi
"""
from tenArm import tenArm
import numpy as np
import matplotlib.pyplot as plt


n_times = 1000
trials = 2000

tenArm = tenArm(n_times = n_times, trials = trials)

reward_plot1, optact_plot1, opt1, std1 = tenArm.UCB()
reward_plot2, optact_plot2, opt2, std2 = tenArm.eGreedySA()


fig,ax = plt.subplots()

x = np.linspace(0,len(reward_plot1)-1,len(reward_plot1))

ax.plot(x,reward_plot1,color = 'r',label='UCB')
ax.fill_between(x, reward_plot1-1.96*std1/np.sqrt(trials), reward_plot1+1.96*std1/np.sqrt(trials), alpha = 0.2, color = 'r')
ax.hlines(opt1,0,n_times, color = 'r', linestyles = 'dashed')

ax.plot(x,reward_plot2,color = 'g', label = 'eGreedy')
ax.fill_between(x, reward_plot2-1.96*std2/np.sqrt(trials), reward_plot2+1.96*std2/np.sqrt(trials), alpha = 0.2, color = 'g')
ax.hlines(opt2,0,n_times, color = 'g', linestyles = 'dashed')
ax.legend(loc="lower right")
ax.set_xlabel('Steps')
ax.set_ylabel('Average Reward')
#%%
fig1,ax1 = plt.subplots()

ax1.plot(x,optact_plot1,color = 'r', label = 'UCB')

ax1.plot(x,optact_plot2,color = 'g', label = 'eGreedy')

ax1.set_xlabel('Steps')
ax1.set_ylabel('% Optimal action')
ax1.legend(loc="lower right")
plt.show()

