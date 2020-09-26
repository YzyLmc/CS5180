#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 23:44:48 2020

@author: ziyi
"""


from tenArm import tenArm
import numpy as np
import matplotlib.pyplot as plt


n_times = 1000
trials = 2000

tenArm = tenArm(n_times = n_times, trials = trials)

reward_plot1, optact_plot1, opt1, std1 = tenArm.eGreedySA(epsilon = 0.1)

reward_plot2, optact_plot2, opt2, std2 = tenArm.eGreedySA(epsilon = 0.01)

reward_plot3, optact_plot3, opt3, std3 = tenArm.eGreedySA(epsilon = 0)


fig,ax = plt.subplots()

x = np.linspace(0,len(reward_plot1)-1,len(reward_plot1))

ax.plot(x,reward_plot1,color = 'r', label = 'e = 0.1')
ax.fill_between(x, reward_plot1-1.96*std1/np.sqrt(trials), reward_plot1+1.96*std1/np.sqrt(trials), alpha = 0.2, color = 'r')
ax.hlines(opt1,0,n_times, color = 'r', linestyles = 'dashed')

ax.plot(reward_plot2,color = 'g', label = 'e = 0.01')
ax.fill_between(x, reward_plot2-1.96*std2/np.sqrt(trials), reward_plot2+1.96*std2/np.sqrt(trials), alpha = 0.2, color = 'g')
ax.hlines(opt2,0,n_times, color = 'g', linestyles = 'dashed')

ax.plot(reward_plot3,color = 'b', label = 'e = 0')
ax.fill_between(x, reward_plot3-1.96*std3/np.sqrt(trials), reward_plot3+1.96*std3/np.sqrt(trials), alpha = 0.2, color = 'b')
ax.hlines(opt3,0,n_times, color = 'b', linestyles = 'dashed')

ax.set_xlabel('Steps')
ax.set_ylabel('Average Reward')
ax.legend(loc = 'lower right')
plt.show()

#%%
fig1,ax1 = plt.subplots()

ax1.plot(x,optact_plot1,color = 'r', label = 'e = 0.1')

ax1.plot(x,optact_plot2,color = 'g', label = 'e = 0.01')

ax1.plot(x,optact_plot3,color = 'b', label = 'e = 0')

ax1.set_xlabel('Steps')
ax1.set_ylabel('% Optimal action')
ax1.legend(loc = 'lower right')
plt.show()