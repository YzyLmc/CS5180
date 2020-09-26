#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:39:34 2020

@author: ziyi
"""


from tenArm import tenArm
import numpy as np
import matplotlib.pyplot as plt


n_times = 10000
trials = 2000

tenArm = tenArm(n_times = n_times, trials = trials,randomWalk = True)

reward_plot1, optact_plot1, opt1, std1 = tenArm.eGreedySA7() #sample average


fig,ax = plt.subplots()

x = np.linspace(0,len(reward_plot1)-1,len(reward_plot1))

ax.plot(x,reward_plot1,color = 'r', label = 'sample average')
ax.fill_between(x, reward_plot1-1.96*std1/np.sqrt(trials), reward_plot1+1.96*std1/np.sqrt(trials), alpha = 0.2, color = 'r')
ax.plot(x, opt1,color = 'black', linestyle = 'dashed')
ax.set_xlabel('Steps')
ax.set_ylabel('Average Reward')

#%% %optimal action of sample average
fig1,ax1 = plt.subplots()

ax1.plot(x,optact_plot1,color = 'r', label = 'sample average')

ax1.set_xlabel('Steps')
ax1.set_ylabel('% Optimal action')


#%%  constant step size


reward_plot2, optact_plot2, opt2, std2 = tenArm.eGreedyCS7()


x = np.linspace(0,len(reward_plot2)-1,len(reward_plot2))

ax.plot(x,reward_plot2,color = 'g', label = 'constant stepsize')
ax.fill_between(x, reward_plot2-1.96*std2/np.sqrt(trials), reward_plot2+1.96*std2/np.sqrt(trials), alpha = 0.2, color = 'g')
ax.plot(x,opt2, color = 'black', linestyle = 'dashed')
ax.legend(loc = 'lower right')

#%% % optimal action of constant step

ax1.plot(x,optact_plot2,color = 'g', label = 'constant stepsize')
ax1.legend(loc = 'lower right')
plt.show()