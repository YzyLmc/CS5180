#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:39:34 2020

@author: ziyi
"""


from tenArm import tenArm
import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.1
n_times = 10000
trials = 20

tenArm = tenArm(randomWalk = True)

reward_plot1, optact_plot1, opt1 = tenArm.eGreedySA7()
std1 = np.std(reward_plot1)
reward_plot2, optact_plot2, opt2 = tenArm.eGreedySA7(epsilon = 0.01)
std2 = np.std(reward_plot2)
reward_plot3, optact_plot3, opt3 = tenArm.eGreedySA7(epsilon = 0)
std3 = np.std(reward_plot3)

fig,ax = plt.subplots()

x = np.linspace(0,len(reward_plot1)-1,len(reward_plot1))

ax.plot(x,reward_plot1,color = 'r')
ax.fill_between(x, reward_plot1-1.96*std1/np.sqrt(trials), reward_plot1+1.96*std1/np.sqrt(trials), alpha = 0.2, color = 'r')
ax.hlines(opt1,0,n_times, color = 'r', linestyles = 'dashed')

ax.plot(reward_plot2,color = 'g')
ax.fill_between(x, reward_plot2-1.96*std2/np.sqrt(trials), reward_plot2+1.96*std2/np.sqrt(trials), alpha = 0.2, color = 'g')
ax.hlines(opt2,0,n_times, color = 'g', linestyles = 'dashed')

ax.plot(reward_plot3,color = 'b')
ax.fill_between(x, reward_plot3-1.96*std3/np.sqrt(trials), reward_plot3+1.96*std3/np.sqrt(trials), alpha = 0.2, color = 'b')
ax.hlines(opt3,0,n_times, color = 'b', linestyles = 'dashed')

plt.show()

#%%  constant step size


reward_plot1, optact_plot1, opt1 = tenArm.eGreedySA7()
std1 = np.std(reward_plot1)
reward_plot2, optact_plot2, opt2 = tenArm.eGreedySA7(epsilon = 0.01)
std2 = np.std(reward_plot2)
reward_plot3, optact_plot3, opt3 = tenArm.eGreedySA7(epsilon = 0)
std3 = np.std(reward_plot3)

fig,ax = plt.subplots()

x = np.linspace(0,len(reward_plot1)-1,len(reward_plot1))

ax.plot(x,reward_plot1,color = 'r')
ax.fill_between(x, reward_plot1-1.96*std1/np.sqrt(trials), reward_plot1+1.96*std1/np.sqrt(trials), alpha = 0.2, color = 'r')
ax.hlines(opt1,0,n_times, color = 'r', linestyles = 'dashed')

ax.plot(reward_plot2,color = 'g')
ax.fill_between(x, reward_plot2-1.96*std2/np.sqrt(trials), reward_plot2+1.96*std2/np.sqrt(trials), alpha = 0.2, color = 'g')
ax.hlines(opt2,0,n_times, color = 'g', linestyles = 'dashed')

ax.plot(reward_plot3,color = 'b')
ax.fill_between(x, reward_plot3-1.96*std3/np.sqrt(trials), reward_plot3+1.96*std3/np.sqrt(trials), alpha = 0.2, color = 'b')
ax.hlines(opt3,0,n_times, color = 'b', linestyles = 'dashed')

plt.show()