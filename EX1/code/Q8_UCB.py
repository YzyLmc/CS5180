#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 00:56:56 2020

@author: ziyi
"""
from tenArm import tenArm
import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.1
n_times = 1000
trials = 20

tenArm = tenArm(n_times = n_times, trials = trials)

reward_plot1, optact_plot1, opt1 = tenArm.UCB()
std1 = np.std(reward_plot1)


fig,ax = plt.subplots()

x = np.linspace(0,len(reward_plot1)-1,len(reward_plot1))

ax.plot(x,reward_plot1,color = 'r')
ax.fill_between(x, reward_plot1-1.96*std1/np.sqrt(trials), reward_plot1+1.96*std1/np.sqrt(trials), alpha = 0.2, color = 'r')
ax.hlines(opt1,0,n_times, color = 'r', linestyles = 'dashed')


plt.show()