#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 03:02:20 2020

@author: ziyi
"""


from gridworld import gridworld
import matplotlib.pyplot as plt

gw = gridworld()

i = 0
reward = 0
reward_plot= []
while i < 10000:
    p, r = gw.randomAgent()
    reward += r
    reward_plot.append(reward)
    i += 1
    
plt.figure()
plt.plot(reward_plot)
plt.show()