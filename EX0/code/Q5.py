#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 19:12:05 2020

@author: ziyi
"""

from evaluate import evaluate, allInOne, plotSteps

agentName = 'q'

evaluate(agentName, randomGoal = True)
#%%
allInOne(randomGoal = True)
#%%
plotSteps(randomGoal = True)