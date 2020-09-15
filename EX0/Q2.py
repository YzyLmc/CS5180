#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 02:18:08 2020

@author: ziyi
"""


from gridworld import gridworld

gw = gridworld()

while True:
    p, r = gw.manualAgent()    
    print('Pos',p)
    print('reward',r)