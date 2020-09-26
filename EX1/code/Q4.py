#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:06:51 2020

@author: ziyi
"""


from tenArm import tenArm

import matplotlib.pyplot as plt

tenArm = tenArm()
collectn_1 = tenArm.manyPull(1)
collectn_2 = tenArm.manyPull(2)
collectn_3 = tenArm.manyPull(3)
collectn_4 = tenArm.manyPull(4)
collectn_5 = tenArm.manyPull(5)
collectn_6 = tenArm.manyPull(6)
collectn_7 = tenArm.manyPull(7)
collectn_8 = tenArm.manyPull(8)
collectn_9 = tenArm.manyPull(9)
collectn_10 = tenArm.manyPull(10)

## combine these different collections into a list
data_to_plot = [collectn_1, collectn_2, collectn_3, collectn_4, collectn_5,
                collectn_6, collectn_7, collectn_8, collectn_9, collectn_10]

# Create a figure instance
fig = plt.figure()

# Create an axes instance
ax = fig.add_axes([0,0,1,1])

# Create the boxplot
bp = ax.violinplot(data_to_plot)
ax.set_xlabel('Action')
ax.set_ylabel('Reward Distribution')
ax.hlines(0,1,10,linestyle='dashed')
plt.show()