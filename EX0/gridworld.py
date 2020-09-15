#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 22:41:01 2020

@author: ziyi
"""
import numpy as np
import random

class gridworld():
    
    def __init__(self, randomGoal = False):
        self.map = np.array([[0,0,0,0,0,-1,0,0,0,0,0],  ##map flipped in matrix form
                            [0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,-1,-1,0,-1,-1],
                            [-1,0,-1,-1,-1,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            [0,0,0,0,0,-1,0,0,0,0,0],
                            ])
        self.agentPos = np.array([0,0])
        self.actionList = {'u':np.array([0,1]),'d':np.array([0,-1]),
                           'l':np.array([-1,0]),'r':np.array([1,0])}
        self.n_steps = 0
        self.path = [self.agentPos]
        
        
        if randomGoal:
            self.randomGoal
        else:
            self.goalPos = np.array([10,10])
        self.map[self.goalPos[0], self.goalPos[1]] = 1
                
    def randomGoal(self):
        notDone = True
        while notDone:
            g = np.array([random.randrange(0,11)] + [random.randrange(0,11)])
            if self.__isValid(g) and g != self.agentPos:
                self.goalPos = g
                notDone = False
            

    def setAgentPos(self,pos):
        if self.__isvalid(pos):
            self.agentPos = pos  ##pos is an array as [x,y]
        else:
            raise Exception("input pos invalid")
            
                                
    def __isValid(self,pos): ##if a pos = [x,y] is valid, x&y is integer        
        
        if pos[0] > 10 or pos[0] < 0 or pos[1] >10 or pos[1] < 0:
            return False
        elif self.map[pos[0],pos[1]] == -1:
            return False
        else:
            return True
        
    def legalActions(self,pos):
        legalActs = []
        if self.__isValid(pos + np.array([0,1])):
            legalActs.append('u')
        if self.__isValid(pos + np.array([0,-1])):
            legalActs.append('d')
        if self.__isValid(pos + np.array([1,0])):
            legalActs.append('r')
        if self.__isValid(pos + np.array([-1,0])):
            legalActs.append('l')
        
        return legalActs
    
    def nextPos(self,action):
        if action not in self.actionList:
            print("action not in list")
        elif self.map[self.agentPos[0],self.agentPos[1]] == 1:
            print('New episode!')
            self.resetAgent()
        else:
            legalActs = self.legalActions(self.agentPos)
            key = random.random()
            if action == 'u':
                if key < 0.8:
                    finalAction = 'u'
                elif key < 0.9:
                    finalAction = 'l'
                else:
                    finalAction = 'r'
            elif action == 'l':
                if key < 0.8:
                    finalAction = 'l'
                elif key < 0.9:
                    finalAction = 'u'
                else:
                    finalAction = 'd'
            elif action == 'd':
                if key < 0.8:
                    finalAction = 'd'
                elif key < 0.9:
                    finalAction = 'r'
                else:
                    finalAction = 'l'
            elif action == 'r':
                if key < 0.8:
                    finalAction = 'r'
                elif key < 0.9:
                    finalAction = 'd'
                else:
                    finalAction = 'u'
            
            if finalAction in legalActs:
                nextPos = self.agentPos + self.actionList[finalAction]
                self.agentPos = nextPos
            else:
                self.agentPos = self.agentPos
    
    def resetAgent(self):
        self.agentPos = np.array([0,0])
        self.path = [self.agentPos]
        
############################################################################
############################################################################
    def randomAgent(self):
        act = random.choice(['u','d','l','r'])
        self.nextPos(act)
        self.n_steps += 1
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]]
    
    
    def manualAgent(self):
        act = input("choose act from ['u', 'd', 'l, 'r'] \t")
        self.nextPos(act)
        self.n_steps += 1
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]]
    
    
    def betterAgent(self):   ###has a preference of moving to top-right corner
        key = random.random()
        if key < 0.2:
            self.nextPos('u')
        elif key < 0.4:
            self.nextPos('r')
        else:
            act = random.choice(['u','d','l','r'])
            self.nextPos(act)
        self.n_steps += 1
        
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]]
    
        
    def worseAgent(self): ##all the way up
        self.nextPos('u')
        self.n_steps += 1
        
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]]
        
        
            
