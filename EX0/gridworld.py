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
        self.qTable= {}
        #self.goalPos = np.array([0,0])
        
        
        if randomGoal:
            self.randomGoal()
        else:
            self.goalPos = np.array([10,10])
        self.map[self.goalPos[0], self.goalPos[1]] = 1
                
    def randomGoal(self):
        notDone = True
        while notDone:
            g = np.array([random.randrange(0,11)] + [random.randrange(0,11)])
            if self.__isValid(g) and repr(g) != repr(self.agentPos):
                self.goalPos = g
                notDone = False
                print('random goalPos:',self.goalPos)
            

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
            #print('New episode!')
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
            
            return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]]
    
    def resetAgent(self):
        self.agentPos = np.array([0,0])
        self.n_steps = 0
    def getQvalue(self, pos, act):
        if repr(pos) in self.qTable:
            if act in self.qTable[repr(pos)]:               
                return self.qTable[repr(pos)][act]
            else:
                self.qTable[repr(pos)][act] = 0
                return 0
                
        else:
            self.qTable[repr(pos)] = {}
            self.qTable[repr(pos)][act] = 0
            return 0
                   
    def resetQtable(self):
        self.qTable = {} 
############################################################################
############################################################################
    def randomAgent(self):
        act = random.choice(['u','d','l','r'])
        self.nextPos(act)
        self.n_steps += 1
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]], self.n_steps
    
    
    def manualAgent(self):
        act = input("choose act from ['u', 'd', 'l, 'r'] \t")
        self.nextPos(act)
        self.n_steps += 1
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]], self.n_steps
    
    
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
        
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]], self.n_steps
    
        
    def worseAgent(self): ##all the way up
        self.nextPos('r')
        self.n_steps += 1
        
        return self.agentPos, self.map[self.agentPos[0],self.agentPos[1]], self.n_steps
    
    
    def qAgent(self, epsilon = 0.2, alpha = 0.1, discount = 0.9):
        actlist = ['u','d','l','r']
        random.shuffle(actlist)      
        key = random.random()
        
        if self.agentPos[0] == self.goalPos[0] and self.agentPos[1] == self.goalPos[1]:            
            steps = self.n_steps
            reward = 0
            self.randomAgent()

            #print(self.getQvalue(np.array([10,9]),'u'), self.getQvalue(np.array([9,10]),'r'))
            
        else:
            if key < epsilon:
                act = random.choice(actlist)
                
            else:
                act = max(actlist, key = lambda act: self.getQvalue(self.agentPos,act))
            
            currentPos = self.agentPos       
            nextPos, reward = self.nextPos(act)
            currentQ = self.getQvalue(currentPos,act)
            act_pi = max(actlist, key=lambda a: self.getQvalue(nextPos,a))
            nextQ = self.getQvalue(nextPos, act_pi)
            
            self.qTable[repr(currentPos)][act] = currentQ + alpha*(reward + discount*nextQ - currentQ)  
            self.n_steps += 1
            steps = self.n_steps
                
            
        return self.agentPos, reward, steps
        
            
                
        
        
        
            
