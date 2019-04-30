# -*- coding: utf-8 -*-
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import cv2
import scipy
import scipy.misc

import itertools


class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class RandomGoalsGridLinear(gym.Env):
    metadata = {'render.modes': ['human']}    
              
    def __init__(self,partial = False,size = 5, pixelsEnv = True, frameSize=21, maxGameLength = 50, replacement = True):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.maxGameLength = maxGameLength
        self.currentStepIndex = 0
        self.pixelsEnv = pixelsEnv
        self.objectSize = (int)(frameSize/(size + 2))
        self.frameSize = frameSize
        self.replacement = replacement
        self.goalIntensity = 1
        self.fireIntensity = 1
        self.heroIntensity = 1
        self.backgroundIntensity = 0
        self.borderIntensity = 0
        self.positiveReward = 1
        self.negativeReward = -1
        self.emptySquareReward = -0.1
        self.observation_space = gym.spaces.Box(low=0, high=999999999, shape=(1, 7), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(n=self.actions)        
        
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(),1,1,2,None,'hero')
        self.objects.append(hero)
        bug = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug)
        hole = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole)
        bug2 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug2)
        hole2 = gameOb(self.newPosition(),1,1,0,-1,'fire')
        self.objects.append(hole2)
        bug3 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug3)
        bug4 = gameOb(self.newPosition(),1,1,1,1,'goal')
        self.objects.append(bug4)
        state = self.renderEnv()
        self.state = state
        return state

    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        state = self.renderEnv()       
        #increment steps in the current game until it reaches maxGameLength steps
        self.currentStepIndex += 1
        if self.currentStepIndex == self.maxGameLength:
            done = True
            self.currentStepIndex = 0
 
        return state,(reward+penalty),done, {}
 

    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        penalize = 0.
        heroNewX = hero.x
        heroNewY = hero.y
        
        if direction == 0 and hero.y >= 1:
            heroNewY -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            heroNewY += 1
        if direction == 2 and hero.x >= 1:
            heroNewX -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            heroNewX += 1     
        if hero.x == heroNewX and hero.y == heroNewY:
            penalize = 0.0

        hero.x = heroNewX
        hero.y = heroNewY

        self.objects[0] = hero
        return penalize
    
    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x,objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        
        location = np.random.choice(range(len(points)),replace=False)
        return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        done = False
        rw = 0
        greenCount = 0
        for other in others:
            if other.name == 'goal':
                greenCount += 1
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if(other.name == 'goal'):
                    greenCount -= 1
                if self.replacement:
                    newPosition = self.newPosition()
                    
                    if other.reward == 1:
                        self.objects.append(gameOb(newPosition, 1, 1, 1, 1, 'goal'))
                        greenCount+=1
                    else: 
                        self.objects.append(gameOb(newPosition, 1, 1, 0, -1, 'fire'))
                    return other.reward,False 
                else:
                    rw = other.reward
        if self.replacement == False and (len(self.objects) == 1 or greenCount == 0):
            done = True
        
        return rw, done


    def renderEnv(self):
        a = []
        for item in self.objects:
             a.append((item.channel<<self.sizeX) + item.y*self.sizeX + item.x)
        
        state = np.array(a, dtype = np.uint8)
        state = state.reshape([1,7])
        return state
    
    def renderEnvMatrix(self):
        a = np.zeros((self.sizeX, self.sizeY))
        for item in self.objects:
            if(item.name == "fire"):
                a[item.y, item.x] = 9
            if(item.name == "hero"):
                a[item.y, item.x] = 4
            if(item.name == "goal"):
                a[item.y, item.x] = 1
        return a

        
    def convertArrayToImage(self, state):
        a = np.ones([3, self.sizeY+2,self.sizeX+2])
        a[:, 1 : -1, 1 : -1] = 0
#        hero = None
        for item in state[0]:
            channel = item>>self.sizeX
            position = item % (2**self.sizeX)
            y = (int)(position / self.sizeX)
            x = (int)(position % self.sizeX)
            a[channel, y+1:y+2, x+1:x+2] = 1
#            if item.name == 'hero':
#                hero = item
#        if self.partial == True:
#            a = a[:,hero.y:hero.y+3,hero.x:hero.x+3]

        b = cv2.resize(a[0,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)
        c = cv2.resize(a[1,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)
        d = cv2.resize(a[2,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)              
        a = np.stack([b,c,d],axis=0)
        return a        
        