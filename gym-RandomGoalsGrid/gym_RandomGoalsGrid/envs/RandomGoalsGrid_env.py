# -*- coding: utf-8 -*-
#
#import gym
#from gym import error, spaces, utils
#from gym.utils import seeding
#
#class FooEnv(gym.Env):
#  metadata = {'render.modes': ['human']}
#
#  def __init__(self):
#    ...
#  def step(self, action):
#    ...
#  def reset(self):
#    ...
#  def render(self, mode='human', close=False):
#    ...

import numpy as np
import cv2
import scipy.misc
#import random
import itertools

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class gameOb():
    def __init__(self,coordinates,size,intensity,channel,reward,name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
        
class RandomGoalsGrid(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def customInit(self, **kwargs):
#        
#        self.sizeX = 5
#        self.sizeY = 5
        if 'size' in kwargs:
            self.sizeX = kwargs.get("size")
            self.sizeY = self.sizeX
#            
#        self.actions = 4
#        self.objects = [] 
        
#        self.partial = False
        if 'partial' in kwargs:
            self.partial = kwargs.get("partial")
        
#        self.maxGameLength = 50
        if 'maxGameLength' in kwargs:
            self.maxGameLength = kwargs.get("maxGameLength")

#        self.currentStepIndex = 0
        
#        self.pixelsEnv = True
        if 'pixelsEnv' in kwargs:
            self.pixelsEnv = kwargs.get("pixelsEnv")
        
#        self.objectSize = 4
        if 'objectSize' in kwargs:
            self.objectSize = kwargs.get("objectSize")
        
#        self.frameSize = 84
        if 'frameSize' in kwargs:
            self.frameSize = kwargs.get("frameSize")
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, self.frameSize, self.frameSize), dtype=np.float64)
        
#        self.replacement = True
        if 'replacement' in kwargs:
            self.replacement = kwargs.get("replacement")
        
#        self.negativeReward = -1
        if 'negativeReward' in kwargs:
            self.negativeReward = kwargs.get("negativeReward")
            
#        self.positiveReward = 1
        if 'positiveReward' in kwargs:
            self.positiveReward = kwargs.get("positiveReward")
            
#        self.emptySquareReward = -0.1
        if 'emptySquareReward' in kwargs:
            self.emptySquareReward = kwargs.get("emptySquareReward")
            
    def __init__(self,partial = False,size = 5, pixelsEnv = True, frameSize=84, maxGameLength = 50, objectSize = 4, replacement = True):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.maxGameLength = maxGameLength
        self.currentStepIndex = 0
        self.pixelsEnv = pixelsEnv
        self.objectSize = objectSize
        self.frameSize = frameSize
        self.replacement = replacement
        self.goalIntensity = 0.114
        self.fireIntensity = 0.299
        self.heroIntensity = 0.587
        self.backgroundIntensity = 1.0
        self.borderIntensity = 0.0
        self.positiveReward = 1
        self.negativeReward = -1
        self.emptySquareReward = -0.1
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1, frameSize, frameSize), dtype=np.float64)
        self.action_space = gym.spaces.Discrete(n=self.actions)
#        a = self.reset()
#        plt.imshow(a,interpolation="nearest")
        
        
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(),1,self.heroIntensity,2,None,'hero')
        self.objects.append(hero)
        bug = gameOb(self.newPosition(),1,self.goalIntensity,1,self.positiveReward,'goal')
        self.objects.append(bug)
        hole = gameOb(self.newPosition(),1,self.fireIntensity,0,self.negativeReward,'fire')
        self.objects.append(hole)
        bug2 = gameOb(self.newPosition(),1,self.goalIntensity,1,self.positiveReward,'goal')
        self.objects.append(bug2)
        hole2 = gameOb(self.newPosition(),1,self.fireIntensity,0, self.negativeReward,'fire')
        self.objects.append(hole2)
        bug3 = gameOb(self.newPosition(),1,self.goalIntensity,1, self.positiveReward,'goal')
        self.objects.append(bug3)
        bug4 = gameOb(self.newPosition(),1,self.goalIntensity,1,self.positiveReward,'goal')
        self.objects.append(bug4)
        state = self.render()

        self.state = state
        return state

    def moveChar(self,direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        #8 up, 2 down, 4 left, 6 right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 8 and hero.y >= 1:
            hero.y -= 1
        if direction == 2 and hero.y <= self.sizeY-2:
            hero.y += 1
        if direction == 4 and hero.x >= 1:
            hero.x -= 1
        if direction == 6 and hero.x <= self.sizeX-2:
            hero.x += 1          
        
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
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
                    if other.reward == 1:
                        self.objects.append(gameOb(self.newPosition(), 1, self.goalIntensity, 1, self.positiveReward, 'goal'))
                        greenCount+=1
                    else: 
                        self.objects.append(gameOb(self.newPosition(), 1, self.fireIntensity, 0, self.negativeReward, 'fire'))
                    return other.reward,False 
                else:
                    rw = other.reward
        #if only the hero is left in the objects then game over. if there are no goals, game over.
        if self.replacement == False and (len(self.objects) == 1 or greenCount == 0):
            done = True
        
        return rw, done

    def render(self,mode='human', close=False):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([1, self.sizeY+2,self.sizeX+2])*self.backgroundIntensity
        a[0, 1 : -1, 1 : -1] = self.borderIntensity
        hero = None
        for item in self.objects:
            a[0, item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[0, hero.y:hero.y+3,hero.x:hero.x+3]
            
#        objectPixelSize = (self.sizeX+2)*self.objectSize
        
#        a = scipy.misc.imresize(a[0,:,:],[self.frameSize,self.frameSize],interp='nearest')
#        c = scipy.misc.imresize(a[1,:,:],[self.frameSize,self.frameSize],interp='nearest')
#        d = scipy.misc.imresize(a[2,:,:],[self.frameSize,self.frameSize],interp='nearest')
        
        a = cv2.resize(a[0,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)
#        c = cv2.resize(a[1,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)
#        d = cv2.resize(a[2,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)        
            
#        a = np.stack([b,c,d],axis=0)
        return np.reshape(a, [1,self.frameSize, self.frameSize])
#        return a
#    
#    def renderEnv2(self):
#        a = np.zeros((self.sizeX, self.sizeY))
#        for item in self.objects:
#            if(item.name == "fire"):
#                a[item.x, item.y] = 9
#            if(item.name == "hero"):
#                a[item.x, item.y] = 4
#            if(item.name == "goal"):
#                a[item.x, item.y] = 1
#        return a
        
        
    def renderEnv2(self):
        a = np.ones(self.sizeX *self.sizeY)
        a[:] = 1
        for item in self.objects:
            itemIndex = item.y * self.sizeX + item.x
            if(item.name == "fire"):
                a[itemIndex] = 2
            if(item.name == "hero"):
                a[itemIndex] = 3
            if(item.name == "goal"):
                a[itemIndex] = 4
        return a

    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        
#        if self.pixelsEnv:
#            #render image
        state = self.render()
#        else:
#            #render matrix form
#            state = self.renderEnv2()
        
        #increment steps in the current game until it reaches maxGameLength steps
        self.currentStepIndex += 1
        if self.currentStepIndex == self.maxGameLength:
            done = True
            self.currentStepIndex = 0
        if reward == None:
            print(done)
            print(reward)
            print(penalty)
            return state,(reward+penalty),done, {}
        else:
            return state,(reward+penalty),done, {}
