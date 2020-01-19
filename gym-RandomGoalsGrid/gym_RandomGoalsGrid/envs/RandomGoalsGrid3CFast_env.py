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
        
class RandomGoalsGrid3CFast(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def customInit(self, **kwargs):
        if 'size' in kwargs:
            self.sizeX = kwargs.get("size")
            self.sizeY = self.sizeX
        
        if 'partial' in kwargs:
            self.partial = kwargs.get("partial")
        
        if 'maxGameLength' in kwargs:
            self.maxGameLength = kwargs.get("maxGameLength")

        
        if 'pixelsEnv' in kwargs:
            self.pixelsEnv = kwargs.get("pixelsEnv")
        
        if 'objectSize' in kwargs:
            self.objectSize = kwargs.get("objectSize")
        
        if 'frameSize' in kwargs:
            self.frameSize = kwargs.get("frameSize")
            self.objectSize = (int)(self.frameSize/(self.sizeX + 2))
        
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, self.frameSize, self.frameSize), dtype=np.float16)
        
        if 'replacement' in kwargs:
            self.replacement = kwargs.get("replacement")
        
        if 'negativeReward' in kwargs:
            self.negativeReward = kwargs.get("negativeReward")
            
        if 'positiveReward' in kwargs:
            self.positiveReward = kwargs.get("positiveReward")
            
        if 'emptySquareReward' in kwargs:
            self.emptySquareReward = kwargs.get("emptySquareReward")
        
        
    def __init__(self,partial = False,size = 5, pixelsEnv = True, frameSize=84, maxGameLength = 50, replacement = True):
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
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, self.frameSize, self.frameSize), dtype=np.float16)
        self.action_space = gym.spaces.Discrete(n=self.actions)
#        a = self.reset()
#        plt.imshow(a,interpolation="nearest")
        
    def seed(self, seed):
        np.random.seed(seed)
        
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
        if self.pixelsEnv:
            state = self.pixelStateInit()
#            state = self.renderEnv2()
        else:
            state = self.renderEnv2()
        self.state = state
        return np.copy(self.state)

    # def generate_random_state(self):


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

#       if hero moved, then color its old position with the background, and color the new pos with the hero color
        if heroNewX != hero.x or heroNewY != hero.y:
            oldPixelPosX0 = (hero.x + 1) * self.objectSize
            oldPixelPosY0 = (hero.y + 1) * self.objectSize
            newPixelPosX0 = (heroNewX + 1) * self.objectSize
            newPixelPosY0 = (heroNewY + 1) * self.objectSize
            
            if self.pixelsEnv:
                #color the old position of the hero with the background. only the channel 2 needs this, which is the channel of the hero
                #+1 is to account for the border in both axes
                self.state[:, oldPixelPosY0: oldPixelPosY0 + self.objectSize, oldPixelPosX0 : oldPixelPosX0 + self.objectSize] = 0
                #color the new pos with the background first (to erase the possible existing object in this pos)
                self.state[:, newPixelPosY0: newPixelPosY0 + self.objectSize, newPixelPosX0 : newPixelPosX0 + self.objectSize] = 0
                #color the new pos with hero's color on second channelgameOb(self.newPosition()
                self.state[2, newPixelPosY0: newPixelPosY0 + self.objectSize, newPixelPosX0 : newPixelPosX0 + self.objectSize] = 1
            
            #update the hero's new coordinates
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
                    newX = newPosition[0]
                    newY = newPosition[1]
                    newPixelX0 = (newX + 1) * self.objectSize
                    newPixelY0 = (newY + 1) * self.objectSize
                    #set background intensity to all channels in the new position
                    self.state[:, newPixelY0: newPixelY0 + self.objectSize, newPixelX0 : newPixelX0 + self.objectSize] = 0
                    
                    if other.reward == 1:
                        self.objects.append(gameOb(newPosition, 1, 1, 1, 1, 'goal'))
                        #set the 'goal' intensity on the channel 1
                        self.state[1, newPixelY0: newPixelY0 + self.objectSize, newPixelX0: newPixelX0 + self.objectSize] = 1
                        greenCount+=1
                    else: 
                        self.objects.append(gameOb(newPosition, 1, 1, 0, -1, 'fire'))
                        #set the 'fire' intensity on the channel 0
                        self.state[0, newPixelY0 : newPixelY0 + self.objectSize, newPixelX0: newPixelX0 + self.objectSize] = 1
                                                
                    return other.reward,False 
                else:
                    rw = other.reward
        #if only the hero is left in the objects then game over. if there are no goals, game over.
        if self.replacement == False and (len(self.objects) == 1 or greenCount == 0):
            done = True
        
        return rw, done

    def pixelStateInit(self):
        #a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([3, self.sizeY+2,self.sizeX+2])
        a[:, 1 : -1, 1 : -1] = 0.
        hero = None
        for item in self.objects:
            a[item.channel, item.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[:,hero.y:hero.y+3,hero.x:hero.x+3]
            
#        objectPixelSize = (self.sizeX+2)*self.objectSize
        
#        b = scipy.misc.imresize(a[0,:,:],[self.frameSize,self.frameSize],interp='nearest')
#        c = scipy.misc.imresize(a[1,:,:],[self.frameSize,self.frameSize],interp='nearest')
#        d = scipy.misc.imresize(a[2,:,:],[self.frameSize,self.frameSize],interp='nearest')
        b = cv2.resize(a[0,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)
        c = cv2.resize(a[1,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)
        d = cv2.resize(a[2,:,:],(self.frameSize,self.frameSize),interpolation=cv2.INTER_NEAREST)              
        a = np.stack([b,c,d],axis=0)
        
        return a

    def renderEnv(self):
        return np.copy(self.state)
   
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

    def step(self,action):
        penalty = self.moveChar(action)
        reward,done = self.checkGoal()
        
        if self.pixelsEnv:
            #render image
#            state = self.state
            state = self.renderEnv()
#            state2 = self.renderEnv2()
#            if (state == state2) is False:
#                print("States not equal")
#                input()
        else:
            #render matrix form
            state = self.renderEnv2()
        
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

