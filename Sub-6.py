import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import time
import os
import pickle
import warnings
import IPython

out = display(IPython.display.Pretty('Starting'), display_id=True)

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

class DQN(nn.Module):
    def __init__(self, NUMBER_OF_ARGUMENTS_PER_STATE, NUM_OF_LAYERS, NUM_OF_NEURONS_PER_LAYER, NUM_OF_ACTIONS):
        super().__init__(),
        
        self.NUM_OF_LAYERS = NUM_OF_LAYERS
        if self.NUM_OF_LAYERS == 0:
            self.fc1 = nn.Linear(in_features=NUMBER_OF_ARGUMENTS_PER_STATE, out_features=32)
        elif self.NUM_OF_LAYERS == 1:
            self.fc1 = nn.Linear(in_features=NUMBER_OF_ARGUMENTS_PER_STATE, out_features=NUM_OF_NEURONS_PER_LAYER)
            self.out_v = nn.Linear(in_features=NUM_OF_NEURONS_PER_LAYER, out_features=1)
            self.out_a = nn.Linear(in_features=NUM_OF_NEURONS_PER_LAYER, out_features=32)
        elif self.NUM_OF_LAYERS == 2:
            self.fc1 = nn.Linear(in_features=NUMBER_OF_ARGUMENTS_PER_STATE, out_features=NUM_OF_NEURONS_PER_LAYER)
            self.fc2 = nn.Linear(in_features=NUM_OF_NEURONS_PER_LAYER, out_features=NUM_OF_NEURONS_PER_LAYER)
            self.out_v = nn.Linear(in_features=NUM_OF_NEURONS_PER_LAYER, out_features=1)
            self.out_a = nn.Linear(in_features=NUM_OF_NEURONS_PER_LAYER, out_features=NUM_OF_ACTIONS)

    def forward(self, t):
        
        t = t.flatten(start_dim=1)
        
        if self.NUM_OF_LAYERS == 0:
            t = self.fc1(t)
            q = t
            return q

        elif self.NUM_OF_LAYERS == 1:
            t = F.relu(self.fc1(t))
            v = self.out_v(t) #Value Stream
            a = self.out_a(t) # Advantage Stream
            q = v + a - a.mean()
            return q
        
        elif self.NUM_OF_LAYERS == 2:
            t = F.relu(self.fc1(t))
            t = F.relu(self.fc2(t))
            v = self.out_v(t) #Value Stream
            a = self.out_a(t) # Advantage Stream
            q = v + a - a.mean()
            return q

Experience = namedtuple(
            'Experience',
            ('state', 'action', 'next_state', 'reward')
                        )

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedyStrategy():

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
                            math.exp(-1. * current_step / self.decay)

class Agent():
    def __init__(self, strategy, num_actions, device):

        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) # explore    
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device) # exploit

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
    @staticmethod        
    def get_next(target_net, next_states):                 
        return target_net(next_states).max(dim=1)[0].detach()

def calc(SUM1,SUM2,SUM3,SUM4,Fairness, moving_avg_period, iteration):
    
    if episode == 999:
            Fairness = [element * 100 for element in Fairness]
            moving_avg_fairness = get_moving_average(moving_avg_period, Fairness)
            moving_avg_SUM1 = get_moving_average(moving_avg_period, SUM1)
            moving_avg_SUM2 = get_moving_average(moving_avg_period, SUM2)
            moving_avg_SUM1 = [element * 50 for element in moving_avg_SUM1]
            moving_avg_SUM2 = [element * 50 for element in moving_avg_SUM2]
            moving_avg_SUM3 = get_moving_average(moving_avg_period, SUM3)
            moving_avg_SUM4 = get_moving_average(moving_avg_period, SUM4)
            moving_avg_SUM3 = [element * 50 for element in moving_avg_SUM3]
            moving_avg_SUM4 = [element * 50 for element in moving_avg_SUM4]
            SUM = np.add(moving_avg_SUM1,moving_avg_SUM2)
            SUM = np.add(SUM,moving_avg_SUM3)
            SUM = np.add(SUM,moving_avg_SUM4)
            
           
            
            with open(f"Sub-6-NLoS-SUM-Rate-{iteration+1}.pickle", "wb") as f:
                   pickle.dump(SUM, f)
            with open(f"Sub-6-NLoS-FAIR-{iteration+1}.pickle", "wb") as f:
                   pickle.dump(moving_avg_fairness, f)
    else:
            out.update(IPython.display.Pretty(f'Episode: {len(SUM1)}. Iteration: {iteration+1}.'))
            
def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
    
def LineOfSight_Check(D,H):
    c = 0.6 #urban 2GHz
    d = 0.11 #urban 2GHz
    RAND = random.uniform(0,1)
    teta = math.atan(H/D) * 180/math.pi
    if teta < 15:
        return 2
    p1 = c * ((teta - 15) ** d)
    p2 = 1 - p1
    if p1 >= p2:
        if RAND >= p2:
            L = 1
        else:
            L = 2
    else:
        if RAND >= p1:
            L = 2
        else:
            L = 1
    return L
    
    
def Average(lst): 
    return sum(lst) / len(lst) 

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

class Blob():
    def __init__(self, size, USER1=False, USER2=False, 
    	USER3=False, USER4=False):
        self.size = size
        if USER1:
            self.x = 35
            self.y = 54
        elif USER2:
            self.x = 94
            self.y = 1
        elif USER3:
            self.x = 29
            self.y = 45
        elif USER4:
            self.x = 1
            self.y = 97
        else:
            self.x = 50
            self.y = 50

    def __str__(self):
        return f"Blob({self.x}, {self.y})"

    def __sub__(self, other):
        return [(self.x-other.x)/10, (self.y-other.y)/10]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        

        if choice == 0:
            self.move(x=1, y=1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H += 1

        elif choice == 1:
            self.move(x=-1, y=-1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H += 1

        elif choice == 2:
            self.move(x=-1, y=1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H += 1

        elif choice == 3:
            self.move(x=1, y=-1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H += 1
            
        elif choice == 4:
            self.move(x=1, y=1)
            self.a1 += 0.01
            self.a3 -=0.01
            self.H += 1

        elif choice == 5:
            self.move(x=-1, y=-1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 6:
            self.move(x=-1, y=1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 7:
            self.move(x=1, y=-1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H += 1
            
        elif choice == 8:
            self.move(x=1, y=1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H += 1
            
        elif choice == 9:
            self.move(x=-1, y=-1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H += 1

        elif choice == 10:
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H += 1

        elif choice == 11:
            self.move(x=1, y=-1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H += 1
            
        elif choice == 12:
            self.move(x=1, y=1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 13:
            self.move(x=-1, y=-1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 14:
            self.move(x=-1, y=1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H += 1

        elif choice == 15:
            self.move(x=1, y=-1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H += 1
            
        if choice == 16:
            self.move(x=1, y=1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H -= 1

        elif choice == 17:
            self.move(x=-1, y=-1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H -= 1

        elif choice == 18:
            self.move(x=-1, y=1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H -= 1

        elif choice == 19:
            self.move(x=1, y=-1)
            self.a1 += 0.01
            self.a3 +=0.01
            self.H -= 1
            
        elif choice == 20:
            self.move(x=1, y=1)
            self.a1 += 0.01
            self.a3 -=0.01
            self.H -= 1

        elif choice == 21:
            self.move(x=-1, y=-1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 22:
            self.move(x=-1, y=1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 23:
            self.move(x=1, y=-1)
            self.a1 += 0.01
            self.a3 -= 0.01
            self.H -= 1
            
        elif choice == 24:
            self.move(x=1, y=1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H -= 1
            
        elif choice == 25:
            self.move(x=-1, y=-1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H -= 1

        elif choice == 26:
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H -= 1

        elif choice == 27:
            self.move(x=1, y=-1)
            self.a1 -= 0.01
            self.a3 += 0.01
            self.H -= 1
            
        elif choice == 28:
            self.move(x=1, y=1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 29:
            self.move(x=-1, y=-1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 30:
            self.move(x=-1, y=1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H -= 1

        elif choice == 31:
            self.move(x=1, y=-1)
            self.a1 -= 0.01
            self.a3 -= 0.01
            self.H -= 1
            
        if self.a1 > 1:
            self.a1 = 1
        elif self.a1 < 0:
            self.a1 = 0
        if self.a3 > 1:
            self.a3 = 1
        elif self.a3 < 0:
            self.a3 = 0
        if self.H <= 10:
            self.H =10
        

    def move(self, x=False, y=False):

        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

class BlobEnv():
    SIZE = 100
    MOVE_PENALTY = 1
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    UAV_N = 1  # UAV key in dict
    USER_N = 2  # USER key in dict
    UAV2_N = 4  # UAV2 key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255),
         4: (175, 0, 255)}

    def reset(self):
        P = 1 # Transmitted power 30dbm (i.e. 1w)
        W = 5e7 # Bandwidth 50MHz
        fc = 2e9 # Carrier frequency = 2GHz
        N0 = 10e-17 # W/Hz
        LOS_PL = 1
        NLOS_PL = 20
        N = N0 * W
        c = 3e8
        lamda = c/fc

        self.UAV = Blob(self.SIZE)
        self.UAV2 = Blob(self.SIZE)
        self.SUM1 = []
        self.SUM2 = []
        self.SUM3 = []
        self.SUM4 = []
        self.Fairness = []
        
        self.UAV.a1 = 0.5
        self.UAV.a2 = 0.5
        self.UAV.a3 = 0.5
        self.UAV.a4 = 0.5
        self.UAV.H = 50
        H = self.UAV.H
        
        self.USER1 = Blob(self.SIZE, True, False, False, False)
        self.USER2 = Blob(self.SIZE, False, True, False, False)
        self.USER3 = Blob(self.SIZE, False, False, True, False)
        self.USER4 = Blob(self.SIZE, False, False, False, True)
        
        ob1 = self.UAV-self.USER1
        ob2 = self.UAV-self.USER2
        ob3 = self.UAV-self.USER3
        ob4 = self.UAV-self.USER4
        
        D1 =  np.sum(np.sqrt([(10*ob1[0])**2, (10*ob1[1])**2]))
        D2 = np.sum(np.sqrt([(10*ob2[0])**2, (10*ob2[1])**2]))
        D3 = np.sum(np.sqrt([(10*ob3[0])**2, (10*ob3[1])**2]))
        D4 = np.sum(np.sqrt([(10*ob4[0])**2, (10*ob4[1])**2]))
        
        self.L1 = LineOfSight_Check(D1,H)
        self.L2 = LineOfSight_Check(D2,H)
        self.L3 = LineOfSight_Check(D3,H)
        self.L4 = LineOfSight_Check(D4,H)
                  

        Dt1 = np.sum(np.sqrt([ (10*ob1[0])**2, (10*ob1[1])**2, H**2  ]))
        Dt2 = np.sum(np.sqrt([ (10*ob2[0])**2, (10*ob2[1])**2, H**2  ]))
        Dt3 = np.sum(np.sqrt([ (10*ob3[0])**2, (10*ob3[1])**2, H**2  ]))
        Dt4 = np.sum(np.sqrt([ (10*ob4[0])**2, (10*ob4[1])**2, H**2  ]))
        
        if self.L1 == 1:
            h1 = 20*math.log10(Dt1) + 20*math.log10(fc) - 147.56 + LOS_PL
        else:
            h1 = 20*math.log10(Dt1) + 20*math.log10(fc) - 147.56 + NLOS_PL
        if self.L2 == 1:
            h2 = 20*math.log10(Dt2) + 20*math.log10(fc) - 147.56 + LOS_PL
        else:
            h2 = 20*math.log10(Dt2) + 20*math.log10(fc) - 147.56 + NLOS_PL
        if self.L3 == 1:
                	h3 = 20*math.log10(Dt3) + 20*math.log10(fc) - 147.56 + LOS_PL
        else:
                	h3 = 20*math.log10(Dt3) + 20*math.log10(fc) - 147.56 + NLOS_PL
        if self.L4 == 1:
                	h4 = 20*math.log10(Dt4) + 20*math.log10(fc) - 147.56 + LOS_PL
        else:
               	h4 = 20*math.log10(Dt4) + 20*math.log10(fc) - 147.56 + NLOS_PL
            
        h1 = -h1
        h2 = -h2
        h3 = -h3
        h4 = -h4

        h1 = 10 ** (h1/10)
        h2 = 10 ** (h2/10)
        h3 = 10 ** (h3/10)
        h4 = 10 ** (h4/10)
        
        a1 =  self.UAV.a1
        a2 =  1 - a1
        a3 =  self.UAV.a3
        a4 =  1 - a3
        
        observation = [ob1[0]] + [ob1[1]] + [ob2[0]] + [ob2[1]]+ [ob3[0]] + [ob3[1]]+ [ob4[0]] + [ob4[1]] + [a1] + [a2] + [a3] + [a4] + [h1] + [h2] + [h3] + [h4] + [H]
        
        self.episode_step = 0

        return observation

    def step(self, action):
        
        done= False
        
        P = 1 # Transmitted power 30dbm (i.e. 1w)
        W = 5e7 # Bandwidth 50MHz
        fc = 2e9 # Carrier frequency = 2GHz
        N0 = 10e-17 # W/Hz
        LOS_PL = 1
        NLOS_PL = 20
        N = N0 * W
        H = self.UAV.H # antenna Height
        c = 3e8
        lamda = c/fc          
        H = self.UAV.H # antenna Height
        
        self.episode_step += 1
        
        ob1 = self.UAV-self.USER1
        ob2 = self.UAV-self.USER2
        ob3 = self.UAV-self.USER3
        ob4 = self.UAV-self.USER4
        
        D1 =  np.sum(np.sqrt([(10*ob1[0])**2, (10*ob1[1])**2]))
        D2 = np.sum(np.sqrt([(10*ob2[0])**2, (10*ob2[1])**2]))
        D3 = np.sum(np.sqrt([(10*ob3[0])**2, (10*ob3[1])**2]))
        D4 = np.sum(np.sqrt([(10*ob4[0])**2, (10*ob4[1])**2]))
        
        self.L1 = LineOfSight_Check(D1,H)
        self.L2 = LineOfSight_Check(D2,H)
        self.L3 = LineOfSight_Check(D3,H)
        self.L4 = LineOfSight_Check(D4,H)
                  
        Dt1 = np.sum(np.sqrt([ (10*ob1[0])**2, (10*ob1[1])**2, H**2  ]))
        Dt2 = np.sum(np.sqrt([ (10*ob2[0])**2, (10*ob2[1])**2, H**2  ]))
        Dt3 = np.sum(np.sqrt([ (10*ob3[0])**2, (10*ob3[1])**2, H**2  ]))
        Dt4 = np.sum(np.sqrt([ (10*ob4[0])**2, (10*ob4[1])**2, H**2  ]))
        
        
        if self.L1 == 1:
            h1 = 20*math.log10(Dt1) + 20*math.log10(fc) - 147.56 + LOS_PL
        else:
            h1 = 20*math.log10(Dt1) + 20*math.log10(fc) - 147.56 + NLOS_PL
        if self.L2 == 1:
            h2 = 20*math.log10(Dt2) + 20*math.log10(fc) - 147.56 + LOS_PL
        else:
            h2 = 20*math.log10(Dt2) + 20*math.log10(fc) - 147.56 + NLOS_PL
        if self.L3 == 1:
                	h3 = 20*math.log10(Dt3) + 20*math.log10(fc) - 147.56 + LOS_PL
        else:
                	h3 = 20*math.log10(Dt3) + 20*math.log10(fc) - 147.56 + NLOS_PL
        if self.L4 == 1:
                	h4 = 20*math.log10(Dt4) + 20*math.log10(fc) - 147.56 + LOS_PL
        else:
               	h4 = 20*math.log10(Dt4) + 20*math.log10(fc) - 147.56 + NLOS_PL
            
        h1 = -h1
        h2 = -h2
        h3 = -h3
        h4 = -h4

        h1 = 10 ** (h1/10)
        h2 = 10 ** (h2/10)
        h3 = 10 ** (h3/10)
        h4 = 10 ** (h4/10)
        
        self.UAV.action(action)
        
        a1 =  self.UAV.a1
        a2 =  1 - a1
        a3 =  self.UAV.a3
        a4 =  1 - a3
        

        reward = 0
        
        if h1 >= h2:
            
            SUM1 = math.log2(1 + h1 * a1 * P/N)
            SUM2 = math.log2(1 + a2 * h2 * P / (a1 * h2 * P + N) )
            if a2 > a1:
              reward += 10

        else: 
        
            SUM1 = math.log2(1 + a1 * h1 * P / (a2 * h1 * P + N) )
            SUM2 =  math.log2(1 + h2 * a2 * P/N)
            if a1 > a2:
                reward += 10
                
        if h3 >= h4:
            SUM3 = math.log2(1 + h3 * a3 * P/N)
            SUM4 = math.log2(1 + a4 * h4 * P / (a3 * h4 * P + N) ) 
            if a4 > a3:
              reward += 10
            
        else: 
            
            SUM3 = math.log2(1 + a3 * h3 * P / (a4 * h3 * P + N) )
            SUM4 = math.log2(1 + h4 * a4 * P/N)
            if a3 > a4:
                reward += 10
                
        reward_3 = (SUM1 + SUM2 + SUM3 + SUM4)**2 / (4 * (SUM1**2 + SUM2**2 + SUM3**2 + SUM4**2))
        self.Fairness.append(reward_3)
        

        if reward_3 >= 0.6 and reward_3 <= 0.65:
          reward += 10
        reward_6 = 1e7 * (h1+h2+h3+h4) 
        reward +=  (SUM1 + SUM2 + SUM3 + SUM4)  + reward_3  + reward_6
        
        self.SUM1.append(SUM1)
        self.SUM2.append(SUM2)
        self.SUM3.append(SUM3)
        self.SUM4.append(SUM4)

        new_observation_m =  ([ob1[0]] + [ob1[1]] + [ob2[0]] + [ob2[1]]+ [ob3[0]] + [ob3[1]] + [ob4[0]] + [ob4[1]] + [a1] + [a2] + [a3] + [a4] + [h1] + [h2] + [h3] + [h4] + [H] )
        new_observation =  new_observation_m  
        
        if self.episode_step >= 300:
            
            SUM11.append(Average(self.SUM1)) 
            SUM22.append(Average(self.SUM2)) 
            SUM33.append(Average(self.SUM3)) 
            SUM44.append(Average(self.SUM4))
            Fairnessl.append(Average(self.Fairness))

            calc(SUM11,SUM22,SUM33,SUM44,Fairnessl, 100, iteration)
            
            done = True
                          
        return new_observation,new_observation_m, reward, done

batch_size = 128
gamma = 0.999
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
target_update = 10
memory_size = 15000
lr = 0.001
num_episodes = 1000
num_of_actions = 32
num_of_arg_per_state = 17
ITERATIONS = 10

NUM_OF_LAYERS = [1]
NUM_OF_NEURONS_PER_LAYER = [128]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for num_of_layers in NUM_OF_LAYERS:
  for num_of_neurons_per_layer in NUM_OF_NEURONS_PER_LAYER:

	em = BlobEnv()
	strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
	agent = Agent(strategy, num_of_actions, device)
	memory = ReplayMemory(memory_size)
	policy_net = DQN(num_of_arg_per_state, num_of_layers, 
	num_of_neurons_per_layer, num_of_actions).to(device)
	target_net = DQN(num_of_arg_per_state, num_of_layers, 
    num_of_neurons_per_layer, num_of_actions).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	target_net.eval()
	optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

	SUM11 = []
	SUM22 = []
	SUM33 = []
	SUM44 = []
	Fairnessl = []

	for episode in range(num_episodes):
	  state = torch.tensor([em.reset()], dtype=torch.float32).to(device)

	  for timestep in count():   
        action = agent.select_action(state, policy_net)
        next_state, next_state_m, reward, done = em.step(action.item())
        reward = torch.tensor([reward], dtype=torch.int64).to(device)
        next_state = torch.tensor([next_state], dtype=torch.float32)
        .to(device)
        next_state_m = torch.tensor([next_state_m], dtype=torch.float32)
        .to(device)        
        memory.push(Experience(state, action, next_state_m, reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, rewards, next_states = 
            	extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions)
            next_q_values = QValues.get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            loss = F.mse_loss(current_q_values, 
            			target_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:         
            break

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

