#Import used libraries 

%matplotlib inline
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from collections import namedtuple
from itertools import count
from PIL import Image
import cv2
import time
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from IPython.display import clear_output, display
import os
import pickle

#Import torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#Print "Starting" in the screen to know that everything has been imported correctly
out = display(IPython.display.Pretty('Starting'), display_id=True)

#Set seed to the randomization process of torch and NumPy libraries to ensure the same performance each run
torch.manual_seed(0)
np.random.seed(0)

#Neural Network class
class DQN(nn.Module):
    def __init__(self, NUMBER_OF_ARGUMENTS_PER_STATE):

        #Override torch library
        super().__init__(),

        #Build two fully connected layers with 128 output features
        self.fc1 = nn.Linear(in_features=NUMBER_OF_ARGUMENTS_PER_STATE, out_features=128) 
        self.fc2 = nn.Linear(in_features=128, out_features=128) 

        #Build two output independent layers, for value and advantage function approximation
        self.out_v = nn.Linear(in_features=128, out_features=1)
        self.out_a = nn.Linear(in_features=128, out_features=32)

    #Forward input through neural network layers
    def forward(self, t):

        #Flatten the input to make it 1D
        t = t.flatten(start_dim=1)

        #Input the flatten layer output to the fully connected layers
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))

        #Get the value function output from the second layer output
        v = self.out_v(t) #Value stream

        #Get the advantage function output from the second layer output
        a = self.out_a(t) #Advantage stream

        #Perform the Q function approximation based on value and advantage functions outputs
        q = v + a - a.mean()

        #Return the Q value
        return q

#Initiating a tuple of experiences
Experience = namedtuple(
            'Experience',
            ('state', 'action', 'next_state', 'reward')
                        )
#Initiating a ReplyMemory with a size of "capacity"
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):

        #Test the memory length to see whether we should append an experience at the end of the list or at the beginning
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    #Sample from reply memory to train the network
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    #Test if we can get a sample from the reply memory
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

#Exploitation and Exploration strategy
class EpsilonGreedyStrategy():

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
                            math.exp(-1. * current_step / self.decay)

#Build an agent class (UAV class)
class Agent():
    def __init__(self, strategy, num_actions, device):

        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    #Select an action from the state given by the environment
    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        #Check if the exploration rate is larger than a random number from 0 to 1, if so, choose a random action
        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device) #Explore  

        #Else choose the action via the neural network  
        else:
            with torch.no_grad(): #This means do not update the weights and biases of the neural network after this forward process 
                return policy_net(state).argmax(dim=1).to(self.device) # exploit


class QValues():

    #Check if CUDA is installed to train the network via the GPU, otherwise train via the CPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Return the current Q values from the neural network output after inputting the states
    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    #Return the target Q values from the neural network output after inputting the next states
    @staticmethod        
    def get_next(target_net, next_states):                 
        return target_net(next_states).max(dim=1)[0].detach()

#Plot the moving averages of the values we saved through training
def plot(values,h1,h2,h3,h4,a1,a2,a3,a4,SUM1,SUM2,SUM3,SUM4,Fairness,H,AVG2, Fairness2, moving_avg_period):

    #Get the moving average of the rewards
    moving_avg_rewards = get_moving_average(moving_avg_period, values)

    #Plot at the end of each training session
    if episode == 999:

        #AdaptSky fairness
        Fairness = [element * 100 for element in Fairness]
        moving_avg_fairness = get_moving_average(moving_avg_period, Fairness)

        #SoA fairness
        Fairness2 = [element * 100 for element in Fairness2]
        moving_avg_fairness2 = get_moving_average(moving_avg_period, Fairness2)

        #Channel conditions of cluster 1
        moving_avg_h1 = get_moving_average(moving_avg_period, h1)
        moving_avg_h2 = get_moving_average(moving_avg_period, h2)

        #Channel conditions of cluster 2
        moving_avg_h3 = get_moving_average(moving_avg_period, h3)
        moving_avg_h4 = get_moving_average(moving_avg_period, h4)

        #Power percentages of cluster 1
        moving_avg_a1 = get_moving_average(moving_avg_period, a1)
        moving_avg_a2 = get_moving_average(moving_avg_period, a2)
        moving_avg_a1 = [element * 100 for element in moving_avg_a1]
        moving_avg_a2 = [element * 100 for element in moving_avg_a2]

        #Power percentages of cluster 2
        moving_avg_a3 = get_moving_average(moving_avg_period, a3)
        moving_avg_a4 = get_moving_average(moving_avg_period, a4)
        moving_avg_a3 = [element * 100 for element in moving_avg_a3]
        moving_avg_a4 = [element * 100 for element in moving_avg_a4]

        #Calculate the moving average of cluster 1 spectral efficiencies and multiply it by 2000MHz to get the achievable sum-rate
        moving_avg_SUM1 = get_moving_average(moving_avg_period, SUM1)
        moving_avg_SUM2 = get_moving_average(moving_avg_period, SUM2)
        moving_avg_SUM1 = [element * 2000 for element in moving_avg_SUM1]
        moving_avg_SUM2 = [element * 2000 for element in moving_avg_SUM2]

        #Calculate the moving average of cluster 2 spectral efficiencies and multiply it by 2000MHz to get the achievable sum-rate
        moving_avg_SUM3 = get_moving_average(moving_avg_period, SUM3)
        moving_avg_SUM4 = get_moving_average(moving_avg_period, SUM4)
        moving_avg_SUM3 = [element * 2000 for element in moving_avg_SUM3]
        moving_avg_SUM4 = [element * 2000 for element in moving_avg_SUM4]

        #Calculate the average sum rate of all clusters
        SUM = np.add(moving_avg_SUM1,moving_avg_SUM2)
        SUM = np.add(SUM,moving_avg_SUM3)
        SUM = np.add(SUM,moving_avg_SUM4)

        #Calculate SoA UAV sum rate
        avg2 = get_moving_average(moving_avg_period, AVG2)
        avg2 = [element * 2000 for element in avg2]
      
        #Calculate the UAV height
        moving_avg_Height = get_moving_average(moving_avg_period, H)
        
        #Print values
        print(f"r = {r}, iteration = {i}")
        print("Sum Rate Moving Average:",round(SUM[-1],2)/1000, " Gbps, Total SE = ", round(SUM[-1]/(2000),2), " bps/Hz")
        print("SE1 = ", round(moving_avg_SUM1[-1]/2000, 2) , "SE2 = ", round(moving_avg_SUM2[-1]/2000, 2), "SE3 = ",round(moving_avg_SUM3[-1]/2000, 2), "SE4 = ", round(moving_avg_SUM4[-1]/2000, 2), "\n")

    else:

        #Print the current rewards to troubleshot any inefficient training process
        out.update(IPython.display.Pretty(f'r = {r}, iteration = {i} \nMoving Average Rewards: {round(moving_avg_rewards[-1], 2)}, Episode: {len(SUM1)}.'))

#Calculate the moving average of any values given the period
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
    
#Check the mmWave line-of-sight probability 
def mmLineOfSight_Check(D,H):
    L = 1
    return L
    C = 9.6117 #Urban LOS probability parameter 
    Y = 0.1581 #Urban LOS probability parameter
    RAND = random.uniform(0,1)
    teta = math.asin(H/D) * 180/math.pi
    p1 = 1 / ( 1 + (C * math.exp( -Y * (teta - C ) ) ) )
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
    
#Calculate the average of any list
def Average(lst): 
    return sum(lst) / len(lst) 

#Extract values from the experiences created above
def extract_tensors(experiences):

    #Convert a batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)


class UAV():
    def __init__(self, size, USER1=False, USER2=False, USER3=False, USER4=False):

        #Locating the initial locations of each user
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
        return f"UAV({self.x}, {self.y})"

#Get the location differences between two objects
    def __sub__(self, other):
        return [(self.x-other.x), (self.y-other.y)]

#Choose an action from the 32 choices
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
        
#Move the UAV and/or the users
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


class mmWave_Env():

    SIZE = 100 #The environment size
    MOVE_PENALTY = 1 #Didn't use it but it was meant to make the UAV loss reward whenever it moves
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  #RGB troubleshoot screen
    UAV_N = 1  #UAV key in the colors dictionary
    USER_N = 2  #USERs key in the colors dictionary
    UAV2_N = 4  # SoA UAV key in the colors dictionary

    #The colors dictionary
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255),
         4: (175, 0, 255)}

#Reset the environment
    def reset(self):
        p = 20
        P = 10**((p-30)/10) #Transmitted power 20dbm (i.e. .1w)
        N_uav = 8 #Number of Tx antennas
        N_ue = 8 #Number of Rx antennas
        G = N_uav * N_ue #MIMO Gain
        P *= G
        W = 2e9 #Bandwidth of 2 GHz
        fc = 28e9 #Carrier frequency of 28 GHz
        NF = 10**(5/10) #5dB noise figure 
        TN = 10**(-114/10) #-84dBm thermal noise
        N = NF * TN

        C_LOS = 10**(-6.4) #mmWave LoS Urban parameter 
        a_LOS = 2 #mmWave LoS Urban parameter
        C_NLOS = 10**(-7.2) #mmWave NLoS Urban parameter
        a_NLOS = 2.92 #mmWave NLoS Urban parameter
    
        #Initiating environment objects
        self.UAV = UAV(self.SIZE)
        self.UAV2 = UAV(self.SIZE)

        self.USER1 = UAV(self.SIZE, True, False, False, False)
        self.USER2 = UAV(self.SIZE, False, True, False, False)
        self.USER3 = UAV(self.SIZE, False, False, True, False)
        self.USER4 = UAV(self.SIZE, False, False, False, True)

        self.UAV2.x = int((self.USER1.x +self.USER2.x + self.USER3.x + self.USER4.x)/4)
        self.UAV2.y = int((self.USER1.y +self.USER2.y + self.USER3.y + self.USER4.y)/4)

        #Initiating lists to store eact time step values
        self.h1 = []
        self.h2 = []
        self.h3 = []
        self.h4 = []
        self.a1 = []
        self.a2 = []
        self.a3 = []
        self.a4 = []
        self.SUM1 = []
        self.SUM2 = []
        self.SUM3 = []
        self.SUM4 = []
        self.Fairness = []
        self.Hl = []
        self.NLOS = []
        self.NOMA = []
        self.reward1 = []
        self.reward2 = []
        self.reward3 = []
        self.reward4 = []
        self.reward5 = []
        self.reward6 = []
        
        #Initiate power percentages
        self.UAV.a1 = 0.5
        self.UAV.a2 = 0.5
        self.UAV.a3 = 0.5
        self.UAV.a4 = 0.5

        #Initiate UAVs height
        self.UAV.H = 50
        H2 = 50
        
        #Get the difference between the UAV and each user
        ob1 = self.UAV-self.USER1
        ob2 = self.UAV-self.USER2
        ob3 = self.UAV-self.USER3
        ob4 = self.UAV-self.USER4

        #Get the perpendicular distance between the UAV and each user
        Dt1 = np.sum(np.sqrt([ (ob1[0])**2, (ob1[1])**2, H**2  ]))
        Dt2 = np.sum(np.sqrt([ (ob2[0])**2, (ob2[1])**2, H**2  ]))
        Dt3 = np.sum(np.sqrt([ (ob3[0])**2, (ob3[1])**2, H**2  ]))
        Dt4 = np.sum(np.sqrt([ (ob4[0])**2, (ob4[1])**2, H**2  ]))
        
        #Line-of-sight check
        self.L1 = mmLineOfSight_Check(Dt1,H)
        self.L2 = mmLineOfSight_Check(Dt2,H)
        self.L3 = mmLineOfSight_Check(Dt3,H)
        self.L4 = mmLineOfSight_Check(Dt4,H)
        
        #Calculate the path loss for each user
        if self.L1 == 1:
            h1 = C_LOS * Dt1**(-a_LOS)
        else:
            h1 = C_NLOS * Dt1**(-a_NLOS)

        if self.L2 == 1:
            h2 = C_LOS * Dt2**(-a_LOS)
        else:
            h2 = C_NLOS * Dt2**(-a_NLOS)
        if self.L3 == 1:
            h3 = C_LOS * Dt3**(-a_LOS)
        else:
            h3 = C_NLOS * Dt3**(-a_NLOS)
        if self.L4 == 1:
            h4 = C_LOS * Dt4**(-a_LOS)
        else:
            h4 = C_NLOS * Dt4**(-a_NLOS)

        #put each state in an observations list
        observation = [ob1[0]] + [ob1[1]] + [ob2[0]] + [ob2[1]]+ [ob3[0]] + [ob3[1]]+ [ob4[0]] + [ob4[1]] + [a1] + [a2] + [a3] + [a4] + [h1] + [h2] + [h3] + [h4] + [H]
        
        #Initiate episodes count
        self.episode_step = 0

        return observation

    #Calculate the states' values and take an action
    def step(self, action):
        done= False
        p = 20
        P = 10**((p-30)/10) #Transmitted power 20dbm (i.e. .1w)
        N_uav = 8
        N_ue = 8
        G = N_uav * N_ue
        P *= G
        W = 2e9 #Bandwidth = 2 GHz
        fc = 28e9 # Carrier frequency = 28 GHz
        NF = 10**(5/10) #5dB noise figure 
        TN = 10**(-114/10) #-84dBm thermal noise
        N = NF * TN
        C_LOS = 10**(-6.4)
        a_LOS = 2
        C_NLOS = 10**(-7.2) 
        a_NLOS = 2.92        
        H = self.UAV.H #Current UAV's antenna height
        
        #There are some redundancies in this part that can easily be managed by a function
        self.episode_step += 1
        
        ob1 = self.UAV-self.USER1
        ob2 = self.UAV-self.USER2
        ob3 = self.UAV-self.USER3
        ob4 = self.UAV-self.USER4
                  
        H = self.UAV.H
        Dt1 = np.sum(np.sqrt([ (ob1[0])**2, (ob1[1])**2, H**2  ]))
        Dt2 = np.sum(np.sqrt([ (ob2[0])**2, (ob2[1])**2, H**2  ]))
        Dt3 = np.sum(np.sqrt([ (ob3[0])**2, (ob3[1])**2, H**2  ]))
        Dt4 = np.sum(np.sqrt([ (ob4[0])**2, (ob4[1])**2, H**2  ]))
        
        self.L1 = mmLineOfSight_Check(Dt1,H)
        self.L2 = mmLineOfSight_Check(Dt2,H)
        self.L3 = mmLineOfSight_Check(Dt3,H)
        self.L4 = mmLineOfSight_Check(Dt4,H)
        
        if self.L1 == 1:
            h1 = C_LOS * Dt1**(-a_LOS)
            self.NLOS.append(0)
        else:
            h1 = C_NLOS * Dt1**(-a_NLOS)
            self.NLOS.append(1)
        if self.L2 == 1:
            h2 = C_LOS * Dt2**(-a_LOS)
            self.NLOS.append(0)
        else:
            h2 = C_NLOS * Dt2**(-a_NLOS)
            self.NLOS.append(1)
        if self.L3 == 1:
            h3 = C_LOS * Dt3**(-a_LOS)
            self.NLOS.append(0)
        else:
            h3 = C_NLOS * Dt3**(-a_NLOS)
            self.NLOS.append(1)
        if self.L4 == 1:
            h4 = C_LOS * Dt4**(-a_LOS)
            self.NLOS.append(0)
        else:
            h4 = C_NLOS * Dt4**(-a_NLOS)
            self.NLOS.append(1)
        
        #Take an action
        self.UAV.action(action)
        
        #Receive the new power percentages 
        a1 =  self.UAV.a1
        a2 =  1 - a1
        a3 =  self.UAV.a3
        a4 =  1 - a3
        
        #Append the new states
        self.h1.append(h1)
        self.h2.append(h2)
        self.h3.append(h3)
        self.h4.append(h4)
        self.a1.append(a1)
        self.a2.append(a2)
        self.a3.append(a3)
        self.a4.append(a4)
        self.Hl.append(H)

        #Reset reward values
        reward = 0
        reward_1 = 0
        reward_2 = 0
        reward_4 = 0
        reward_5 = 0
        reward_6 = 0
        
        #SIC check and spectral efficiency calculations of cluster 1 
        if h1 >= h2:
            SUM1 = math.log2(1 + h1 * a1 * P/N)
            SUM2 = math.log2(1 + a2 * h2 * P / (a1 * h2 * P + N) )
            reward_1 += SUM1
            reward_2 += SUM2
        else: 
            SUM1 = math.log2(1 + a1 * h1 * P / (a2 * h1 * P + N) )
            SUM2 =  math.log2(1 + h2 * a2 * P/N)
            reward_1 += SUM2
            reward_2 += SUM1
            
        #SIC check and spectral efficiency calculations of cluster 2
        if h3 >= h4:
            SUM3 = math.log2(1 + h3 * a3 * P/N)
            SUM4 = math.log2(1 + a4 * h4 * P / (a3 * h4 * P + N) ) 
            reward_4 += SUM3
            reward_5 += SUM4
        else: 
            SUM3 = math.log2(1 + a3 * h3 * P / (a4 * h3 * P + N) )
            SUM4 = math.log2(1 + h4 * a4 * P/N)
            reward_4 += SUM4
            reward_5 += SUM3  
        
        #Fairness calculations
        reward_3 = (SUM1 + SUM2 + SUM3 + SUM4)**2 / (4 * (SUM1**2 + SUM2**2 + SUM3**2 + SUM4**2))
        self.Fairness.append(reward_3)


        self.SUM1.append(SUM1)
        self.SUM2.append(SUM2)
        self.SUM3.append(SUM3)
        self.SUM4.append(SUM4)

        #Check if each user spectral efficiency is above the threshold
        if SUM1 >= r:
            reward += 100
        if SUM2 >= r:
            reward += 100
        if SUM3 >= r:
            reward += 100
        if SUM4 >= r:
            reward += 100

        #A motivation to force the UAV to achieve our objective
        if reward >= 400:
          SUM1*=10
          SUM2*=10
          SUM3*=10
          SUM4*=10

        #Reward function weights
        w1 = 1
        w2 = 0
        w3 = 0

        reward_3 *= w3
        reward_6 += 2e10 * (h1+h2+h3+h4) * w2 

        reward +=   w1* (SUM1 + SUM2 + SUM3 + SUM4)  + reward_3  + reward_6
        

        self.reward1.append(reward_1)
        self.reward2.append(reward_2)
        self.reward3.append(reward_3)
        self.reward4.append(reward_4)
        self.reward5.append(reward_5)
        self.reward6.append(reward_6)


        new_observation_m =  ([ob1[0]] + [ob1[1]] + [ob2[0]] + [ob2[1]]+ [ob3[0]] + [ob3[1]] + [ob4[0]] + [ob4[1]] + [a1] + [a2] + [a3] + [a4] + [h1] + [h2] + [h3] + [h4] + [H])

        #new_obervation is used in moving users only, otherwise new_observation =  new_observation_m 
        new_observation =  new_observation_m  
        
        #At the end of each episode calculate SoA values
        if self.episode_step >= 300:

            ob21 = self.UAV2-self.USER1
            ob22 = self.UAV2-self.USER2
            ob23 = self.UAV2-self.USER3
            ob24 = self.UAV2-self.USER4
            H2 = 50

            Dt21 = np.sum(np.sqrt([ (ob21[0])**2, (ob21[1])**2, H2**2  ]))
            Dt22 = np.sum(np.sqrt([ (ob22[0])**2, (ob22[1])**2, H2**2  ]))
            Dt23 = np.sum(np.sqrt([ (ob23[0])**2, (ob23[1])**2, H2**2  ]))
            Dt24 = np.sum(np.sqrt([ (ob24[0])**2, (ob24[1])**2, H2**2  ]))
        
            h221 = C_LOS * Dt21**(-a_LOS)
            h222 = C_LOS * Dt22**(-a_LOS)
            h223 = C_LOS * Dt23**(-a_LOS)
            h224 = C_LOS * Dt24**(-a_LOS)

            if h221 >= h222:
                a222 = ((2**r - 1)/2**r) * (1 + N/(P*h222))
                if a222 >= 1:
                  a222 = 1
                a221 = 1 - a222
                SUM221 = math.log2(1 + h221 * a221 * P/N)
                SUM222 = math.log2(1 + a222 * h222 * P / (a221 * h222 * P + N) )
            else: 
                a221 = ((2**r - 1)/2**r) * (1 + N/(P*h221))
                if a221 >= 1:
                  a221 = 1
                a222 = 1-a221
                SUM221 = math.log2(1 + a221 * h221 * P / (a222 * h221 * P + N) )
                SUM222 =  math.log2(1 + h222 * a222 * P/N)
            if h223 >= h224:

                a224 = ((2**r - 1)/2**r) * (1 + N/(P*h224))
                if a224 >= 1:
                  a224 = 1
                a223 = 1 - a224
                SUM223 = math.log2(1 + h223 * a223 * P/N)
                SUM224 = math.log2(1 + a224 * h224 * P / (a223 * h224 * P + N) ) 
            else: 
                a223 = ((2**r - 1)/2**r) * (1 + N/(P*h223))
                if a223 >= 1:
                  a223 = 1
                a224 = 1 - a223
                SUM223 = math.log2(1 + a223 * h223 * P / (a224 * h223 * P + N) )
                SUM224 = math.log2(1 + h224 * a224 * P/N)
                
            #Calculate SoA's sum rate and fairness
            average_sum_rate2 =  SUM221 + SUM222 + SUM223 + SUM224  
            Fairness222 = (SUM221 + SUM222 + SUM223 + SUM224)**2 / (4 * (SUM221**2 + SUM222**2 + SUM223**2 + SUM224**2))

            #Take the average of the episode generated values
            h11.append(Average(self.h1))
            h22.append(Average(self.h2)) 
            h33.append(Average(self.h3)) 
            h44.append(Average(self.h4)) 
            a11.append(Average(self.a1)) 
            a22.append(Average(self.a2)) 
            a33.append(Average(self.a3)) 
            a44.append(Average(self.a4)) 
            SUM11.append(Average(self.SUM1)) 
            SUM22.append(Average(self.SUM2)) 
            SUM33.append(Average(self.SUM3)) 
            SUM44.append(Average(self.SUM4))
            reward1.append(Average(self.reward1))
            reward2.append(Average(self.reward2))
            reward3.append(Average(self.reward3))
            reward4.append(Average(self.reward4))
            reward5.append(Average(self.reward5))
            reward6.append(Average(self.reward6))
            average_episode_reward = episode_reward/self.episode_step 
            Fairnessl.append(Average(self.Fairness))
            episode_rewards.append(average_episode_reward)
            episode_durations.append(timestep)
            Height.append(Average(self.Hl))
            AVG2.append(average_sum_rate2)
            Fairnessl_2.append(Fairness222)

            #End the episode
            done = True

            #Call the plot function            
            plot(episode_rewards,reward1,reward2,reward3,reward4,reward5,reward6,h11,h22,h33,h44,a11,a22,a33,a44,SUM11,SUM22,SUM33,SUM44,Fairnessl,Height,AVG2,Fairnessl_2, 100)
            
            #At the end of the training session, calculate and plot the last moving average value
            if episode >=999:
              average_h1 = 10 * math.log10(h11[-1])
              average_h2 = 10 * math.log10(h22[-1])
              average_h3 = 10 * math.log10(h33[-1])
              average_h4 = 10 * math.log10(h44[-1])

              average_h21 = 10* math.log10(h221)
              average_h22 = 10* math.log10(h222)
              average_h23 = 10* math.log10(h223)
              average_h24 = 10* math.log10(h224)
            
              average_sum_rate = SUM11[-1] + SUM22[-1] + SUM33[-1] + SUM44[-1]
            

              print("\n                          UAV2                            ")
              print("Sum Rate:", round(2*average_sum_rate2, 2), "Gbps, Total SE = ", round(average_sum_rate2, 2), " EE = ", round(average_sum_rate2/(P),2))
              print("SE1: ",round(SUM221, 2),"Bits/s/Hz, SE2: ",round(SUM222, 2),"Bits/s/Hz, SE3: ",round(SUM223, 2),"Bits/s/Hz, SE4: ",round(SUM224, 2),"Bits/s/Hz")
            

                          
        return new_observation,new_observation_m, reward, done

    #Render the troubleshoot function
    def render(self):
        img = self.get_image()
        img = img.resize((500, 500)) # Resizing
        cv2.imshow("UAV Beta 1.0", np.array(img)) 
        cv2.waitKey(1)

    #Get the environment image
    def get_image(self):
        env = np.full((self.SIZE, self.SIZE, 3), 255, dtype=np.uint8)  #Start an RGB image
        env[self.USER1.x][self.USER1.y] = self.d[(self.L1+1)] #Set the pixel value to d[(self.L1+1)] color from the colors dictionary
        env[self.USER2.x][self.USER2.y] = self.d[(self.L2+1)] #Set the pixel value to d[(self.L2+1)] color from the colors dictionary
        env[self.USER3.x][self.USER3.y] = self.d[(self.L3+1)] #Set the pixel value to d[(self.L3+1)] color from the colors dictionary
        env[self.USER4.x][self.USER4.y] = self.d[(self.L4+1)] #Set the pixel value to d[(self.L4+1)] color from the colors dictionary
        env[self.UAV.x][self.UAV.y] = self.d[self.UAV_N] #Set the pixel value to d[self.UAV_N] color from the colors dictionary
        img = Image.fromarray(env, 'RGB') #Transform the array to an RGB image
        return img 

#Set DRL parameters
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
SHOW_PREVIEW = False
AGGREGATE_STATS_EVERY = 10
ITERATIONS = 5

#For threshold spectral efficiency 0 to 3.5bps/Hz
for r in np.arange(0, 3.5, 0.5):

    #Repeat training sessions "ITERATIONS" number of times to take the confidence interval afterward
    for i in range(ITERATIONS):

        #Check if CUDA is installed to train the network via the GPU, otherwise train via the CPU 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Initialize the environment
        em = mmWave_Env()

        #Initialize Epsilon Greedy Strategy with the parameters we set above
        strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

        #Initialize the agent with the parameters we set above
        agent = Agent(strategy, num_of_actions, device)

        #Initialize the replay memory
        memory = ReplayMemory(memory_size)

        #Initialize the policy network
        policy_net = DQN(num_of_arg_per_state).to(device)

        #Initialize the target network
        target_net = DQN(num_of_arg_per_state).to(device)

        #Copy policy net' parameters to target net
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        #Initialize Adam's optimizer and the learning rate to lr
        optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

        #Initialize the evaluation lists
        episode_durations = []
        episode_rewards = []
        episode_wins = []
        h11 = []
        h22 = []
        h33 = []
        h44 = []
        a11 = []
        a22 = []
        a33 = []
        a44 = []
        SUM11 = []
        SUM22 = []
        SUM33 = []
        SUM44 = []
        TOTAL_SUM = []
        Fairnessl = []
        Height = []
        reward1 = []
        reward2 = []
        reward3 = []
        reward4 = []
        reward5 = []
        reward6 = []
        AVG2 = []
        Fairnessl_2 = []

        #Start training session
        for episode in range(num_episodes):

            #Reset the environment
            state = torch.tensor([em.reset()], dtype=torch.float32).to(device)
            episode_reward = 0

            for timestep in count():   

                #Choose an action and store it in the action variable
                action = agent.select_action(state, policy_net)

                #Execute the action and receive the next_state and reward
                next_state, next_state_m, reward, done = em.step(action.item())

                #Add the rewards to the episode_reward variable
                episode_reward += reward

                #Change the reward, next_state, next_state_m to a torch tensor variable so the torch library can handle it
                reward = torch.tensor([reward], dtype=torch.int64).to(device)
                next_state = torch.tensor([next_state], dtype=torch.float32).to(device)
                next_state_m = torch.tensor([next_state_m], dtype=torch.float32).to(device) 

                #Store state, action, next_state_m, reward in the reply memory after converting it to an experience tuble
                memory.push(Experience(state, action, next_state_m, reward))
                state = next_state

                #Test if memory can provide a sample of size "batch_size"
                if memory.can_provide_sample(batch_size):

                    #Sample a random batch with the size of "batch_size"
                    experiences = memory.sample(batch_size)

                    #Extract values from Experiences tuble
                    states, actions, rewards, next_states = extract_tensors(experiences)

                    #Get the current and next Q value
                    current_q_values = QValues.get_current(policy_net, states, actions)
                    next_q_values = QValues.get_next(target_net, next_states)

                    #Calculate the target Q values
                    target_q_values = (next_q_values * gamma) + rewards

                    #Calculate the loss between current_q_values and target_q_values, and update the neural network parameters accordingly
                    loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                #Show the troubleshooting screen if SHOW_PREVIEW is true and the episode is "AGGREGATE_STATS_EVERY" multiples
                if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                    
                    #Call the render function
                    em.render()

                #Break, if the episode is finished
                if done:         
                    break

            #Clone the policy network parameters to the target network parameters if episode number is "target_update" multiples
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())