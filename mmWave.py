
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