import pyautogui
    
import numpy as np
from grabscreen import grab_screen
from collections import namedtuple, deque
import cv2
import time

pyautogui.FAILSAFE = False

scores=[0,0,0,0]
OFFSET = 25
filename = "C:\\Users\\KOZLOV-PC\\Sync\\Sync\\rtpp.txt"
attente = 0

def getPos():
    x,y = pyautogui.position()
    return x,y

def left_up():
    x,y = getPos()
    #print("left_up()")
    if (x-OFFSET) >= 0 and (y-OFFSET) >=40:    
        pyautogui.moveTo(x - OFFSET, y - OFFSET)

def up():
    x,y = getPos()
    #print("up()")
    if (y-OFFSET) >=40:
        pyautogui.moveTo(None, y - OFFSET)
    
def right_up():
    x,y = getPos()
    #print("right_up()")
    if (x+OFFSET) <= 800 and (y-OFFSET) >=40:
        pyautogui.moveTo(x + OFFSET, y - OFFSET)
    
def left():
    x,y = getPos()
    #print("left()")
    if (x-OFFSET) >= 0:
        pyautogui.moveTo(x - OFFSET, None)
    
def middle():
    #print("middle()")
    pyautogui.moveTo(None, None)

def right():
    x,y = getPos()
    #print("right()")
    if (x+OFFSET) <= 800:
        pyautogui.moveTo(x + OFFSET, None)
    
def left_down():
    x,y = getPos()
    #print("left_down()")
    if (x-OFFSET) >= 0 and (y+OFFSET) <=640:
        pyautogui.moveTo(x - OFFSET, y + OFFSET)

def down():
    x,y = getPos()
    #print("down()")
    if (y+OFFSET) <=640:
        pyautogui.moveTo(None, y + OFFSET)  
    
def right_down():
    x,y = getPos()
    #print("right_down()")
    if (x+OFFSET) <= 800 and (y+OFFSET) <=640:
        pyautogui.moveTo(x + OFFSET, y + OFFSET)
    
def doAction(index):
    if index==0:
        left_up()
    elif index==1:
        up()
    elif index==2:
        right_up()
    elif index==3:
        left()
    elif index==4:
        middle()
    elif index==5:
        right()
    elif index==6:
        left_down()
    elif index==7:
        down()
    elif index==8:
        right_down()
    return index

def getReward():
    fini = False
    last_reward=0
    file = open(filename, "r")
    try:
        lines = file.readlines()
        line = lines[0].split(";")
        line[3] = line[3].split("\n")[0]
        try:
            if int(line[0]) != scores[0]:
                print("300")
                attente = 0
                last_reward = 10
            elif int(line[1]) != scores[1]:
                print("100")
                last_reward = 5
            elif int(line[2]) != scores[2]:
                print("50")
                attente = 0
                last_reward = 2.5
            elif int(line[3]) != scores[3]:
                print("Miss")
                last_reward = -3
            else:
                last_reward = -0.05
                x,y = getPos()
                if x < 50:
                    last_reward = -0.33
                elif x > 750:
                    last_reward = -0.33
                elif y > 550:
                    last_reward = -0.33
                elif y < 50:
                    last_reward = -0.33
            scores[0] = int(line[0])
            scores[1] = int(line[1])
            scores[2] = int(line[2])
            scores[3] = int(line[3])
        except:
            print("Erreur")
            pass
    except:
        print("Map fini")
        fini = True
        last_reward = 0
        pass
    return last_reward, fini


# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])

# Making the AI progress on several (n_step) steps

class NStepProgress:
    
    def __init__(self, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.n_step = n_step
    
    def __iter__(self):
        indexA = 0
        state = grab_screen(region=(0,40,800,620)) #Récupération de l'image en 3D [601,801,3]
        state = cv2.resize(state,(100,75))
        state = state.reshape(1,75,100)
        history = deque()
        reward = 0.0
        last_time = time.time()
        while True:
            action = self.ai(np.array([state]))[0][0]
            doAction(action)       
            next_state = grab_screen(region=(0,40,800,620)) #Récupération de l'image en 3D [601,801,3]
            next_state = cv2.resize(next_state,(100,75))
            #print(next_state)
            cv2.imshow('window',next_state)
            next_state = next_state.reshape(1,75,100)
            r, is_done = getReward()
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
            indexA = indexA + 1
            #print('Frame took {} seconds'.format(time.time()-last_time))
            #last_time = time.time()
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            if indexA % 30 == 0:
                indexA = 0
                is_done = True
            if is_done:
                print("Terminé, reward : ", reward)
                self.rewards.append(reward)
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                reward = 0.0           
                state = grab_screen(region=(0,40,800,620)) #Récupération de l'image en 3D [601,801,3]
                state = cv2.resize(state,(100,75))
                state = state.reshape(1,75,100)
                history.clear()
    
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps

# Implementing Experience Replay

class ReplayMemory:
    
    def __init__(self, n_steps, capacity = 10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()

    def sample_batch(self, batch_size): # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            entry = next(self.n_steps_iter) # 10 consecutive steps
            self.buffer.append(entry) # we put 200 for the current episode
            samples -= 1
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()
