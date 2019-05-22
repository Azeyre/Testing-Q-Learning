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
mods_path = "C:\\Program Files (x86)\\StreamCompanion\\Files\\np_list.txt"
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
    x = index % 65
    y = int((index - x) / 65)
    mult = 800 / 65
    pyautogui.moveTo(int(x * mult),int(y * mult))
    return index

def getReward():
    fini = True
    last_reward=0
    file = open(filename, "r")
    mods_file = open(mods_path, "r")
    try:
        lines_mods = mods_file.readlines()
        if len(lines_mods[0]) > 0:
            print("Esc")
            pyautogui.keyDown('esc')
            time.sleep(0.1)
            pyautogui.keyUp('esc')
            return 0, True
    except:
        pass
    try:
        lines = file.readlines()
        line = lines[0].split(";")
        line[3] = line[3].split("\n")[0]
        try:
            if int(line[0]) != scores[0]:
                #print("300")
                attente = 0
                last_reward = 10
            elif int(line[1]) != scores[1]:
                #print("100")
                last_reward = 5
            elif int(line[2]) != scores[2]:
                #print("50")
                attente = 0
                last_reward = 2.5
            elif int(line[3]) != scores[3]:
                #print("Miss")
                last_reward = -5
            scores[0] = int(line[0])
            scores[1] = int(line[1])
            scores[2] = int(line[2])
            scores[3] = int(line[3])
            fini = False
        except:
            pass
    except:
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
        state = cv2.resize(state,(65,45))
        state = state.reshape(1,45,65)
        history = deque()
        reward = 0.0
        last_time = time.time()
        while True:
            action = self.ai(np.array([state]))[0][0]
            doAction(action)       
            next_state = grab_screen(region=(0,40,800,620)) #Récupération de l'image en 3D [601,801,3]
            next_state = cv2.resize(next_state,(65,45))
            #print(next_state)
            cv2.imshow('window',next_state)
            next_state = next_state.reshape(1,45,65)
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
            if indexA % 100 == 0:
                indexA = 0
                is_done = True
            if is_done:
                #print("Terminé, reward : ", reward)
                self.rewards.append(reward)
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                reward = 0.0           
                state = grab_screen(region=(0,40,800,620)) #Récupération de l'image en 3D [601,801,3]
                state = cv2.resize(state,(65,45))
                state = state.reshape(1,45,65)
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
        #print("Sample batch 1")
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            #print("Sample batch 2")
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1
        #print("Sample batch 3")

    def run_steps(self, samples):
        while(True):
            entry = next(self.n_steps_iter) # 10 consecutive steps
            self.buffer.append(entry) # we put 200 for the current episode
            #print("Append")
            last_reward, fini = getReward()
            if fini == True:
                #print("Break")
                break
        print("Fin")
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()
