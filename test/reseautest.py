import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pyautogui

# Importing the packages for OpenAI and Doom
import gym
from gym import wrappers

# Importing the other Python files
import experience_replay
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() == False:
    print("Cuda device not found, exit.")
    exit()

# Part 1 - Building the AI
# Making the brain
for i in range(0,1):
    print(i)
    time.sleep(1)

class CNN(nn.Module):
    
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 1465, kernel_size = 8)
        self.convolution2 = nn.Conv2d(in_channels = 1465, out_channels = 256, kernel_size = 4)
        self.convolution3 = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3)
        self.fc1 = nn.Linear(in_features = self.count_neurons((1, 45, 65)), out_features = 50)
        self.fc2 = nn.Linear(in_features = 50, out_features = number_actions)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = x.cuda()
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.cpu()


# Making the body

class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)   
        actions = probs.multinomial(num_samples=1)
        return actions

# Making the AI

class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        output = self.brain(input)
        actions = self.body(output)
        return actions.data.numpy()

# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment
number_actions = 2925

# Building an AI
#cnn = CNN(2925).to(device)
cnn = torch.load("D:\\envPython\\OsuIA\\training\\brain-test-102.ty").to(device)
cnn.eval()
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 20000)
    
# Implementing Eligibility Trace
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making the moving average on 100 steps
class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    def average(self):
        return np.mean(self.list_of_rewards)
ma = MA(100)

# Training the AI
loss = nn.MSELoss()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epochs = 0
try:
    while(True):
        last_reward, fini = experience_replay.getReward()
        if fini == False:
            nb_epochs = nb_epochs + 1
            print("--- Start memory ---")
            memory.run_steps(1)
            print("--- End memory ---")
            print("--- Depart ---")
            for batch in memory.sample_batch(128):
                inputs, targets = eligibility_trace(batch)
                inputs, targets = Variable(inputs), Variable(targets)
                predictions = cnn(inputs)
                loss_error = loss(predictions, targets)
                optimizer.zero_grad()
                loss_error.backward()
                optimizer.step()
            rewards_steps = n_steps.rewards_steps()
            ma.add(rewards_steps)
            avg_reward = ma.average()
            print("Epoch: %s, Average Reward: %s" % (str(nb_epochs), str(avg_reward)))
            print("Saving brain...")
            torch.save(cnn, "D:\\envPython\\OsuIA\\training\\brain-test-2-{}.ty".format(nb_epochs))
            print("Successful")
            print("---- FIN -----")
        else:
            print("Waiting for map...")
            pyautogui.moveTo(400,300)
            time.sleep(1)
            pyautogui.keyDown('f2')
            time.sleep(0.1)
            pyautogui.keyUp('f2')
            time.sleep(2)
            pyautogui.keyDown('enter')
            time.sleep(0.1)
            pyautogui.keyUp('enter')
            time.sleep(2)
            pyautogui.keyDown('space')
            time.sleep(0.1)
            pyautogui.keyUp('space')
except KeyboardInterrupt:
    print("Loop^interrupt, closing app.")

