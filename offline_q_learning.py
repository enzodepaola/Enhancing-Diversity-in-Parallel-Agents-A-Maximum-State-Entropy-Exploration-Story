import numpy as np
import random
from replay_buffer import make_replay_loader
from tqdm import tqdm

class QLearningWithReplay:
    def __init__(self, state_size, action_size, replay_buffer, external_target=None,alpha=0.1, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = replay_buffer
        self.external_target = external_target
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = np.zeros((state_size, action_size))
        self.STATE = 0
        self.ACTION = 1
        self.REWARD = 2
        self.NEXT_STATE = 3
        self.TRUNCATED = 4
        self.DONE = 5
    
    def set_new_target(self,target):
        self.target = target
        return
    
    def decode_minibatch(self,minibatch):
        state = minibatch[self.STATE].astype(int)
        action = minibatch[self.ACTION].astype(int)
        reward = minibatch[self.REWARD]
        next_state = minibatch[self.NEXT_STATE].astype(int)
        done = minibatch[self.DONE]
        truncated = minibatch[self.TRUNCATED]
        return state, action, reward, next_state, done, truncated
    
    
    def update_q_values(self):
        minibatch = np.array(next(self.replay_buffer))
        state,action,reward,next_state,truncated,done = self.decode_minibatch(minibatch)

        for s, a, r, next_s,t,d in zip(state,action,reward,next_state,truncated,done):
            if self.external_target:
                d = (next_s == self.target) 
                if d:
                   target = 1.0
                else:
                    target = 0.0
            else:
                target = r
            if not d:
                target += self.gamma * np.max(self.q_table[next_s])
            
            self.q_table[s, a] += self.alpha * (target - self.q_table[s, a])
    
    def train_from_replay(self, episodes):
        for episode in tqdm(range(episodes)):
            self.update_q_values()
    
    def evaluate_policy(self,env, test_episodes=10):
        total_rewards = []
        ob = []
        actions = []
        for _ in range(test_episodes):
            state,_ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            while not done and not truncated:
                ob.append(state)
                action = np.argmax(self.q_table[state])  # Always exploit
                actions.append(action)
                state, reward, truncated,done,info = env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)

        return np.mean(total_rewards)#,ob,actions
    
    def reset(self):
        self.q_table = np.zeros((self.state_size, self.action_size))
        return