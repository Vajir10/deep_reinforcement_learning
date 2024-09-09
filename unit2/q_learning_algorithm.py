import numpy as np
import gymnasium as gym
import random
import imageio
import os
import tqdm

import pickle

from tqdm import tqdm

from unit2.config import FrozenLakeConfig, TaxiV3Config

class QLearning:
    def __init__(self,
                 env_id="FrozenLake-v1",
                 render_mode="rgb_array"
                 ):
        if env_id == "FrozenLake-v1":
            # ENv creation
            self.env = gym.make(env_id, map_name=FrozenLakeConfig.MAP_NAME,
                                is_slippery=FrozenLakeConfig.SLIPPERY,
                                render_mode=render_mode
                                )
        elif env_id == "Taxi-v3":
            self.env = gym.make(env_id, render_mode=render_mode)

        print("_____OBSERVATION SPACE_____ \n")
        print("Observation Space", self.env.observation_space)
        print("Sample observation", self.env.observation_space.sample())  # Get a random observation

        print("\n _____ACTION SPACE_____ \n")
        print("Action Space Shape", self.env.action_space.n)
        print("Action Space Sample", self.env.action_space.sample())  # Take a random action

        # initializing q table
        state_space = self.env.observation_space.n
        action_space = self.env.action_space.n
        self.Qtable = np.zeros((state_space, action_space))


    def greedy_policy(self, state):
        # taking the action with highest state, action value
        action = np.argmax(self.Qtable[state])
        return action

    def epsilon_greedy_policy(self, state, epsilon):
        # randomly generate number between 0 and 1
        random_num = np.random.uniform(0, 1)
        # if random_num > epsilon -> exploitation
        if random_num > epsilon:
            action = self.greedy_policy(state)
        else:
            # taking exploratio
            action = self.env.action_space.sample()
        return action

    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps):
        for episode in tqdm(range(n_training_episodes)):
            # Reduce epsilon (because we need less and less exploration)
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

            # reset tjhe env
            state, info = self.env.reset()
            step= 0
            terminated = False
            truncated = False

            #loop over episode
            for step in range(max_steps):
                # choose action at using epsilon greedy policy
                action = self.epsilon_greedy_policy(state, epsilon)

                # take action and observae the next state, and reward
                new_state, reward, terminated, truncated, info = self.env.step(action)

                # update q table
                self.Qtable[state][action] = self.Qtable[state][action] + FrozenLakeConfig.LEARNING_RATE * (
                    reward + FrozenLakeConfig.GAMMA * np.max(self.Qtable[new_state]) - self.Qtable[state][action]
                )

                # if terminated or truncated
                if terminated or truncated:
                    break

                state = new_state
        print(f'TRAINING COMPLETED, TOTAL EPOCH: {n_training_episodes}')

    def evaluate_agent(self,max_steps, n_eval_episodes, seed):
        episodes_rewards = []
        for episode in tqdm(range(n_eval_episodes)):
            if seed:
                state, info = self.env.reset(seed=seed[episode])
            else:
                state, info = self.env.reset()
            step =0
            truncated = False
            terminated = False
            total_rewards_ep = 0
            for step in range(max_steps):
                # take action greedliy
                action = self.greedy_policy(state)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                total_rewards_ep += reward

                if truncated or terminated:
                    break
                state = new_state
            episodes_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episodes_rewards)
        std_reward = np.std(episodes_rewards)
        print(f'MEAN REWARD: {mean_reward}   STANDARD DEVIATION: {std_reward}')

        return mean_reward, std_reward
    def record_video(self, out_dir, fps=1):
        images  = []
        teriminated = False
        truncated = False
        state, info = self.env.reset(seed=random.randint(0, 500))
        img = self.env.render()
        images.append(img)
        while not teriminated or truncated:
            action = np.argmax(self.Qtable[state][:])
            state, reward, teriminated, truncated, info = self.env.step(action)
            img = self.env.render()
            images.append(img)
        imageio.mimsave(out_dir, [np.array(img) for i, img in enumerate(images)], fps=fps)
        print(f'VIDEO SAVED: {out_dir}')

if __name__ == '__main__':
    # qlearning_obj = QLearning(
    #     env_id="FrozenLake-v1"
    # )
    qlearning_obj = QLearning(
        env_id="Taxi-v3"
    )
    '''
    # Frozen lake experiments
    qlearning_obj.train(
        n_training_episodes=FrozenLakeConfig.N_TRAINING_EPISODES,
        min_epsilon=FrozenLakeConfig.MIN_EPSILON,
        max_epsilon=FrozenLakeConfig.MAX_EPSILON,
        decay_rate=FrozenLakeConfig.DECAY_RATE,
        max_steps=FrozenLakeConfig.MAX_STEPS,
    )
    print(f'evaluating agent')
    qlearning_obj.evaluate_agent(
        max_steps=FrozenLakeConfig.MAX_STEPS,
        n_eval_episodes=FrozenLakeConfig.N_EVAL_EPISODES,
        seed = None
    )

    qlearning_obj.record_video(out_dir=FrozenLakeConfig.VIDEO_FILENAME)
    '''

    # Taxi experminets
    qlearning_obj.train(
        n_training_episodes=TaxiV3Config.N_TRAINING_EPISODES,
        min_epsilon=TaxiV3Config.MIN_EPSILON,
        max_epsilon=TaxiV3Config.MAX_EPSILON,
        decay_rate=TaxiV3Config.DECAY_RATE,
        max_steps=TaxiV3Config.MAX_STEPS,
    )
    print(f'evaluating agent')
    qlearning_obj.evaluate_agent(
        max_steps=TaxiV3Config.MAX_STEPS,
        n_eval_episodes=TaxiV3Config.N_EVAL_EPISODES,
        seed=TaxiV3Config.EVAL_SEED
    )
    qlearning_obj.record_video(out_dir=TaxiV3Config.VIDEO_FILENAME)

