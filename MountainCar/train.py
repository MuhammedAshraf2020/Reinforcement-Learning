import gym
from collections import deque
from agent import DQNAGENT 
import numpy as np
import random
import os

random.seed(2212)
np.random.seed(2212)

EPISODES = 10000
REPLAY_MEMORY_SIZE = 100000
MINIMUM_REPLAY_MEMORY = 1000
MINIBATCH_SIZE = 12
EPSLION = 1
EPSLION_DECAY = 0.99
MINIMUM_EPSLION = 0.001
DISCOUNT = 0.99
VISUALIZATION = True
ENV_NAME = "MountainCar-v0"


env = gym.make(ENV_NAME)
action_dim = env.action_space.n
observation_dim = env.observation_space.shape
replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)
agent = DQNAGENT(action_dim , observation_dim)


def train_dqn_agent():
	minibatch = random.sample(replay_memory , MINIBATCH_SIZE) # [12 , 5]
	X_cur_states  = []
	X_next_states = []
	for index , sample in enumerate(minibatch):
		cur_state , action , reward , next_state , done = sample 
		X_cur_states.append(cur_state)
		X_next_states.append(next_state)

	X_cur_states  = np.array(X_cur_states)  # [12 , 1 , 2] s
	X_next_states = np.array(X_next_states) # [12 , 1 , 2] s'

	cur_action_values  = agent.model.predict(X_cur_states)  # a  [12 , 3]
	next_action_values = agent.model.predict(X_next_states) # a' [12 , 3]

	for index , sample in enumerate(minibatch):
		cur_state , action , reward , next_state , done = sample # _ , 1 , -1 , _2 , False
		if not done:
			cur_action_values[index][action] = reward + DISCOUNT * np.amax(next_action_values[index])
			# cur_action_values[0][1]  = -1 + 0.99 * max(next_action_values[0])
		else:
			cur_action_values[index][action] = reward # cur_action_values[0][1] = reward

	agent.model.fit(X_cur_states , cur_action_values , verbose = 0) # input = X_cur_states , output = cur_action_values 


max_rewards = -99999
for episode in range(EPISODES):
	cur_state = env.reset()
	done = False
	episode_reward = 0
	episode_length = 0
	render = True if (episode > 200)  else  False
	while not done:
		if render:
			env.render()
		episode_length += 1
		
		if np.random.uniform(low = 0 , high = 1) < EPSLION:
			action = np.random.randint(0 , action_dim)
		else:
			action = np.argmax(agent.model.predict(np.expand_dims(cur_state , axis = 0)) [0])

		next_state , reward , done , _ = env.step(action)
		episode_reward += reward

		if done and episode_length < 200:
			reward = 250 + episode_reward
			if episode_reward > max_rewards:
				agent.model.save_weights(str(episode_reward) + "_agent_.h5")
		else:
			reward = 5 * abs(next_state[0] - cur_state[0]) + 3 * abs(cur_state[1])

		replay_memory.append((cur_state , action , reward , next_state , done))
		cur_state = next_state
		if(len(replay_memory) < MINIMUM_REPLAY_MEMORY):
			continue

		train_dqn_agent()

	if EPSLION > MINIMUM_EPSLION :
		EPSLION *= EPSLION_DECAY

	max_rewards = max(episode_reward , max_rewards)
	print("EPISODE " , episode , "Reward " , episode_reward , "Maximum Reward " , max_rewards , "EPSLION " , EPSLION)