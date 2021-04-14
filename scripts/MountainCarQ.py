import numpy as np
import gym

env  = gym.make("MountainCar-v0")
done = False

# hyperparameter

DISCRETE_OS_SIZE = [20 , 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
print(discrete_os_win_size)
Q = np.random.uniform(low = -2 , high = 0 , size = (DISCRETE_OS_SIZE + [env.action_space.n]))
print(Q.shape)

DISCOUNT   = 0.99
EPISODES   = 25000
SHOW_EVERY = 2000
learning_rate = 0.1

epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING   = EPISODES // 2
epsilon_decay_value    = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING )

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low) / discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

for eps in range(EPISODES):
	if eps % 10 == 0:
		render = True
	else:
		render = False
	discrete_state = get_discrete_state(env.reset())
	done = False
	while not done:
		max_rewards = []
		if np.random.random() > epsilon:
			action = np.argmax(Q[discrete_state])
		else:
			action = np.random.randint(0, env.action_space.n)
		action = np.argmax(Q[discrete_state]) # take action
		new_state  , reward , done , _ = env.step(action) # get newstate , reward , done
		new_discrete_state = get_discrete_state(new_state) # new_discrete_state
		max_rewards.append(reward)
		if render:
			env.render()
		if not done:
			max_future_q = np.max(Q[new_discrete_state]) # calclate max rewarded action
			current_q = Q[discrete_state + (action , )]  # get Q value of this state
			Q[discrete_state + (action ,)] = (1 - learning_rate) * current_q + learning_rate * (reward + DISCOUNT * max_future_q)

		elif new_state[0] >= env.goal_position:
			Q[discrete_state + (action , )] = 0
		discrete_state = new_discrete_state
	print("episode {} , epslion = {} , reward = {}".format(eps , epsilon , sum(max_rewards) / len(max_rewards)))
	if END_EPSILON_DECAYING >= eps >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value

env.close()
