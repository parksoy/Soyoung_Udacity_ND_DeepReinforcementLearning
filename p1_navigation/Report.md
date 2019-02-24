[image2]: /Users/parksoy/Desktop/deep-reinforcement-learning/p1_navigation/banana_POR_episode_534_withtitle.jpg "Plot_score"

# Report
##### A description of the implementation  

This project scopes to learn how to train an agent to navigate and collect bananas in a large, square world.  A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas. The state space has 37 dimensions including the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.


# Methodology

The same methodology Deep Q Learning (DQN) example in this nanodegree was used as published in

```
https://github.com/parksoy/Soyoung_Udacity_ND_DeepReinforcementLearning/blob/master/dqn/exercise/model.py
https://github.com/parksoy/Soyoung_Udacity_ND_DeepReinforcementLearning/blob/master/dqn/exercise/dqn_agent.py
```

(1) Learning algorithm  

DQN network is used for each qnetwork_local and qnetwork_target.

(3) The model architectures for neural networks

The given neural network for DQN network was used as is, which consists of three hidden linear neural network layers:

```
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```

Each layer is forwarded and mapped through relu.


(2) The chosen hyperparameters

The same given hyperparameters are used as the following:
```
fc1_units=64,
fc2_units=64
```


# A plot of rewards

![Plot_score][image2]

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. In order for the environment to achieve at least score of 13, 434 episodes was needed to solve the environment.


# Ideas for Future Work

In order to improve the agent's performance (less number of episodes to be explored to achieve the same score of 13 or more), the following ideas are considered:

(1) The more complex network other than three linear neural networks can be considered.
(2) The hyperparameters other than number of units in each layer can be varied.
