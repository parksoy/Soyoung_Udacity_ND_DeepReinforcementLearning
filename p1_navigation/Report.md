[image2]: /Users/parksoy/Desktop/deep-reinforcement-learning/p1_navigation/banana_POR_episode_534_withtitle.jpg "Plot_score"

# Report
##### A description of the implementation  

This project scopes to learn how to train an agent to navigate and collect bananas in a large, square world.  A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas. The state space has 37 dimensions including the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.


# Methodology

The same version of Deep Q Learning (DQN) example in this nanodegree was used as published in

```
https://github.com/parksoy/Soyoung_Udacity_ND_DeepReinforcementLearning/blob/master/dqn/exercise/model.py
https://github.com/parksoy/Soyoung_Udacity_ND_DeepReinforcementLearning/blob/master/dqn/exercise/dqn_agent.py
```
and was integrated into Unity ML-Agents environment. The agent's actions are controlled by DQN learning process and reward is obtained from Unity environment.

#### (1) Learning algorithm  

DQN network is used for each qnetwork_local and qnetwork_target.

DQN has the following key features :

* Experience replay buffer(class ReplayBuffer):
to store and sample the experience tuples consisting of (state, action, reward, next_state, done) fields. step() function saves new experience tuples as they are encountered and then randomly samples experience tuples to learn from.

* Parameters are decoupled (qnetwork_local parameters and qnetwork_target parameters) to be updated for learning from the ones being used to produce target values.

* Soft updates :
learn() method uses soft_update to smooth out individual learning batches by preventing large fluctuations in actions from being generated during learning.
```
target_param = tau*local_param + (1.0-tau)*target_param
```
where local_model (PyTorch model): weights will be copied from
      target_model (PyTorch model): weights will be copied to

* Epsilon-greedy action :
eps_start=1.0, eps_end=0.01, eps_decay=0.995  
to encourage exploratory behavior in the agent during the early episodes in act() method. Later on when enough learning happened, action with the most rewards is chosen.

```
if random.random() > eps:
    return np.argmax(action_values.cpu().data.numpy())
else:
    return random.choice(np.arange(self.action_size))
```

#### (3) The model architectures for neural networks

The given neural network for DQN network was used as is, which consists of three hidden linear neural network layers:

```
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```
where state_size=37, fc1_units=64, fc2_units=64, action_size=4.

The first hidden layer fc1 takes 37 dimensions and forwards to the second layer with the size of 64 through the relu. The second hidden layer takes 64 inputs and forwards to the output with the size of 4 actions through relu. This networks makes links between 37 states dimensions and 4 actions.


#### (2) The purpose of each hyperparameter and the values

The same given hyperparameters are used as the following:

* fc1_units = 64  
:number of units in the first NN hidden layer

* fc2_units = 64  
:number of units in the second NN hidden layer

* TAU = 1e-3     (interpolation parameter)  
:to control soft-update process in agent's Learn() method.

* epsilon  (epsilon-greedy action selection)  
eps_start=1.0, eps_end=0.01, eps_decay=0.995
: in act() function , an epsilon-greedy action selection mechanism is used to encourage exploratory behavior in the agent, particularly during the early episodes. epsilon value is set to control this process - when high, more exploitation is encouraged, when low, more exploration is encouraged.

* LR = 5e-4 (learning rate)  
:the local network learns with gradient decent based optimizer with slow step size. Used in the following fashion
```
optim.Adam(self.qnetwork_local.parameters(), lr=LR)
```
* GAMMA = 0.99 (discount factor)  
: During learn(), the future reward is discounted by (1 - dones)*0.99 as opposed to the current reward being fully counted.
```
Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
```

* UPDATE_EVERY = 4  
: how often (every 4 step in this case) to update the network in section of step() method. step() function saves new experience tuples as they are encountered and then randomly selecting experience tuples to learn from.


# A plot of rewards

![Plot_score][image2]

A plot of rewards per episode is included to illustrate that the agent is able to receive an average reward (over 100 episodes) of at least +13. In order for the environment to achieve at least score of 13, 434 episodes was needed to solve the environment.


# Ideas for Future Work

#### (1) With the same DQN, the hyperparameters other than number of units in each layer can be varied.  

For example, since navigation time between bananas is pretty short, I need to eat yellow banana as quickly as when available at sight before advancing to unwanted place. If I give more discount on the future rewards by lowering GAMMA from 0.99, would it learn better and achieve higher score?

#### (2) Fundamental learning algorithm expansion  

In order to improve the agent's performance faster and to achieve higher score, this [reference](https://arxiv.org/pdf/1710.02298.pdf) summarizes, limitations of conventional DQN and discusses several fundamental learning strategy changes as following and they can be pursued for further improvement beyond score of 13 with 434 episodes for this banana project.

* Conventional Q-learning could have an overestimation bias due to the maximization step. Double Q-learning addresses this overestimation issue by decoupling, in the maximization performed for the bootstrap target, the selection of the action from its evaluation.  

* DQN samples uniformly from the replay buffer. Ideally, we want to *sample more frequently* those transitions from which there is much to learn. As a proxy for learning potential, __prioritized experience replay__(Schaul et al. 2015) samples transitions with probability relative to the last encountered absolute TD error. New transitions are inserted into the replay buffer with maximum priority, providing a bias towards recent transitions. Note that stochastic transitions might also
be favored, even when there is little left to learn about them.

* The __dueling network__ for value based RL features two streams of computation. The value and advantage streams share convolutional encoder, and merged by a special aggregator (Wang et al. 2016).

More references (suggested by reviewer #2):  
* [Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)  
* [Dueling Network](https://arxiv.org/pdf/1511.06581.pdf)  
