class MADDPG():
    """Meta agent that contains the two DDPG agents and shared replay buffer."""

    def __init__(self, action_size=2, seed=0, load_file=None,
                 n_agents=2,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 gamma=GAMMA,
                 update_every=UPDATE_EVERY,
                 noise_start=NOISE_START,
                 noise_decay=NOISE_DECAY,
                 evaluation_only=False):
        """action_size (int): dimension of each action
        seed (int): Random seed
        load_file (str): path of checkpoint file to load
        n_agents (int): number of distinct agents
        buffer_size (int): replay buffer size
        batch_size (int): minibatch size
        gamma (float): discount factor
        update_every (int): how often to update the network
        noise_start (float): initial noise weighting factor
        noise_decay (float): noise decay rate
        evaluation_only (bool): set to True to disable updating gradients and adding noise"""

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.n_agents = n_agents
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
        self.evaluation_only = evaluation_only

        # create two agents, each with their own actor and critic
        models = [model.DoubleAgent(n_agents=n_agents) for _ in range(n_agents)]
        self.agents = [DDPG(0, models[0]), DDPG(1, models[1])]

        # create shared replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

        if load_file:
            for i, save_agent in enumerate(self.agents):
                actor_file = torch.load(load_file + '.' + str(i) + '.actor.pth', map_location='cpu')
                critic_file = torch.load(load_file + '.' + str(i) + '.critic.pth', map_location='cpu')
                save_agent.actor_local.load_state_dict(actor_file)
                save_agent.actor_target.load_state_dict(actor_file)
                save_agent.critic_local.load_state_dict(critic_file)
                save_agent.critic_target.load_state_dict(critic_file)
            print('Loaded: {}.actor.pth'.format(load_file))
            print('Loaded: {}.critic.pth'.format(load_file))

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and self.evaluation_only == False:
            if len(self.memory) > self.batch_size: # If enough samples are available in memory, get random subset and learn
                for _ in range(NUM_UPDATES):
                    experiences = [self.memory.sample() for _ in range(self.n_agents)]# each agent does it's own sampling from the replay buffer
                    self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True):
        all_actions = []# pass each agent's state from the environment and calculate it's action
        for agent, state in zip(self.agents, all_states):
            if self.evaluation_only:
                action = agent.act(state, noise_weight=self.noise_weight, add_noise=False)
            else:
                action = agent.act(state, noise_weight=self.noise_weight, add_noise=True)
            self.noise_weight -= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1) # reshape 2x2 into 1x4 dim vector

    def learn(self, experiences, gamma):
        all_next_actions = []# each agent uses it's own actor to calculate next_actions
        for i, agent in enumerate(self.agents):
            _, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)

        # each agent uses it's own actor to calculate actions
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, _, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)

        for i, agent in enumerate(self.agents):# each agent learns from it's experience sample
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)
