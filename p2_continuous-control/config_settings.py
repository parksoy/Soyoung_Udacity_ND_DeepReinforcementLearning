class Args:
  max_steps=300 #4 #Original DDPG pendulum : n_episodes=1000 max_t=300
  num_episodes=200 #2 #225  #225 Original DDPG pendulum : n_episodes=1000
  batch_size=256  #8 128
  buffer_size=300000 #int(1e6) #10 #300000
  actor_learn_rate=0.0005
  critic_learn_rate=0.001
  
  update_every=10#20
  print_every=3 #Original DDPG pendulum : print_every=100
    
  tau=0.0005 ## for soft update of target parameters
  gamma=0.99              # discount factor
    
  C=350
  layer_sizes=[128,128]#[400,300]
  cpu=True
  e=0.3 # exploration rate
  vmin=0.0
  vmax=0.3
  num_atoms=100
  eval=True
  force_eval=True
  
  nographics=True
  quiet=True
  resume=True
  rollout=5
  save_every=10
  log_every=50
  

  latest=True
  filename=None
  save_dir='saves'
  device='cpu'
