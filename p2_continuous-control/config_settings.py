class Args:
  max_steps=4 #1000 Original DDPG pendulum : n_episodes=1000 #1000  #Original DDPG pendulum : max_t=300
  num_episodes=2 #225  #225 Original DDPG pendulum : n_episodes=1000
  #pretrain=9#129 #5000 assert args.pretrain 5000 >= args.batch_size 128, "PRETRAIN less than BATCHSIZE."
  batch_size=8 #128
  buffer_size=10 #300000
  actor_learn_rate=0.0005
  critic_learn_rate=0.001

  buffer_size=300000
  C=350
  layer_sizes=[400,300]
  cpu=True
  e=0.3 # exploration rate
  vmin=0.0
  vmax=0.3
  num_atoms=100
  eval=True
  force_eval=True
  gamma=0.99
  nographics=True
  quiet=True
  resume=True
  rollout=5
  save_every=10
  log_every=50
  print_every=3 #Original DDPG pendulum : print_every=100
  tau=0.0005
  latest=True
  filename=None
  save_dir='saves'
  device='cpu'
