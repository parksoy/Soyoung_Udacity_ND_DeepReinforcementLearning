class Args:
    
    max_steps=1000             #4 #Original DDPG pendulum : n_episodes=1000 max_t=300
    batch_size=1024            #8 128

    actor_learn_rate=0.001
    critic_learn_rate=0.001
    
    update_every=20           #20
    num_updates=10            #10
    
    tau=0.005                 ## for soft update of target parameters
    gamma=0.99                # discount factor
    
    noise_sigma=0.2
    noise_factor_decay = 1e-6

    layer_sizes=[128,128]     #[400,300]
    
    num_episodes=105          #2 #225  #225 Original DDPG pendulum : n_episodes=1000
    buffer_size=3000000       #int(1e6) #10 #300000
    
    ##############################
    #NOT SO IMPORTANT
    ##############################
    noise_factor=1
    print_every=10            #Original DDPG pendulum : print_every=100
    
    C=350
    cpu=True
    e=0.3                     # exploration rate
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
