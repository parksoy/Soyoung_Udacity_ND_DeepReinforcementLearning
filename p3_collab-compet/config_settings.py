import torch

class Args:
    num_episodes=3000          # Num of games/episode
    max_steps=1000             # max num of times of steps in one episode
    batch_size=1024            # how many experiences are sampled and learned in batch processing to go through neural networks

    actor_learn_rate=0.001    # actors learning rate
    critic_learn_rate=0.001    # critic learning rate

    update_every=20            # frequency to learn with stability
    num_updates=10             # how many times to learn during "update_every" times

    tau=0.005                  # soft update speed of target parameters
    gamma=0.99                 # discount factor

    noise_sigma=0.2            # noise spread
    noise_factor_decay = 1e-6  # Noise decay speed

    layer_sizes=[512,512]      # Neural Network unit size for first, second layers. ,128

    buffer_size=3000000        # replay buffer size
    noise_factor=1
    print_every=100            # stdout print frequency

    scoreGoal=0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   #automatically determine 'gpu' or 'cpu'
