import matplotlib.pyplot as plt
import numpy as np

####################################
#Plot Score vs. Episode
####################################
def plot_scoreOverEpisodes(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Score vs. Episode #')
    plt.show()
