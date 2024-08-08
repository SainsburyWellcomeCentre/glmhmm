import matplotlib.pyplot  as plt
import numpy as np

def plot_model_weights(mouse, GLM, bias = True):

    if bias:

        xlabels = [
            'Cue identity',
            'History of last choice 1',
            'History of last choice 2',
            'History of last choice 3',
            'History of last choice 4',
            'History of last choice 5',
            'Last rewarded choice',
            'Distance to 0',
            'Distance to 1',
            'bias'
        ]
    else:
            
            xlabels = [
            'Cue identity',
            'History of last choice 1',
            'History of last choice 2',
            'History of last choice 3',
            'History of last choice 4',
            'History of last choice 5',
            'Last rewarded choice',
            'Distance to 0',
            'Distance to 1'
        ]

    fig, ax  = plt.subplots()
    ax.set_facecolor('white')
    ax.plot(GLM.w)
    ax.set_xticklabels(xlabels, rotation =  90)
    ax.plot(xlabels,np.zeros((len(xlabels),1)),'k--')
    ax.set_xticks(np.arange(0,len(xlabels)))
    ax.set_ylabel('Weights')
    fig.suptitle(f'GLM weights for {mouse}, {GLM.n} trials')

    return fig, ax