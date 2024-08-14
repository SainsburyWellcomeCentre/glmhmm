import matplotlib.pyplot  as plt
import numpy as np
import glmhmm.utils as uti

def plot_model_weights(mouse, GLM, bias = True):

    if bias:

        xlabels = [
            'Bias',
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

def plot_model_weights_states(mouse, w_all, lls_all, GLMHMM, bias=True):

    bestix = uti.find_best_fit(lls_all) # find the initialization that led to the best fit
    weights_end = w_all[bestix,  :, :, :]
    print(weights_end.shape)
    if bias:

        xlabels = [
            'Bias',
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
    fig, ax = plt.subplots(GLMHMM.k)
    for i in range(GLMHMM.k):
        ax[i].plot(weights_end[i,:,:])

    ax[GLMHMM.k-1].set_xticks(np.arange(0,len(xlabels)))
    ax[GLMHMM.k-1].plot(xlabels,np.zeros((len(xlabels),1)),'k--')
    ax[GLMHMM.k-1].set_xticklabels(xlabels, rotation =  90)
    trials = GLMHMM.n
    states = GLMHMM.k
    fig.suptitle(f'GLMHMM weights for {mouse}, {trials} trials, {states} states')

    return fig, ax
