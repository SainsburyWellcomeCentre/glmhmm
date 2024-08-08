import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import GoalSelection.training_metrics as tm
from pathlib import Path
import glmhmm.glm as glm

PORTS = [[0.6, 0.35], 
         [-0.6, 0.35], 
         [0, -0.7]]

def design_matrix_per_mouse(mouse, start_session = -10):

    date_dirs = tm.get_date_dirs(mouse)
    date_dirs = date_dirs[start_session:]

    design_list  = []
    output_list =  []
    date_list = []
    trial_list = []

    for date in date_dirs:
        print(date)
        #build the matrix
        exp_data = tm.build_exp_data(mouse, date)
        filtered_data = filter_data(exp_data)
        design, y = build_design_matrix(filtered_data)
        design_list.append(design)
        output_list.append(y)
        date_list.append([date]*len(design))
        trial_list.append(design['trial'])

    design_concat = pd.concat(design_list, ignore_index   = True)
    y = np.concatenate(output_list)
    date_concat = np.concatenate(date_list)
    trial_concat =  pd.concat(trial_list, ignore_index = True)

    X = format_matrix(design_concat)
    row_identity = pd.DataFrame({'date': date_concat, 
                                'trial': trial_concat})

    return X, y, row_identity, design_concat


def filter_data(exp_data):
    '''
    We will: 
        - Get rid of aborted trials
        - Get rid of 5 first trials
        - Assume that last-5-trials can be computing ignoring 
        aborted trials. This  is  even though there are prolongued periods of time w/o many trials. 
    '''
    filtered_data = exp_data[~exp_data['TrialCompletionCode'].str.startswith('Aborted')]
    return  filtered_data

def build_design_matrix(filtered_data, bias =  True):
    '''
    Make the df for the design matrix. Individual variables have their
    own functions, see below. Will skip rows for  which one of the variables is
    not defined, see comments in  function body. 

    Args:
        filtered_data: An exp_data-like pandas df, filtered by filter_data
    Out:
        design: a pd df w trial number and glm variables
        y: a np.array() of Bernouilli outputs
    '''
    trial=[]
    cue  =  []
    last_rewarded = []
    last_1 = []
    last_2 = []
    last_3 = []
    last_4 = []
    last_5 = []
    distance_0 = []
    distance_1 = []
    iloc = 4 #to generate a purely positional index,not the pandas index,so as to look
    #at last rows

    output_vector = [] #For the actual choice in each trial

    for index, row in filtered_data.iloc[5:].iterrows():
        iloc+=1

        #Deal with not having a last rewarded trial before the begginning, 
        #so trial 5. Just advance one by one and raise warnings. 
        try:
            last_rewarded.append(get_last_rewarded(iloc, filtered_data))
        except ValueError:
            warnings.warn(f'ValueError encountered at iloc={iloc}, continuing. This row will be EXCLUDED')
            continue

        trial.append(row['TrialNumber'])
        cue.append(get_cue(row))
        distance_0.append(distance_to_port(row, 0))
        distance_1.append(distance_to_port(row, 1))
        last_1.append(get_last(1, iloc, filtered_data))
        last_2.append(get_last(2, iloc, filtered_data))
        last_3.append(get_last(3, iloc, filtered_data))
        last_4.append(get_last(4, iloc, filtered_data))
        last_5.append(get_last(5, iloc, filtered_data))

        output_vector.append(row['TrialCompletionCode'][-1])

    design_matrix = {
        'trial': trial,
        'cue': cue,
        'distance_0': distance_0,
        'distance_1': distance_1,
        'last_rewarded': last_rewarded,
        'last_1': last_1,
        'last_2': last_2,
        'last_3': last_3,
        'last_4': last_4,
        'last_5': last_5
    }

    design = pd.DataFrame(design_matrix)
    if bias:
        bias_vector = np.zeros_like(cue)+1
        design['bias'] =  bias_vector
    y = np.array(output_vector).astype(float)

    return design, y


# Functions  to  extract individual varirables from exp_data
def get_cue(row):
    '''
    Get which cue was played from the sound card index
    '''
    if row['AudioCueIdentity']==10:
        cue = 1
    elif row['AudioCueIdentity']==14:
        cue = 0
    else:
        print('UNRECOGNISED SOUND CUE')
        cue  =  None
    return  cue

def distance_to_port(row, port):
    '''
    Get the euclidean  distance from the dot of light to the 
    port specified in port (0, 1, 2)
    '''
    port = np.array(PORTS[port])
    dot = np.array([row['DotXLocation'], row['DotYLocation']])
    v_distance = port-dot
    distance = np.sqrt((v_distance[0]**2)+(v_distance[1]**2))
    
    return distance

def get_last_rewarded(iloc, filtered_data):
    '''
    Identity of the port in the last rewarded choice
    '''
    if iloc <= 0 or iloc > len(filtered_data):
        raise ValueError("Invalid iloc value")

    new_row = filtered_data.iloc[iloc-1]
    jump_back = 1
    
    while not new_row['TrialCompletionCode'].startswith('Rewarded'):
        
        jump_back += 1
        if iloc - jump_back < 0:
            raise ValueError(f"No previous 'Rewarded' TrialCompletionCode found for iloc {iloc}")
        new_row = filtered_data.iloc[iloc-jump_back]
    
    last_rew = new_row['TrialCompletionCode'][-1]
    return last_rew

def get_last(position, iloc, filtered_data):
    '''
    What the animal chose {position} positions back, useful
    for choice history bias
    '''
    new_row = filtered_data.iloc[iloc-position]
    past_choice= new_row['TrialCompletionCode'][-1]
    return past_choice

#Interfacing with the GLM class

def build_GLM(design, y):
    '''
    c  is 2  in a  bernouilli choice. 

    n: number of data/time points
    d: number of features (inputs to design matrix)
    c: number of classes (possible observations)
    x: design matrix (nxm)
    y: observations (nxc)
    w: weights mapping x to y (mxc or mx1)
    '''
    n = len(design)
    d = len(design.columns)-1
    c = 2

    GLM = glm.GLM(n, d, c) 

    return GLM

def format_matrix(design, bias = True):
    '''
    Make design df into an array of the right shape
    Args: 
        design: pd df from prev function
    Out:
        X: np.array of floats, n x d (timepoints by variables)
    '''
    if bias:
        X = np.array([
            design['cue'].tolist(), 
            design['last_1'].tolist(), 
            design['last_2'].tolist(), 
            design['last_3'].tolist(), 
            design['last_4'].tolist(), 
            design['last_5'].tolist(), 
            design['last_rewarded'].tolist(),
            design['distance_0'].tolist(),
            design['distance_1'].tolist(), 
            design['bias'].tolist()
        ])
    else:
        X = np.array([
            design['cue'].tolist(), 
            design['last_1'].tolist(), 
            design['last_2'].tolist(), 
            design['last_3'].tolist(), 
            design['last_4'].tolist(), 
            design['last_5'].tolist(), 
            design['last_rewarded'].tolist(),
            design['distance_0'].tolist(),
            design['distance_1'].tolist()
        ])
    X = X.T
    X = X.astype(float)

    return X