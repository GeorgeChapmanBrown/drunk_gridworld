# This is needed (on my machine at least) due to weird python import issues
from generative_model import REWARD_LOCATION
import os
import sys
from pathlib import Path

import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from pymdp import maths, utils
from pymdp.maths import spm_log_single as log_stable # @NOTE: we use the `spm_log_single` helper function from the `maths` sub-library of pymdp. This is a numerically stable version of np.log()
from pymdp import control

class GridWorldEnv():

    def __init__(self,A,B):
        self.A = deepcopy(A)
        self.B = deepcopy(B)
        # print("B:", B.shape)
        self.state = np.zeros(24)
        # start at state 3
        self.state[0] = 1

    def step(self,a):
        self.state = np.dot(self.B[:,:,a], self.state)
        obs = utils.sample(np.dot(self.A, self.state))
        return obs

    def reset(self):
        self.state =np.zeros(24)
        self.state[0] =1 
        obs = utils.sample(np.dot(self.A, self.state))
        return obs

def KL_divergence(q,p):
    return np.sum(q * (log_stable(q) - log_stable(p)))

def compute_free_energy(q,A, B):
    return np.sum(q * (log_stable(q) - log_stable(A) - log_stable(B)))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def perform_inference(likelihood, prior):
    return softmax(log_stable(likelihood) + log_stable(prior))

def evaluate_policy(policy, Qs, A, B, C):
    # initialize expected free energy at 0
    G = 0

    # loop over policy
    for t in range(len(policy)):

        # get action entailed by the policy at timestep `t`
        u = int(policy[t])

        # work out expected state, given the action
        Qs_pi = B[:,:,u].dot(Qs)

        # work out expected observations, given the action
        Qo_pi = A.dot(Qs_pi)

        # get entropy
        H = - (A * log_stable(A)).sum(axis = 0)

        # get predicted divergence
        # divergence = np.sum(Qo_pi * (log_stable(Qo_pi) - log_stable(C)), axis=0)
        divergence = KL_divergence(Qo_pi, C)

        # compute the expected uncertainty or ambiguity 
        uncertainty = H.dot(Qs_pi)

        # increment the expected free energy counter for the policy, using the expected free energy at this timestep
        G += (divergence + uncertainty)

    return -G

def infer_action(Qs, A, B, C, n_actions, policies):

    # initialize the negative expected free energy
    neg_G = np.zeros(len(policies))

    # loop over every possible policy and compute the EFE of each policy
    for i, policy in enumerate(policies):
        neg_G[i] = evaluate_policy(policy, Qs, A, B, C)

    # get distribution over policies
    Q_pi = maths.softmax(neg_G)

    # initialize probabilites of control states (convert from policies to actions)
    Qu = np.zeros(n_actions)

    # sum probabilites of control states or actions 
    for i, policy in enumerate(policies):
        # control state specified by policy
        u = int(policy[0])
        # add probability of policy
        Qu[u] += Q_pi[i]

    # normalize action marginal
    utils.norm_dist(Qu)

    # sample control from action marginal
    u = utils.sample(Qu)

    return u

def change_reward(C, z, home, bar, lake):
    # when z = 0, set everything but lake and home to reward 0.5
    if z == 0:
        for i in range(len(C)):
            if i in lake or i == home:
                pass
            else:
                C[i] = 0.5
    # when z > 0, set home to 1, lake to z*0.05 and bar to z*0.1
    if z > 0:
        C[home] = 1.
        

    # print(C)
    return C




def start_generative_model(action):

    w = 1.
    print(w)
    path = Path(os.getcwd())
    print(path)
    module_path = str(path.parent) + '/'
    sys.path.append(module_path)

    REWARD_LOCATION = 23

    # bar locations
    bar = [1, 3, 12, 14]
    # lake locations
    lake = [4, 5, 6, 17, 18]
    # home location
    home = REWARD_LOCATION
    # drunk level
    z = 0

    # A matrix
    A = np.eye(24)

    # construct B matrix

    P = {}
    dim_x, dim_y = 8, 3
    actions = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'STAY':4}
    state_mapping = {}
    counter = 0

    for y in range(dim_y):
        for x in range(dim_x):
            state_mapping[counter] = ((x,y))
            counter +=1
    print(state_mapping)

    labels = [state_mapping[i] for i in range(A.shape[1])]

    for state_index, xy_coordinates in state_mapping.items():
        P[state_index] = {a : [] for a in range(len(actions))}
        x, y = xy_coordinates

        '''if your y-coordinate is all the way at the top (i.e. y == 0), you stay in the same place -- otherwise you move one upwards (achieved by subtracting 3 from your linear state index'''
        P[state_index][actions['UP']] = state_index if y == 0 else state_index - dim_x 
        # if state_mapping[P[state_index][actions['UP']]] == bar and z < 4:
        #     z += 1

        '''f your x-coordinate is all the way to the right (i.e. x == 2), you stay in the same place -- otherwise you move one to the right (achieved by adding 1 to your linear state index)'''
        P[state_index][actions["RIGHT"]] = state_index if x == (dim_x -1) else state_index+1 
        # if state_mapping[P[state_index][actions['RIGHT']]] == bar and z < 4:
        #     z += 1

        '''if your y-coordinate is all the way at the bottom (i.e. y == 2), you stay in the same place -- otherwise you move one down (achieved by adding 3 to your linear state index)'''
        P[state_index][actions['DOWN']] = state_index if y == (dim_y -1) else state_index + dim_x 
        # if state_mapping[P[state_index][actions['DOWN']]] == bar and z < 4:
        #     z += 1

        ''' if your x-coordinate is all the way at the left (i.e. x == 0), you stay at the same place -- otherwise, you move one to the left (achieved by subtracting 1 from your linear state index)'''
        P[state_index][actions['LEFT']] = state_index if x == 0 else state_index -1 
        # if state_mapping[P[state_index][actions['LEFT']]] == bar and z < 4:
        #     z += 1

        ''' Stay in the same place (self explanatory) '''
        P[state_index][actions['STAY']] = state_index

    # print(f'P: {P}')

    num_states = 24
    B = np.zeros([num_states, num_states, len(actions)])
    for s in range(num_states):
        for a in range(len(actions)):
            ns = int(P[s][a])
            B[ns, s, a] = 1

    env = GridWorldEnv(A,B)

    # setup initial prior beliefs -- uncertain -- completely unknown which state it is in
    Qs = np.ones(24) * 1/9

    # C matrix -- desires

    reward_state = state_mapping[REWARD_LOCATION]
    ## C is the reward state. Maybe playing around with this value changes where the agent wants to go?
    # C = np.zeros(num_states)
    # C[REWARD_LOCATION] = 1. 

    # number of time steps
    T = 10

    #n_actions = env.n_control
    n_actions = 5

    # length of policies we consider
    policy_len = 4

    # this function generates all possible combinations of policies
    policies = control.construct_policies([B.shape[0]], [n_actions], policy_len)

    # reset environment
    o = env.reset()

    ##############
    # from matplotlib import animation

    Qs = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    cur_pos = list(Qs).index(1)

    x_cord, y_cord = state_mapping[cur_pos]
    x_cord_prev = x_cord
    y_cord_prev = y_cord

    ##############


    # loop over time
    while 1:

        C = np.zeros(num_states)
        # change reward state based on drunk level
        C = change_reward(C, z, home, bar, lake)

        # infer which action to take
        a = infer_action(Qs, A, B, C, n_actions, policies)

        # perform action in the environment and update the environment
        o = env.step(int(a))

        # infer new hidden state (this is the same equation as above but with PyMDP functions)
        likelihood = A[o,:]
        prior = B[:,:,int(a)].dot(Qs)

        Qs = maths.softmax(log_stable(likelihood) + log_stable(prior))

        # print(Qs.round(3))
      
        # print(list(Qs).index(1))
        cur_pos = list(Qs).index(1)
        # print(cur_pos)
        # x_cord = plot_pos(fig, axim, cur_pos, bar, lake, home)

        x_cord, y_cord = state_mapping[cur_pos]

        if x_cord > x_cord_prev:
            movement = 'right'
        if x_cord < x_cord_prev:
            movement = 'left'
        if y_cord > y_cord_prev:
            movement = 'down'
        if y_cord < y_cord_prev:
            movement = 'up'
        if x_cord == x_cord_prev and y_cord == y_cord_prev:
            movement = 'stay'

        action.put(movement)

        
        if cur_pos in bar:
            if z < 4:
                z += 1
            # pass
            # print("\n\n\nBAR\n\n\n")

            # Increase drunkness vector whenever we enter here

        print(z)
        if cur_pos == REWARD_LOCATION:
            print('\n\n\nHOME\n\n\n')
            sys.exit()

        if cur_pos in lake:
            pass
            # print('\n\n\nLake\n\n\n')
            # break

        x_cord_prev = x_cord
        y_cord_prev = y_cord

        # plot_beliefs(Qs, "Beliefs (Qs) at time {}".format(t))

# if __name__ == '__main__':
# 	generative_model()