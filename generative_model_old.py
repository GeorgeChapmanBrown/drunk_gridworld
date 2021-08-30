# This is needed (on my machine at least) due to weird python import issues
from numba.np.ufunc import parallel
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

from numba import njit, jit, cuda, u1, u4, f8, i4
import random

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
    # small epsilon value for np.log to make it stable
    EPS_VAL = 1e-16

    # loop over policy
    for t in range(len(policy)):

        # get action entailed by the policy at timestep `t`
        u = int(policy[t])
        # work out expected state, given the action
        Qs_pi = B[:,:,u].dot(Qs)
        # work out expected observations, given the action
        Qo_pi = A.dot(Qs_pi)
        # get entropy
        # H = - (A * log_stable(A)).sum(axis = 0)
        H = - (A * np.log(A + EPS_VAL)).sum(axis = 0)
        # get predicted divergence
        # divergence = np.sum(Qo_pi * (log_stable(Qo_pi) - log_stable(C)), axis=0)
        divergence = np.sum(Qo_pi * (np.log(Qo_pi + EPS_VAL) - np.log(C + EPS_VAL)), axis=0)
        # divergence = KL_divergence(Qo_pi, C)
        # compute the expected uncertainty or ambiguity 
        uncertainty = H.dot(Qs_pi)
        # increment the expected free energy counter for the policy, using the expected free energy at this timestep
        G = G + (divergence + uncertainty)
    return -G

@njit
def infer_action(Qs, A, B, C, n_actions, policies, move):

    # initialize the negative expected free energy
    neg_G = np.zeros(len(policies))
    # loop over every possible policy and compute the EFE of each policy
    # This is where it slows down
    # for i, policy in enumerate(policies):
    #     neg_G[i] = evaluate_policy(policy, Qs, A, B, C)

    for i, policy in enumerate(policies):

        G = 0
        # small epsilon value for np.log to make it stable
        EPS_VAL = 1e-16

        # loop over policy
        for t in range(len(policy)):

            # get action entailed by the policy at timestep `t`
            # u = int(policy[t])
            u = policy[t][0]
            # work out expected state, given the action
            cont_B = np.ascontiguousarray(B[:,:,u])
            Qs_pi = np.dot(cont_B, Qs)
            # Qs_pi = B[:,:,u].dot(Qs)
            # work out expected observations, given the action
            Qo_pi = np.dot(A, Qs_pi)
            # Qo_pi = A.dot(Qs_pi)
            # get entropy
            # H = - (A * log_stable(A)).sum(axis = 0)
            H = - np.sum(A * np.log(A + EPS_VAL), axis=0)
            # get predicted divergence
            # divergence = np.sum(Qo_pi * (log_stable(Qo_pi) - log_stable(C)), axis=0)
            divergence = np.sum(Qo_pi * (np.log(Qo_pi + EPS_VAL) - np.log(C + EPS_VAL)))
            # divergence = KL_divergence(Qo_pi, C)
            # compute the expected uncertainty or ambiguity 
            uncertainty = np.dot(H, Qs_pi)
            # uncertainty = H.dot(Qs_pi)
            # increment the expected free energy counter for the policy, using the expected free energy at this timestep
            G = G + (divergence + uncertainty)

        neg_G[i] = -G

    # get distribution over policies
    # Q_pi = maths.softmax(neg_G)

    # output = neg_G - neg_G.max(axis=0)
    # output = np.exp(output)
    # Q_pi = output / np.sum(output, axis=0)
    Q_pi = np.exp(neg_G) / np.sum(np.exp(neg_G))

    # initialize probabilites of control states (convert from policies to actions)
    Qu = np.zeros(n_actions)

    # sum probabilites of control states or actions 
    for i, policy in enumerate(policies):
        # control state specified by policy
        # u = int(policy[0])
        u = policy[t][0]
        # add probability of policy
        Qu[u] += Q_pi[i]

    # normalize action marginal
    # utils.norm_dist(Qu)

    # if Qu.ndim == 3:
    #     new_dist = np.zeros_like(Qu)
    #     for c in range(Qu.shape[2]):
    #         new_dist[:, :, c] = np.divide(Qu[:, :, c], np.sum(Qu[:, :, c], axis=0))
    #     Qu = new_dist
    # else:
    Qu = np.divide(Qu, np.sum(Qu, axis=0))

    # sample control from action marginal
    # u = utils.sample(Qu)
    sample_onehot = np.random.multinomial(1, Qu)
    move[0] = np.where(sample_onehot == 1)[0][0]

    # return u

def change_reward(C, z, home, bar, lake, checkpoint, checkpoint_reached_one, checkpoint_reached_two):
    # when z = 0, set everything but lake and home to reward 0.5
    if z == 0:
        for i in range(len(C)):
            if i in lake or i == home:
                pass
            else:
                C[i] = 1.
    # when z > 0, set home to 1, lake to z*0.05 and bar to z*0.1
    if z > 0:
        for i in range(len(C)):
            if i in lake:
                pass
            else:
                C[i] = 0.1 
        C[home] = 1.
        if checkpoint_reached_one == False:
            C[checkpoint[0]] = 1.
        if checkpoint_reached_two == False:
            C[checkpoint[1]] = 1.
        
    # print(C)
    return C

def randomizer(z):
    return random.randint(1,5) > 5-z

def random_move():
    return random.randint(0,4)

def drunk_movement(z, a):
    result = randomizer(z)

    if result is True:
        #move to random (0,4)
        stagger = random_move()
    
    else:
        stagger = a

    return stagger

def start_generative_model(action, drunk):

    w = 1.
    # print(w)
    path = Path(os.getcwd())
    # print(path)
    module_path = str(path.parent) + '/'
    sys.path.append(module_path)

    REWARD_LOCATION = 23
    CHECKPOINT_LOCATION = [10, 20] 

    # bar locations
    bar = [1, 3, 12, 14]
    # lake locations
    lake = [4, 5, 6, 17, 18]
    # home location
    home = REWARD_LOCATION
    # checkpoint location
    checkpoint = CHECKPOINT_LOCATION
    checkpoint_reached_one = False
    checkpoint_reached_two = False
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

    # fig, axes = plt.subplots(2,3, figsize = (15,8))
    # a = list(actions.keys())
    # count = 0
    # for i in range(dim_x-1):
    #     for j in range(dim_y):
    #         if count >= 5:
    #             break 
    #         g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
    #         g.set_title(a[count])
    #         count +=1 
    # fig.delaxes(axes.flatten()[5])
    # plt.tight_layout()
    # plt.show()


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
    policy_len = 6

    # this function generates all possible combinations of policies
    policies = control.construct_policies([B.shape[0]], [n_actions], policy_len)
    # reset environment
    o = env.reset()

    ##############
    # from matplotlib import animation

    Qs = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    cur_pos = list(Qs).index(1)

    x_cord, y_cord = state_mapping[cur_pos]
    x_cord_prev = x_cord
    y_cord_prev = y_cord
    prev_pos = cur_pos

    ##############

    move = [0]
    from timeit import default_timer as timer

    C = np.zeros(num_states)
        # change reward state based on drunk level
    C = change_reward(C, z, home, bar, lake, checkpoint, checkpoint_reached_one, checkpoint_reached_two)
    # start = timer()
    infer_action(Qs, A, B, C, n_actions, policies, move)
    # print(timer()-start)

    # loop over time
    while 1:

        EPS_VAL = 1e-16

        C = np.zeros(num_states)
        # change reward state based on drunk level
        C = change_reward(C, z, home, bar, lake, checkpoint, checkpoint_reached_one, checkpoint_reached_two)
        # infer which action to take
        # start = timer()
        infer_action(Qs, A, B, C, n_actions, policies, move)
        # infer_action_gpu(Qs, A, B, C, n_actions, policies, move)
        # print(timer()-start)
        a = move[0]
        print(f'a: {a}')
        # perform action in the environment and update the environment

        # do stagger calculation
        if z > 0:
            a = drunk_movement(z, a)
            print(f'a_stagger: {a}')

        o = env.step(int(a))

        # infer new hidden state (this is the same equation as above but with PyMDP functions)
        likelihood = A[o,:]
        prior = B[:,:,int(a)].dot(Qs)

        # Qs = maths.softmax(log_stable(likelihood) + log_stable(prior))
        log_ = np.log(likelihood + EPS_VAL) + np.log(prior + EPS_VAL)
        Qs = np.exp(log_) / np.sum(np.exp(log_))
        # print(Qs.round(3))
      
        # print(list(Qs).index(1))
        # cur_pos = list(Qs).index(1)
        cur_pos = (np.where(Qs.round(3) == 1))[0][0]
        # print(cur_pos)
        # x_cord = plot_pos(fig, axim, cur_pos, bar, lake, home)

        x_cord, y_cord = state_mapping[cur_pos]

        print(f'(x_cord, y_cord, z): ({x_cord}, {y_cord}, {z})')

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

            for i, bar_loc in enumerate(bar):
                if cur_pos == bar_loc:
                    print(bar_loc)
                    bar = np.delete(bar, i)
                else:
                    pass
            print(bar)

            # pass
            # print("\n\n\nBAR\n\n\n")

            # Increase drunkness vector whenever we enter here

        drunk.put(z)
        
        # print(z)
        if cur_pos == home:
            print('\n\n\nHOME\n\n\n')
            sys.exit()

        if cur_pos in lake:
            # time.sleep(100000)
            print('\n\n\nLake\n\n\n')
            sys.exit()
        
        if cur_pos == checkpoint[0]:
            checkpoint_reached_one = True

        if cur_pos == checkpoint[1]:
            checkpoint_reached_two = True

        if checkpoint_reached_one == True and x_cord < 2:
            checkpoint_reached_one = False

        if checkpoint_reached_two == True and x_cord < 4:
            checkpoint_reached_two = False

        x_cord_prev = x_cord
        y_cord_prev = y_cord

        # plot_beliefs(Qs, "Beliefs (Qs) at time {}".format(t))

# if __name__ == '__main__':
# 	generative_model()