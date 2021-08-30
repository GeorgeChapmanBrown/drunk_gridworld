# This is needed (on my machine at least) due to weird python import issues
import os
import sys
from pathlib import Path

import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from pymdp import maths, utils
from pymdp.maths import \
    spm_log_single as log_stable  # @NOTE: we use the `spm_log_single` helper function from the `maths` sub-library of pymdp. This is a numerically stable version of np.log()
from pymdp import control

from numba import njit

REWARD_LOCATION = 23

state_mapping = {}
reverse_state_mapping = {}
counter = 0
dim_x, dim_y, dim_z = 8, 3, 5

for z in range(dim_z):
    for y in range(dim_y):
        for x in range(dim_x):
            state_mapping[counter] = ((x, y, z))
            counter += 1

for y in range(len(state_mapping)):
    x, y, z = state_mapping[y]
    reverse_state_mapping[(x, y, z)] = y

A = np.eye(9)

labels = [state_mapping[i] for i in range(A.shape[1])]

# print(f'state_mapping: {state_mapping}')

def state_mapping_to_xy(state_num: int):
    return (state_mapping[state_num][0], state_mapping[state_num][1])

# bar locations
# bar = (1,0),(3,0),(4,1),(6,1)
bar = [state_mapping_to_xy(1), state_mapping_to_xy(3), state_mapping_to_xy(12), state_mapping_to_xy(14), ]
# lake = (4,0),(5,0),(6,0),(1,2),(2,2)
lake = [state_mapping_to_xy(4), state_mapping_to_xy(5), state_mapping_to_xy(6), state_mapping_to_xy(17),
        state_mapping_to_xy(18), ]
# home location
# home = (7,2)
home = state_mapping_to_xy(REWARD_LOCATION)
# checkpoint location
# CHECKPOINT_LOCATION = [10, 20] 
checkpoint = [state_mapping_to_xy(10), state_mapping_to_xy(20), ]
checkpoint_reached_one = False
checkpoint_reached_two = False

bar_loc = np.array([1, 3, 12, 14])
lake_loc = np.array([4, 5, 6, 17])
home_loc = np.array([23])
checkpoint_loc = np.array([10, 20])

for i in range(dim_z-1):
    bar_loc = np.concatenate((bar_loc, bar_loc[i*4:(i+1)*4] + (dim_x*dim_y)))
    lake_loc = np.concatenate((lake_loc, lake_loc[i*4:(i+1)*4] + (dim_x*dim_y)))
    home_loc = np.concatenate((home_loc, home_loc[i:i+1] + (dim_x*dim_y)))
    checkpoint_loc = np.concatenate((checkpoint_loc, checkpoint_loc[i*2:(i+1)*2] + (dim_x*dim_y)))

class GridWorldEnv():

    def __init__(self, A, B):
        self.A = deepcopy(A)
        self.B = deepcopy(B)
        # print("B:", B.shape)
        self.state = np.zeros(120)
        # start at state 3
        self.state[0] = 1

    def step(self, a):
        self.state = np.dot(self.B[:, :, a], self.state)
        obs = utils.sample(np.dot(self.A, self.state))
        return obs

    def reset(self):
        self.state = np.zeros(120)
        self.state[0] = 1
        obs = utils.sample(np.dot(self.A, self.state))
        return obs


def KL_divergence(q, p):
    return np.sum(q * (log_stable(q) - log_stable(p)))


def compute_free_energy(q, A, B):
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
        Qs_pi = B[:, :, u].dot(Qs)

        # work out expected observations, given the action
        Qo_pi = A.dot(Qs_pi)

        # get entropy
        H = - (A * log_stable(A)).sum(axis=0)

        # get predicted divergence
        # divergence = np.sum(Qo_pi * (log_stable(Qo_pi) - log_stable(C)), axis=0)
        divergence = KL_divergence(Qo_pi, C)

        # compute the expected uncertainty or ambiguity 
        uncertainty = H.dot(Qs_pi)

        # increment the expected free energy counter for the policy, using the expected free energy at this timestep
        G += (divergence + uncertainty)

    return -G


# def infer_action(Qs, A, B, C, n_actions, policies):
#     # initialize the negative expected free energy
#     neg_G = np.zeros(len(policies))

#     # loop over every possible policy and compute the EFE of each policy
#     for i, policy in enumerate(policies):
#         neg_G[i] = evaluate_policy(policy, Qs, A, B, C)

#     # get distribution over policies
#     Q_pi = maths.softmax(neg_G)

#     # initialize probabilites of control states (convert from policies to actions)
#     Qu = np.zeros(n_actions)

#     # sum probabilites of control states or actions 
#     for i, policy in enumerate(policies):
#         # control state specified by policy
#         u = int(policy[0])
#         # add probability of policy
#         Qu[u] += Q_pi[i]

#     # normalize action marginal
#     utils.norm_dist(Qu)

#     # sample control from action marginal
#     u = utils.sample(Qu)

#     return u

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

def change_reward(C, z, home_loc, bar_loc, lake_loc, checkpoint_loc, checkpoint_reached_one, checkpoint_reached_two):
    if z == 0:
        for i in range(len(C)):
            if i in lake_loc or i in home_loc:
                pass
            else:
                C[i] = 1.
    # when z > 0, set home to 1, lake to z*0.05 and bar to z*0.1
    if z > 0:
        for i in range(len(C)):
            if i in lake_loc:
                pass
            else:
                C[i] = 0.1 
            if i in home_loc:
                C[i] = 1.
        if checkpoint_reached_one == False:
            for i in range(len(checkpoint_loc)):
                if i % 2 == 0:
                    C[checkpoint_loc[i]] = 1.
        if checkpoint_reached_two == False:
            for i in range(len(checkpoint_loc)):
                if i % 2 == 1:
                    C[checkpoint_loc[i]] = 1.
        
    return C

def start_generative_model(action, drunk):
    global bar, bar_loc, home, home_loc, lake, lake_loc, checkpoint, checkpoint_loc, \
         checkpoint_reached_one, checkpoint_reached_two
    path = Path(os.getcwd())
    # print(path)
    module_path = str(path.parent) + '/'
    sys.path.append(module_path)

    # A matrix
    A = np.eye(120)
    # plot_likelihood(A)

    # construct B matrix

    P = {}

    actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'STAY': 4}

    for state_index, xyz_coordinates in state_mapping.items():
        P[state_index] = {a: [] for a in range(len(actions))}
        x, y, z = xyz_coordinates
        '''if your y-coordinate is all the way at the top (i.e. y == 0), you stay in the same place -- otherwise you move one upwards (achieved by subtracting 3 from your linear state index'''
        P[state_index][actions['UP']] = state_index if y == 0 else state_index - dim_x
        new_x, new_y, new_z = state_mapping[P[state_index][actions['UP']]]
        if (new_x, new_y,) in bar and new_z < 4:
            P[state_index][actions['UP']] += dim_x * dim_y

        '''f your x-coordinate is all the way to the right (i.e. x == 2), you stay in the same place -- otherwise you move one to the right (achieved by adding 1 to your linear state index)'''
        P[state_index][actions["RIGHT"]] = state_index if x == (dim_x - 1) else state_index + 1
        new_x, new_y, new_z = state_mapping[P[state_index][actions['RIGHT']]]
        if (new_x, new_y,) in bar and new_z < 4:
            P[state_index][actions['RIGHT']] += dim_x * dim_y

        '''if your y-coordinate is all the way at the bottom (i.e. y == 2), you stay in the same place -- otherwise you move one down (achieved by adding 3 to your linear state index)'''
        P[state_index][actions['DOWN']] = state_index if y == (dim_y - 1) else state_index + dim_x
        new_x, new_y, new_z = state_mapping[P[state_index][actions['DOWN']]]
        if (new_x, new_y,) in bar and new_z < 4:
            P[state_index][actions['DOWN']] += dim_x * dim_y

        ''' if your x-coordinate is all the way at the left (i.e. x == 0), you stay at the same place -- otherwise, you move one to the left (achieved by subtracting 1 from your linear state index)'''
        P[state_index][actions['LEFT']] = state_index if x == 0 else state_index - 1
        new_x, new_y, new_z = state_mapping[P[state_index][actions['LEFT']]]
        if (new_x,new_y,) in bar and new_z < 4:
            P[state_index][actions['LEFT']] += dim_x * dim_y

        ''' Stay in the same place (self explanatory) '''
        P[state_index][actions['STAY']] = state_index
        new_x, new_y, new_z = state_mapping[P[state_index][actions['STAY']]]
        # if (new_x, new_y,) in bar and new_z < 4:
        #     P[state_index][actions['STAY']] += dim_x * dim_y

    # print(f'P: {P}')
    drunkState = state_mapping_to_xy(4)
    num_states = 120
    B = np.zeros([num_states, num_states, len(actions)])
    for s in range(num_states):
        x_cord_prev, y_cord_prev, z_cord_prev = state_mapping[s]
        for a in range(len(actions)):
            ns = int(P[s][a])
            #uncomment the else and change the if below to if 'z_cord_prev!=0' to add stagger
            if z_cord_prev<5:
                B[ns, s, a] = 1
                '''
            else:
                B[ns, s, a] = 1 - 0.1*z_cord_prev
                for new_choice in actions.values():
                     if new_choice != a:
                         print("new choice ="+str(new_choice))
                         B[int(P[s][new_choice]), s, a]=0.025*z_cord_prev
                         '''
    env = GridWorldEnv(A, B)

    # setup initial prior beliefs -- uncertain -- completely unknown which state it is in
    Qs = np.ones(120) * 1 / 120
    # plot_beliefs(Qs)

    # C matrix -- desires

    reward_state = state_mapping[REWARD_LOCATION]
    # print(reward_state)

    # print(C)
    # plot_beliefs(C)

    # number of time steps
    T = 10

    # n_actions = env.n_control
    n_actions = 5

    # length of policies we consider
    policy_len = 5

    # this function generates all possible combinations of policies
    policies = control.construct_policies([B.shape[0]], [n_actions], policy_len)

    # reset environment
    o = env.reset()

    ##############

    # Qs = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    Qs = np.array([1.] + [0.] * 119)

    cur_pos = list(Qs).index(1)

    x_cord, y_cord, z_cord = state_mapping[cur_pos]
    x_cord_prev = x_cord
    y_cord_prev = y_cord
    z_cord_prev = z_cord

    move = [0]
    # C = np.zeros(num_states)
    # C = change_reward(C, z_cord, home, bar, lake, checkpoint, checkpoint_reached_one, checkpoint_reached_two, state_mapping)

    # loop over time
    while 1:

        C = np.zeros(num_states)
        # change reward state based on drunk level
        C = change_reward(C, z_cord, home_loc, bar_loc, lake_loc, checkpoint_loc, checkpoint_reached_one, checkpoint_reached_two)
        # infer which action to take
        infer_action(Qs, A, B, C, n_actions, policies, move)
        a = move[0]
        # perform action in the environment and update the environment
        o = env.step(int(a))

        # infer new hidden state (this is the same equation as above but with PyMDP functions)
        likelihood = A[o, :]
        prior = B[:, :, int(a)].dot(Qs)

        Qs = maths.softmax(log_stable(likelihood) + log_stable(prior))
        # print(Qs.round(3))
        try:
            # print(list(Qs).index(1))
            cur_pos = list(Qs).index(1)
        # print(cur_pos)
        # x_cord = plot_pos(fig, axim, cur_pos, bar, lake, home)

        except ValueError:
            pass

        x_cord, y_cord, z_cord = state_mapping[cur_pos]

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

        # print(f'B: {B}')
        print(f'x_cord, y_cord, z_cord: {x_cord} {y_cord} {z_cord}')

        if cur_pos in bar:
            print("\n\n\nBAR\n\n\n")

        drunk.put(z_cord)
        # Increase drunkness vector whenever we enter here

        if cur_pos == home:
            print('\n\n\nHOME\n\n\n')
            sys.exit()

        if cur_pos in lake:
            print('\n\n\nLake\n\n\n')
        # break

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
#     generative_model()
