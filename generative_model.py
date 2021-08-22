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

REWARD_LOCATION = 23

print("imports loaded")
state_mapping = {}
reverse_state_mapping = {}
counter = 0
dim_x, dim_y, dim_z = 8, 3, 5

for y in range(dim_y):
    for x in range(dim_x):
        for z in range(dim_z):
            state_mapping[counter] = ((x, y, z))
            counter += 1

for y in range(len(state_mapping)):
    x, y, z = state_mapping[y]
    reverse_state_mapping[(x, y, z)] = y

def state_mapping_to_xy(state_num: int):
    return (state_mapping[state_num][0], state_mapping[state_num][1])
bar = [state_mapping_to_xy(1), state_mapping_to_xy(3), state_mapping_to_xy(12), state_mapping_to_xy(14), ]
# lake locations
lake = [state_mapping_to_xy(4), state_mapping_to_xy(5), state_mapping_to_xy(6), state_mapping_to_xy(17),
        state_mapping_to_xy(18), ]
# home location
home = state_mapping_to_xy(REWARD_LOCATION)

def plot_beliefs(Qs, title=""):
    # values = Qs.values[:, 0]
    plt.grid(zorder=0)
    plt.bar(range(Qs.shape[0]), Qs, color='r', zorder=3)
    plt.xticks(range(Qs.shape[0]))
    plt.title(title)
    plt.show()


def plot_likelihood(A):
    fig = plt.figure(figsize=(6, 6))
    ax = sns.heatmap(A, xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("Likelihood distribution (A)")
    plt.show()


def plot_empirical_prior(B):
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'STAY']
    count = 0
    for i in range(3):
        for j in range(2):
            if count >= 5:
                break

            g = sns.heatmap(B[:, :, count], cmap="OrRd", linewidth=2.5, cbar=False, ax=axes[i, j])

            g.set_title(actions[count])
            count += 1
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()


def plot_transition(B):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    a = list(actions.keys())
    count = 0
    for i in range(dim - 1):
        for j in range(dim):
            if count >= 5:
                break
            g = sns.heatmap(B[:, :, count], cmap="OrRd", linewidth=2.5, cbar=False, ax=axes[i, j], xticklabels=labels,
                            yticklabels=labels)
            g.set_title(a[count])
            count += 1
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()


class GridWorldEnv():

    def __init__(self, A, B):
        self.A = deepcopy(A)
        self.B = deepcopy(B)
        print("B:", B.shape)
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


def plot_pos(fig, axim, cur_pos, bar, lake, home):
    grid = np.zeros((3, 8))

    for position in bar:
        x_cord, y_cord = state_mapping[position]
        grid[y_cord, x_cord] = 16

    for position in lake:
        x_cord, y_cord = state_mapping[position]
        grid[y_cord, x_cord] = 5

    x_cord, y_cord = state_mapping[home]
    grid[y_cord, x_cord] = 20

    x_cord, y_cord = state_mapping[cur_pos]
    grid[y_cord, x_cord] = 13

    # fig = plt.figure(figsize = (9,9))
    # plt.imshow(grid)
    # plt.show()
    axim.set_data(grid)
    fig.canvas.flush_events()

    return x_cord


def start_generative_model(action):
    global bar
    path = Path(os.getcwd())
    print(path)
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
        if (new_x, new_y,) in bar and new_z<=4:
            P[state_index][actions['UP']] += x * y

        '''f your x-coordinate is all the way to the right (i.e. x == 2), you stay in the same place -- otherwise you move one to the right (achieved by adding 1 to your linear state index)'''
        P[state_index][actions["RIGHT"]] = state_index if x == (dim_x - 1) else state_index + 1
        new_x, new_y, new_z = state_mapping[P[state_index][actions['UP']]]
        if (new_x, new_y,) in bar and new_z<=4:
            P[state_index][actions['UP']] += x * y
        '''if your y-coordinate is all the way at the bottom (i.e. y == 2), you stay in the same place -- otherwise you move one down (achieved by adding 3 to your linear state index)'''
        P[state_index][actions['DOWN']] = state_index if y == (dim_y - 1) else state_index + dim_x
        new_x, new_y, new_z = state_mapping[P[state_index][actions['UP']]]
        if (new_x, new_y,) in bar and new_z<=4:
            P[state_index][actions['UP']] += x * y
        ''' if your x-coordinate is all the way at the left (i.e. x == 0), you stay at the same place -- otherwise, you move one to the left (achieved by subtracting 1 from your linear state index)'''
        P[state_index][actions['LEFT']] = state_index if x == 0 else state_index - 1
        new_x, new_y, new_z = state_mapping[P[state_index][actions['UP']]]
        if (new_x, new_y,) in bar and new_z<=4:
            P[state_index][actions['UP']] += x * y
        ''' Stay in the same place (self explanatory) '''
        P[state_index][actions['STAY']] = state_index
        new_x, new_y, new_z = state_mapping[P[state_index][actions['UP']]]
        if (new_x, new_y,) in bar and new_z<=4:
            P[state_index][actions['UP']] += x * y

    num_states = 120
    B = np.zeros([num_states, num_states, len(actions)])
    for s in range(num_states):
        for a in range(len(actions)):
            ns = int(P[s][a])
            x, y, z = state_mapping[ns]
            #spread some movement chance around if in a drunk state
            if z > 0:
                B[ns, s, a] = 1 - 0.1*z
                if x < dim_x-1:
                    B[ns+1, s, a] = 0.025*z
                else:
                    B[ns, s, a] +=0.025*z

                if x != 0:
                    B[ns -1, s, a] = 0.025*z
                else:
                    B[ns, s, a] += 0.025*z

                if y < dim_y-1:
                    B[ns +dim_x, s, a] = 0.025*z
                else:
                    B[ns, s, a] += 0.025*z

                if y != 0:
                    B[ns -1, s, a] = 0.025*z
                else:
                    B[ns, s, a] += 0.025*z
            else:
                B[ns, s, a] = 1

    # plot_transition(B)

    env = GridWorldEnv(A, B)

    # setup initial prior beliefs -- uncertain -- completely unknown which state it is in
    Qs = np.ones(120) * 1 / 9
    # plot_beliefs(Qs)

    # C matrix -- desires


    reward_state = state_mapping[REWARD_LOCATION]
    print(reward_state)

    C = np.zeros(num_states)
    C[REWARD_LOCATION] = 1.
    print(C)
    # plot_beliefs(C)

    # number of time steps
    T = 10

    # n_actions = env.n_control
    n_actions = 5

    # length of policies we consider
    policy_len = 4

    # this function generates all possible combinations of policies
    policies = control.construct_policies([B.shape[0]], [n_actions], policy_len)

    # reset environment
    o = env.reset()

    ##############
    # from matplotlib import animation
    # bar locations
    bar = [state_mapping_to_xy(1), state_mapping_to_xy(3), state_mapping_to_xy(12), state_mapping_to_xy(14), ]
    # lake locations
    lake = [state_mapping_to_xy(4), state_mapping_to_xy(5), state_mapping_to_xy(6), state_mapping_to_xy(17),
            state_mapping_to_xy(18), ]
    # home location
    home = state_mapping_to_xy(REWARD_LOCATION)

    # Qs = [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    Qs = [1.] + [0.] * 119
    print(Qs)
    cur_pos = list(Qs).index(1)
    # plt.ion()
    # fig = plt.figure(figsize = (9,9))
    # ax = fig.add_subplot(111)

    # grid = np.zeros((3,8))
    # grid[0,0] = 20
    # axim = ax.imshow(grid)

    # for position in bar:
    #     x_cord, y_cord = state_mapping[position]
    #     grid[y_cord,x_cord] = 16

    # for position in lake:
    #     x_cord, y_cord = state_mapping[position]
    #     grid[y_cord,x_cord] = 5

    # x_cord, y_cord = state_mapping[home]
    # grid[y_cord,x_cord] = 20

    x_cord, y_cord, z_cord = state_mapping[cur_pos]
    x_cord_prev = x_cord
    y_cord_prev = y_cord
    z_cord_prev = z_cord
    # grid[y_cord,x_cord] = 13

    # plot_pos(fig, axim, cur_pos, bar, lake, home)
    ##############

    # loop over time
    while 1:

        # infer which action to take
        a = infer_action(Qs, A, B, C, n_actions, policies)

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

        if cur_pos in bar:
            print("\n\n\nBAR\n\n\n")

        # Increase drunkness vector whenever we enter here

        if cur_pos == REWARD_LOCATION:
            print('\n\n\nHOME\n\n\n')
            sys.exit()

        if cur_pos in lake:
            print('\n\n\nLake\n\n\n')
        # break

        x_cord_prev = x_cord
        y_cord_prev = y_cord

        # plot_beliefs(Qs, "Beliefs (Qs) at time {}".format(t))



# if __name__ == '__main__':
#     generative_model()
