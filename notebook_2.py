#!/usr/bin/env python
# coding: utf-8

# # Tutorial Notebook 2. Inference and Planning
# 
# 
# In this notebook, we will continue on from the last notebook to build a fully fledged active inference agent capable of performing inference and planning using Active Inference in the simple grid-world environment. We will also begin to use some aspects of PyMDP although this will mostly be utility functions while we will implement the core functionality of the agent ourselves.
# 

# First, we simply start out by defining our generative model as we did last time.

# ## Add `pymdp` module

# In[1]:


# This is needed (on my machine at least) due to weird python import issues
import os
import sys
from pathlib import Path
path = Path(os.getcwd())
print(path)
module_path = str(path.parent.parent) + '/'
sys.path.append(module_path)


# ## Imports

# In[2]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

from pymdp.distributions import Categorical
import pymdp.core.maths as F
import pymdp.core.control as control
print("imports loaded")


# ## Plotting

# In[3]:


state_mapping = {0: (0,0), 1: (1,0), 2: (2,0), 3: (0,1), 4: (1,1), 5:(2,1), 6: (0,2), 7:(1,2), 8:(2,2)}

A = np.eye(9)
def plot_beliefs(Qs, title=""):
    #values = Qs.values[:, 0]
    plt.grid(zorder=0)
    plt.bar(range(Qs.shape[0]), Qs, color='r', zorder=3)
    plt.xticks(range(Qs.shape[0]))
    plt.title(title)
    plt.show()
    
labels = [state_mapping[i] for i in range(A.shape[1])]
def plot_likelihood(A):
    fig = plt.figure(figsize = (6,6))
    ax = sns.heatmap(A, xticklabels = labels, yticklabels = labels, cbar = False)
    plt.title("Likelihood distribution (A)")
    plt.show()
    
def plot_empirical_prior(B):
    fig, axes = plt.subplots(3,2, figsize=(8, 10))
    actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'STAY']
    count = 0
    for i in range(3):
        for j in range(2):
            if count >= 5:
                break
                
            g = sns.heatmap(B[count].values, cmap="OrRd", linewidth=2.5, cbar=False, ax=axes[i,j])
            g.set_title(actions[count])
            count += 1
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()
    
def plot_transition(B):
    fig, axes = plt.subplots(2,3, figsize = (15,8))
    a = list(actions.keys())
    count = 0
    for i in range(dim-1):
        for j in range(dim):
            if count >= 5:
                break 
            g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
            g.set_title(a[count])
            count +=1 
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()


# ## Generative model
# 
# Here, we setup our generative model which is the same as in the last notebook. This is formed of a likelihood distribution $P(o_t|s_t)$, denoted `A`, and a empirical prior (transition) distribution $P(s_t|s_{t-1},a_{t-1})$, denoted `B`.
# 
# Since this was covered in more detail in the previous tutorial, we quickly skip over the details here
# 

# In[4]:


# A matrix
A = np.eye(9)
plot_likelihood(A)


# In[5]:


# construct B matrix

P = {}
dim = 3
actions = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'STAY':4}
for s in state_mapping.keys():
    P[s] = {a : [] for a in range(len(actions))}
    x, y = state_mapping[s][0], state_mapping[s][1]

    P[s][actions['UP']] = s if y == 0 else s - dim
    P[s][actions["RIGHT"]] = s if x == (dim -1) else s+1
    P[s][actions['DOWN']] = s if y == (dim -1) else s + dim 
    P[s][actions['LEFT']] = s if x == 0 else s -1 
    P[s][actions['STAY']] = s



num_states = 9
B = np.zeros([num_states, num_states, len(actions)])
for s in range(num_states):
    for a in range(len(actions)):
        ns = int(P[s][a])
        B[ns, s, a] = 1

plot_transition(B)


# # Create Environment Class

# To make things simple we will parcel up the $A$ and $B$ matrices into a class which represents the environment. The environment has two functions `step` which when given an action will update the environment a single step and `reset` which resets the environment back to its initial condition. The API of our simple environment class is similar to the `Env` base class used by PyMDP, although the PyMDP version has many more features than we use here.

# In[6]:


class GridWorldEnv():
    
    def __init__(self,A,B):
        self.A = deepcopy(A)
        self.B = deepcopy(B)
        print("B:", B.shape)
        self.state = np.zeros(9)
        # start at state 3
        self.state[2] = 1
    
    def step(self,a):
        self.state = np.dot(self.B[:,:,a], self.state)
        return self.state
    def reset(self):
        self.state =np.zeros(9)
        self.state[2] =1 
    
env = GridWorldEnv(A,B)


# # Inference

# Now that we have the generative model setup, we turn to the behaviour of the active inference agent itself. To recap, we assume that this agent receives observations from the environment and can emit actions. Moreover, we assume that this agent has some kind of goal or preferences over the state of the environment it wants to create, and will choose actions so as to make this state occur. For now, however we will not deal with the problem of action selection but only of inference.
# 
# The agent receives observations $o_t$ from the environment but does not naturally know the environments true state $x_t$, which would be useful to know. Thus, the agent must *infer* this state by computing the posterior distribution $p(x_t | o_t)$. It can do this by Bayesian inference using Bayes rule but, as we discussed last time, explicitly computing Bayes rule is often intractable because the marginal likelihood requires the averaging over an infinite number of hypotheses. We therefore need some other way to compute or approximate this posterior. Active inference assumes that this posterior can be approximated through a family of methods called *variational inference* which only approximate the posterior, but are fast and computationally efficient.

# # Variational Inference
# 

# Variational inference is a set of inference methods which can rapidly and efficiently compute *approximate* posteriors for Bayesian inference problems. The key idea behind variational inference is that instead of trying to compute the true posterior $p(x_t | o_t)$ which may be extremely complex, is that we instead optimize an *approximate posterior*. Specifically, we will define another distribution $q(x_t | o_t; \phi)$ which has some parameters $\phi$ which we then optimize so as to make $q(x_t | o_t; \phi)$ as close as possible to the true distribution. Typically, we choose this $q$ distribution to be some simple distribution which is easy to work with mathematically. If the process works, then we can get the $q$ distribution very close to the true posterior, and as such get a good estimate of the posterior without ever explicitly computing it using Bayes rule. 

# Mathematically, we can do this by setting up an optimization problem. We have the true posterior $p(x_t | o_t)$, which is unknown, and we have our $q$ distribution which we do know. We then want to optimize the $q$ distribution to make it as *close as possible* to the true posterior $p(x_t | o_t)$. To do this, we first need a way to quantify *how close* two probability distributions are. The way we do this is by using a quantity known as the *Kullback-Leibler (KL) divergence*. This is a metric derived from information theory which lets us quantify the distance (in bits) of two distributions. The KL divergence between two distributions $q(x)$ and $p(x)$ is defined as,
# $$
# \begin{align}
# KL[q(x) || p(x)] = \sum_x q(x) (\ln q(x) - \ln p(x))
# \end{align}
# $$
# Mathematically it can be thought of as the average of the difference of the logarithms of the probabilities assigned by $q$ and $p$ to the states $x$. The KL divergence is smallest when $q(x) = p(x)$ when it is equal to 0, and can grow to be infinitely large which happens wherever $q(x)$ assigns a nonzero probability but $p(x)$ doesn't. In code, we can compute the KL divergence as:
# 

# In[7]:


def KL_divergence(q,p):
    return np.sum(q * (np.log(q) - np.log(p)))


# Now that we know about the KL divergence, we can express our variational problem of making our approximate posterior $q(x_t | o_t)$ as close as possible to the true posterior as simply minimizing the KL divergence between the two distributions. That is, we can define the optimal approximate distribution as,
# $$
# \begin{align}
# q^*(x_t | o_t) = min \, \, KL[q(x_t | o_t) || p(x_t | o_t)]
# \end{align}
# $$
# 
# And then simply try to optimize this objective so as to find the $q(x_t | o_t)$ distribution which minimzes this KL divergence. The trouble with this is that our objective actually explicitly contains the true posterior in it and so, since we can't conpute the true posterior, we can't compute this objective either -- so we are stuck!
# 
# Variational inference provides a clever way to get around this problem by instead minimizing an *upper bound* on this divergence called the *variational free energy*. Importantly this bound is computable so we can actually optimize it and moreover since it is an upper bound, if we minimize it, we can make $q$ as close as possible to the real bound, thus still managing to obtain a good approximate posterior distribution. Deriving the variational free energy is very simple, we first take our initial objective and apply Bayes rule to the true posterior, and then take out the marginal likelihood term separately
# $$
# \begin{align}
#  KL[q(x_t | o_t) || p(x_t |  o_t)] &= KL[q(x_t | o_t) || \frac{p(o_t,x_t)}{p(o_t)}] \\
#  &= KL[q(x_t | o_t) || p(o_t, x_t)] + \sum_x q(x_t | o_t) \ln p(o_t) \\
#  &= KL[q(x_t | o_t) || p(o_t, x_t)] +  \ln p(o_t) 
# \end{align}
# $$
# 
# Where in the final line we have used the fact that the sum is over a different variable than the distribution, and the sum of a probability distribution is $1$ -- i.e $\sum_x q(x_t | o_t) \ln p(o_t) = \ln p(o_t) * \sum_x q(x_t | o_t)$ and $\sum_x q(x_t | o_t) = 1$. Specifically, since $ \ln p(o_t)$ is the log of a probability distribution it is always negative, since the probability of a state is always between 0 and 1. This means that we know that this term we have devised $KL[q(x_t | o_t) || p(o_t, x_t)]$ is always necessarily greater than our original divergence between the approximate posterior and the true posterior, so it is an *upper bound*. We call this term the *variational free energy* and denote it by $\mathcal{F}$.
# $$
# \begin{align}
# \mathcal{F} = KL[q(x_t | o_t) || p(o_t, x_t)]
# \end{align}
# $$
# 
# The free energy here is simply the divergence between the approximate posterior and the *generative model* of the agent. Since we know both the approximate posterior (as we defined it in the first place!) and the generative model, then both terms of this divergence are computable. We thus have our algorithm to approximate the posterior! Since the free energy is an upper bound, if we minimize the free energy, we also implicitly minimize the true divergence between the true and approximate posteriors, which will force the approximate posterior to be close to the true posterior and thus a good approximation! Moreover, since we can compute the free energy, we can actually perform this optimization! 
# 
# In many cases, we typically perform variational inference by taking the gradients of the free energy and then doing gradient descent on the parameters defining $q(x_t | o_t)$. However, when the distributions are discrete, the parameters of the approximate distribution are simply the probability values for each state, and we can actually solve this optimization problem directly to perform inference in a single step instead of as a gradient descent.
# 

# # Directly solving variational inference in the discrete case

# To recap, remember that we have turned the problem of computing the posterior distribution $p(x_t | o_t)$ into that of minimizing the variational free energy: $\mathcal{F} = KL[q(x_t | o_t) || p(o_t, x_t)]$ with respect to an approximate posterior distribution $q(x_t | o_t)$. 
# 
# The optimal distribution $q^*(x_t | o_t)$ is simply the minimum of the KL divergence. Now, remember from high-school calculus that we can explicitly compute the minimum of a function by taking its derivative and setting it to 0 (i.e. at the minimum the first derivative of the function is 0) (if you don't remember this from calculus, trust me on this). This means that to solve this problem all we need to do is take the derivative of the free energy and set it to 0 and rearrange. First, let's write out the free energy explicitly.
# 
# $$
# \begin{align}
# \mathcal{F} &= KL[q(x_t | o_t) || p(o_t, x_t)] \\
# &= \sum_x q(x_t | o_t) (\ln q(x_t | o_t) - \ln p(o_t,x_t))
# \end{align}
# $$
# If we then split the generative model up into a prior and posterior, we can write it as,
# 
# $$
# \begin{align}
# \mathcal{F} = \sum_x q(x_t | o_t) \big[ \ln q(x_t | o_t) - \ln p(o_t | x_t)  - \ln p(x_t | x_{t-1}, a_{t-1}) \big]
# \end{align}
# $$
# Then, recalling that we have explicitly defined the likelihood and transition matrices as the $\textbf{A}$ and $\textbf{B}$ matrices. We also explicitly define our approximate posterior beliefs $q(x_t | o_t)$ as a *vector* of probabilities which we denote as $\textbf{q} = [q_1, q_2, q_3 \dots]$. With this all defined, we can write out the free energy as,
# 
# $$
# \begin{align}
# \mathcal{F} = \sum_x \textbf{q} * \big[ \ln \textbf{q} - \ln \textbf{A}  - \ln \textbf{B} \big]
# \end{align}
# $$
# 
# 
# And for fun, we can explicitly compute it in code:

# In[8]:


def compute_free_energy(q,A, B):
    return np.sum(q * (np.log(q) - np.log(A) - np.log(B)))


# Then, all we need to do is take the derivative of the free energy with respect to the approximate posterior distribution which is as follows,
# $$
# \begin{align}
# \frac{\partial \mathcal{F}}{\partial \textbf{q}} = \ln \textbf{q} - \ln \textbf{A} - \ln \textbf{B} - \textbf{1}
# \end{align}
# $$
# 
# Where $\textbf{1}$ is just a vector of ones of equal length to $\textbf{q}$ and comes from the $q \frac{\partial \ln q}{\partial q} = q * \frac{1}{q} = 1$. Thus, if we set this derivative to 0 and rearrange, we can get,
# $$
# \begin{align}
# 0 &= \ln \textbf{q} - \ln \textbf{A} - \ln \textbf{B} - \textbf{1} \\
# &\implies \textbf{q}^* = \sigma(\ln \textbf{A} + \ln \textbf{B})
# \end{align}
# $$
# 
# Where $\sigma$ is a softmax function $\sigma(x) = \frac{e^x}{\sum_x e^x}$ which ensures that the resulting probability distribution is normalized. This expression lets us compute the optimal approximate posterior instantly as a straightforward function of $\textbf{A}$ and $\textbf{B}$. We can thus quickly right the code for inference:

# In[9]:


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def perform_inference(A, B):
    return softmax(np.log(A) + np.log(B))


# This means that inference in the discrete state space setting is very simple. All we need to do is have some initial set of beliefs $\textbf{q}_0$ and then update them according to these rules over time using the $\textbf{A}$ and $\textbf{B}$ matrices in our generative model. A slight subtlety is that if have just received an observation, the only part of the $\textbf{A}$ matrix that matters is the column corresponding to that observation. Similarly, if we know the action that we took in the last timestep, then we only need to use the section of the $\textbf{B}$ matrix corresponding to that action.

# # Planning through Active Inference

# So far we just have an agent that can perform perform inference in the discrete state space, but inference by itself isn't really that useful. Instead what we really want to do is *planning*. That is, the agent needs to be able to figure out how to emit a series of actions which will take it to a certain goal state. A key part of Active Inference is that this process of planning, or more broadly action selection, can also be phrased and solved as a process of variational inference. This is why it is called *Active* Inference, after all. 
# 
# However, when starting out thinking about this, it is not immediately obvious how to turn planning into an inference problem. What are the hypotheses? What are the observations? To turn the problem of planning into an inference problem, we need to introduce two additional concepts. The first is the idea of a *goal state*. To make planning useful, the agent has to *want something*. This is different from just performing objective inference about the state of the world in which there is no goal except to infer correctly. In Active Inference we define the goal state as a separate goal vector denoted $\textbf{C}$. When performing planning we then modify the generative model of the agent so that it no longer reflects the true distribution of observations in the environment, but rather includes the goal vector. We denote this new generative model $\tilde{p}(o_t, x_t) = p(o_t | x_t)\tilde{p}(x_t)$ where we use $\tilde{p}$ to say that this distribution is not a true distribution describing the agents model of the world, but is instead a *desired distribution*. Here, we set the desired distribution to be over the state of the environment $x_t$ such that $\tilde{p}(x_t) = \textbf{C}$.
# 
# By changing the generative model in this way, we have effectively changed the inference problem from: *infer the most likely states and actions given the true generative model of the world* to *infer the most likely states and actions given a false model of the world which says that I will achieve my goals*. Perhaps more intuitively, we can think of this inference problem as saying: *Given that I have achieved my goals, infer what actions I must have taken*.
# 
# The second thing we need to do to perform planning is to also extend the inference problem to *actions in the future*, since these are the fundamental things that the agent controls which it can use to adjust the environment. We call a sequence of future actions from now (time $t$) until some set future time $T$ a *policy* and denote it $\pi = [a_t, a_{t+1}, a_{t+2} \dots a_T]$. The goal is then to infer the optimal policy $\pi^*$ given the goal state $\textbf{C}$. 
# 
# However here there is a problem. Typically we would perform variational inference to solve this inference problem, but the variational free energy is not defined over future trajectories of observations which are uncertain. Instead, we define a new objective which can handle this -- the *Expected Free Energy (EFE)* which we denote as $\mathcal{G}$. We define the EFE as,
# $$
# \begin{align}
# \mathcal{G}_t = \sum_{o_t, x_t} q(x_t | o_t)q(o_t) \big[ \ln q(x_t | o_t) - \ln \tilde{p}(o_t, x_t) \big]
# \end{align}
# $$
# 
# The expected free energy is defined for a single time-step of a trajectory *in the future* where the observation is unknown.  The key difference between the standard variational free energy and the expected free energy is that the expected free energy also averages over the *expected observations* $q(o_t)$ which is necessary since with the expected free energy we are evaluating possible future trajectories without a given observation, unlike the variational free energy where we can assume that we have already received the observation.
# 
# To get an intuitive handle on what the expected free energy *means* we can decompose it into two more intuitive quantities.
# 
# $$
# \begin{align}
# \mathcal{G}_t &= \sum_{o_t, x_t} q(x_t | o_t)q(o_t) \big[ \ln q(x_t | o_t) - \ln \tilde{p}(o_t, x_t) \big] \\
# &= -\underbrace{\sum_{o_t, x_t} q(x_t | o_t)q(o_t) \big[ \ln p(o_t | x_t) \big]}_{\text{Uncertainty}} + \underbrace{KL[q(x_t | o_t) || \tilde{p}(x_t) ]}_{\text{Divergence}}
# \end{align}
# $$
# 
# The first term is called *uncertainty* or sometimes *novelty* and represents essentially the spread of the observations expected in the future. Since we are minimizing the expected free energy essentially we want to take actions that *maximize* novelty. We can think of this as a bonus to aid exploration since active inference agents will preferentially pursue uncertainty (with the hope of ultimately being able to resolve it). The second term is the *divergence* term which is the KL divergence between the approximate beliefs of the state and the goal distribution. Effectively this term scores how far away the agent expects it will be from the goal if it follows that specific policy. Since we are minimizing the expected free energy this term is positive and so also minimized -- that is by minimizing the expected free energy, we are trying to choose trajectories which will bring the expected beliefs in the future close to the desired states, which makes sense since we want to reach these desired states.
# 
# Now that we have the expected free energy to score possible trajectories, we now need to infer the optimal policy. A simple approach (which can be derived explicitly although it is somewhat complex) is to say that the posterior probability of a policy is proportional to the (exponentiated) sum of the expected free energy accumulated along the trajectory through the environment created by that policy. While this sounds complex, mathematically we can express it very simply as,
# $$
# \begin{align}
# q(\pi) = \sigma( \sum_t^T \mathcal{G}_t)
# \end{align}
# $$
# We can then choose which policy we implement for the next timestep by just sampling a policy from $q(\pi)$ and then emitting the first action of that policy.
# 
# While all this may seem exceptionally long and complex, it results in an algorithm which is actually remarkably simple. The algorithm is:
# 
# 1.) There is an agent with a generative model of the environment ($\textbf{A}$ and $\textbf{B}$ matrices), some initial set of approximate posterior beliefs $q(x_t | o_t)$ and a desired state vector $\textbf{C}$. 
# 
# 2.) The agent receives an observation $o_t$ and computes its posterior beliefs as we did earlier by minimizing the free energy.
# 
# 3.) The agent now needs to choose what action to make to achieve its goals. It does this by:
# 
#  3.1.) First creating a set of potential policies to evaluate.
#     
#    3.2.) For each policy in this set, use the generative model to simulate the agent's trajectory in the environment *as if* it had emitted the actions prescribed by the policy
#     
#    3.3.) For each future timestep of each future trajectory, compute the expected free energy of that time-step
#     
#    3.4.) Sum the expected free energies for each timestep of each trajectory to get a total expected free energy for each possible policy.
#     
#    3.5.) Use these total expected free energies to compute the posterior distribution $q(\pi)$ as done above and sample a policy from it.
#     
# 4.) Execute the sampled policy and go back to step 1.
# 
# 

# And that's it! We're done. We have the full algorithm to create an active inference agent. Now all we do is show how to translate this algorithm into code for our specific case.

# ## Beliefs
# 
# First we need to setup an initial belief distribution which we will then update according to the observations we will receive.

# In[10]:


# setup initial prior beliefs -- uncertain -- completely unknown which state it is in
Qs = np.ones(9) * 1/9
plot_beliefs(Qs)


# # Preferences

# Now we have to encode the agent's preferences, so that it can learn to go to its reward state. In the current context, the agent wants (i.e. expects) to be in the reward location 7.

# In[11]:


# C matrix -- desires

REWARD_LOCATION = 7
reward_state = state_mapping[REWARD_LOCATION]
print(reward_state)

C = np.zeros(num_states)
C[REWARD_LOCATION] = 1. 
print(C)
plot_beliefs(C)


# The C matrix is a 1x9 matrix, where each value represents the preference to occupy a given state. We will create a one-hot C matrix, so that the agent only has a preference to be in state 7. 

# # Implementing the Active Inference Agent

# First, as a minor technical note. We simply convert our original numpy distributions into the PyMDP `Categorical` class. This class contains many utility functions for handling categorical distributions which will make our lives easier.

# In[12]:


A = Categorical(values = A)
B = np.array([B[:,:,i] for i in range(5)],dtype="object")
B = Categorical(values = B)
C = Categorical(values = C)
Qs = Categorical(values = Qs)


# ## Evaluate policy 
# 
# This helper function we evaluate the negative expected free energy for a given policy $-\mathcal{G}(\pi)$ All we do is loop through all the timesteps of the policy, simulate what the environment will do (using our generative model) given the actions emitted by the policy and then compute the expected free energy of our beliefs about the environment state and the potential observations returned by the environment according to our decomposition of the expected free energy.
# 

# In[13]:


def evaluate_policy(policy, Qs, A, B, C):
    # initialize expected free energy at 0
    G = 0
    
    # create copy of our state
    Qs = Qs.copy()
        
    # loop over policy
    for t in range(len(policy)):

        # get action
        u = int(policy[t])

        # work out expected state
        Qs = B[u].dot(Qs)

        # work out expected observations
        Qo = A.dot(Qs)

        # get entropy
        H = A.entropy()

        # get predicted divergence and uncertainty and novelty
        divergence = F.kl_divergence(Qo, C)
        uncertainty = H.dot(Qs)[0, 0]
        # increment the expected free energy counter.
        G += (divergence + uncertainty)

    return -G


# ## Infer action
# 
# This helper function will infer the most likely action. Specifically, it computes steps 3.1 to 3.5 in the active inference algorithm. First, it constructs all possible policies for a given policy length and set of actions. Then it loops through every possible policy and computes the expected free energy of that policy using our previous function, and then computing the policy distribution $q(\pi)$ using the softmax over the expected free energies.

# In[14]:


def infer_action(Qs, A, B, C, n_actions, policy_len):
    
    # this function generates all possible combinations of policies
    policies = control.construct_policies([9], [5], policy_len)
    n_policies = len(policies)

    # initialize the negative expected free energy
    neg_G = np.zeros([n_policies, 1])

    # loop over every possible policy and compute the EFE of each policy
    for i, policy in enumerate(policies):
        neg_G[i] = evaluate_policy(policy, Qs, A, B, C)

    # get distribution over policies
    Q_pi = F.softmax(neg_G)

    # initialize probabilites of control states (convert from policies to actions
    Qu = Categorical(dims=n_actions)

    # sum probabilites of controls 
    for i, policy in enumerate(policies):
        # control state specified by policy
        u = int(policy[0])
        # add probability of policy
        Qu[u] += Q_pi[i]

    # normalize
    Qu.normalize()

    # sample control
    u = Qu.sample()

    return u


# ## Main loop

# Here we simply implement the main loop of the active inference agent interacting with the environment. Specifically, this essentially implements steps 1-5 of the MDP "program" we discussed in notebook 1. Specifically, for each timestep, the agent infers an action, it emits that action to the environment, the environment is updated and returns an observation to the agent. The agent then infers the new state of the environment given that observation.

# In[15]:


# number of time steps
T = 10

#n_actions = env.n_control
n_actions = 5

# length of policies we consider
policy_len = 4

# reset environment
o = env.reset()

# loop over time
for t in range(T):

    # infer which action to take
    a = infer_action(Qs, A, B, C, n_actions, policy_len)
    a = int(a)
    
    # perform action in the environment and update the environment
    o = env.step(a)
    o_idx = int(np.argmax(o))
    
    # infer new hidden state (this is the same equation as above but with PyMDP functions)
    Qs = F.softmax(A[o_idx,:].log() + B[a].dot(Qs).log())
    
    print(Qs)
    plot_beliefs(Qs[:,0], "Beliefs (Qs) at time {}".format(t))


# And that's it! In the last two notebooks, we have implemented a basic active inference agent which can successfully navigate around a 3x3 gridworld using active inference. Moreover, we have implemented this all from scratch using mostly basic numpy functions and not using much of the functionality of PyMDP. 
# 
# Hopefully after going through this you now understand roughly what active inference is and how it works, as well as ideally have some intuitions about how inference as well as policy selection work "under the hood", as well as learnt a lot about Bayesian and specifically variational inference. In the next notebook, we will focus more on the PyMDP library itself and demonstrate how PyMDP provides a useful set of abstractions to allow us to easily create active inference agent as well as perform inference and policy selection in considerably more complex environments than the one given here. We will discuss the high level structure of the library and show how its possible to replicate these notebooks in a much smaller amount of code using the PyMDP abstractions.

# In[ ]:




