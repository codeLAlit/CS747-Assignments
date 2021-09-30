import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import random
def kl(x, y):
    if x==y:
        return 0
    elif y==1 or y==0:
        return float('inf')
    elif x==0:
        return np.log2(1/(1-y))
    else: # confirm log base and correct formulation
        return x*np.log2(x/y)+(1-x)*np.log2((1-x)/(1-y))

def update_klucb_values(emp_mean, num_turns, time, c=3):
    rhs = np.log(time)+c*np.log(np.log(time))
    klucb_val = np.zeros(len(emp_mean))
    for i in range(len(emp_mean)):
        start = emp_mean[i]
        end = 1
        mid = (start+end)/2
        ukl = kl(emp_mean[i], mid)
        local_rhs = rhs/num_turns[i]
        iterations = 0
        while (end - start) > 1e-6 and iterations<=50:
            if ukl > local_rhs:
                end = mid
            else:
                start = mid
            
            mid = (start+end)/2
            ukl = num_turns[i]*kl(emp_mean[i], mid)

            if 1-mid < 1e-6:
                break
            iterations+=1
        klucb_val[i] = mid
    return klucb_val

def epsilon_greedy(means, horizon, seed, epsilon):
    num_arms = len(means)
    emp_mean = np.zeros(num_arms)
    num_turns = np.zeros(num_arms)
    rew = 0
    np.random.seed(seed)
    rewards = np.random.binomial(1, means, size=(horizon, num_arms))
    # initial pulls
    for i in range(num_arms):
        num_turns[i] +=1
        emp_mean[i] = (emp_mean[i]*(num_turns[i]-1)+rewards[i, i])/num_turns[i]
        rew += rewards[i, i]

    # algorithm
    for t in range(num_arms, horizon):
        path = np.random.binomial(1, epsilon)
        if path==1: # sample uniformly
            arm = np.random.randint(0, num_arms)            
        else: # highest empirical mean
            arm = np.argmax(emp_mean)

        num_turns[arm]+=1
        emp_mean[arm] = (emp_mean[arm]*(num_turns[arm]-1)+rewards[t, arm])/num_turns[arm]
        rew += rewards[t, arm]
    reg = horizon*np.max(means)-rew
    return reg

def ucb(means, horizon, seed):
    num_arms = len(means)
    emp_mean = np.zeros(num_arms)
    ucb_val = np.zeros(num_arms)
    num_turns = np.zeros(num_arms)
    rew = 0
    np.random.seed(seed)
    rewards = np.random.binomial(1, means, size=(horizon, num_arms))
    # initial pulls
    for i in range(num_arms):
        num_turns[i] +=1
        emp_mean[i] = (emp_mean[i]*(num_turns[i]-1)+rewards[i, i])/num_turns[i]
        rew += rewards[i, i]    
    ucb_val = emp_mean + np.sqrt(2*np.log(num_arms)/num_turns)
    # algorithm
    for t in range(num_arms, horizon):
        arm = np.argmax(ucb_val)
        num_turns[arm] += 1
        emp_mean[arm] = (emp_mean[arm]*(num_turns[arm]-1)+rewards[t, arm])/num_turns[arm]
        rew += rewards[t, arm]
        ucb_val = emp_mean + np.sqrt(2*np.log(t+1)/num_turns) # updating ucb value for all arms
    
    reg = horizon*np.max(means)-rew
    return reg

def kl_ucb(means, horizon, seed):
    num_arms = len(means)
    emp_mean = np.zeros(num_arms)
    klucb_val = np.zeros(num_arms)
    num_turns = np.zeros(num_arms)
    rew = 0
    np.random.seed(seed)
    rewards = np.random.binomial(1, means, size=(horizon, num_arms))
    # initial pulls
    for i in range(num_arms):
        num_turns[i] +=1
        emp_mean[i] = (emp_mean[i]*(num_turns[i]-1)+rewards[i, i])/num_turns[i]
        rew += rewards[i, i]
    klucb_val = update_klucb_values(emp_mean, num_turns, time=num_arms, c=3)

    for t in range(num_arms, horizon):
        arm = np.argmax(klucb_val)
        num_turns[arm] += 1
        emp_mean[arm] = (emp_mean[arm]*(num_turns[arm]-1)+rewards[t, arm])/num_turns[arm]
        rew += rewards[t, arm]
        klucb_val = update_klucb_values(emp_mean, num_turns, t+1, 3) # updating klucb value for all arms
    reg = horizon*np.max(means)-rew
    return reg

def thompson_sampling(means, horizon, seed):
    num_arms = len(means)
    emp_mean = np.zeros(num_arms)
    num_success = np.zeros(num_arms)
    num_failures = np.zeros(num_arms)
    rew = 0
    np.random.seed(seed)
    rewards = np.random.binomial(1, means, size=(horizon, num_arms))
    for t in range(horizon):
        x_as = np.random.beta(num_success+1, num_failures+1)
        arm = np.argmax(x_as)
        if rewards[t, arm] == 1:
            num_success[arm]+=1
        else:
            num_failures[arm]+=1
        rew+=rewards[t, arm]
        emp_mean = (num_success+1)/(num_success+num_failures+2)
    reg = horizon*np.max(means)-rew
    return reg

def ucb_t2(means, horizon, seed, c):
    num_arms = len(means)
    emp_mean = np.zeros(num_arms)
    ucb_val = np.zeros(num_arms)
    num_turns = np.zeros(num_arms)
    rew = 0
    np.random.seed(seed)
    rewards = np.random.binomial(1, means, size=(horizon, num_arms))
    # initial pulls
    for i in range(num_arms):
        num_turns[i] +=1
        emp_mean[i] = (emp_mean[i]*(num_turns[i]-1)+rewards[i, i])/num_turns[i]
        rew += rewards[i, i]    
    ucb_val = emp_mean + np.sqrt(c*np.log(num_arms)/num_turns)
    # algorithm
    for t in range(num_arms, horizon):
        arm = np.argmax(ucb_val)
        num_turns[arm] += 1
        emp_mean[arm] = (emp_mean[arm]*(num_turns[arm]-1)+rewards[t, arm])/num_turns[arm]
        rew += rewards[t, arm]
        ucb_val = emp_mean + np.sqrt(c*np.log(t+1)/num_turns) # updating ucb value for all arms
    
    reg = horizon*np.max(means)-rew
    return reg

def algo_t3(support, means, horizon, seed, c=0.3):
    num_rew = len(support)
    num_arms = len(means)
    rewards = np.zeros((horizon, num_arms))
    mucb_val = np.zeros(num_arms)
    num_turns_awards = np.zeros((num_arms, num_rew))
    emp_expectation = np.zeros(num_arms)
    actual_expectation = np.zeros(num_arms)
    rew = 0 
    np.random.seed(seed)
    for i in range(num_arms):
        rewards[:, i] = np.random.choice(np.arange(0, num_rew), size=horizon, p=means[i])
    # rewards[t, i] = reward for arm i at time intant t, rewards will be like 0, 1, 2,... is the index of support
    
    actual_expectation = np.sum(np.array(support)*np.array(means), axis=1)
    # initializing
    for i in range(num_arms):
        rew_idx = int(rewards[i, i])
        rew += support[rew_idx]
        num_turns_awards[i, rew_idx] += 1
        # emp_mean_rew[i, rew_idx] = (emp_mean_rew[i, rew_idx]*(num_turns_awards[i, rew_idx]-1) + support[rew_idx])/num_turns_awards[i, rew_idx]
    
    emp_expectation = np.sum(np.array(support)*num_turns_awards, axis=1)/np.sum(num_turns_awards, axis=1)
    mucb_val = emp_expectation + np.sqrt(c*np.log(num_arms)/np.sum(num_turns_awards, axis=1))
    
    for t in range(num_arms, horizon):
        arm = np.argmax(mucb_val)
        rew_idx = int(rewards[t, arm])
        rew += support[rew_idx]
        num_turns_awards[arm, rew_idx] += 1
        # emp_mean_rew[arm, rew_idx] = (emp_mean_rew[arm, rew_idx]*(num_turns_awards[arm, rew_idx]-1) + support[rew_idx])/num_turns_awards[arm, rew_idx]
        emp_expectation = np.sum(np.array(support)*num_turns_awards, axis=1)/np.sum(num_turns_awards, axis=1)
        mucb_val = emp_expectation + np.sqrt(c*np.log(t+1)/np.sum(num_turns_awards, axis=1))

    reg = horizon*np.max(actual_expectation)-rew
    return reg

def algo_t4(support, means, horizon, seed, thresh):
    num_rew = len(support)
    num_arms = len(means)
    rewards = np.zeros((horizon, num_arms))
    num_success = np.zeros(num_arms)
    num_failures = np.zeros(num_arms)
    num_highs = 0 
    np.random.seed(seed)
    for i in range(num_arms):
        rewards[:, i] = np.random.choice(np.arange(0, num_rew), size=horizon, p=means[i])
    # rewards[t, i] = reward for arm i at time intant t, rewards will be like 0, 1, 2,... is the index of support

    for t in range(horizon):
        x_as = np.random.beta(num_success+1, num_failures+1)
        arm = np.argmax(x_as)
        rew_idx = int(rewards[t, arm])
        if support[rew_idx] > thresh:
            num_success[arm] += 1
            num_highs += 1
        else:
            num_failures[arm] += 1
        
    return num_highs
