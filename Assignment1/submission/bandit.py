import argparse

import matplotlib.pyplot as plt
import numpy as np

from banditAlgos import algo_t3, algo_t4, epsilon_greedy, kl_ucb, thompson_sampling, ucb, ucb_t2

available_algos = ['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1', 'ucb-t2', 'alg-t3', 'alg-t4']

def parse_means_normal(instance):
    data = None
    with open(instance, 'r') as f:
        data = f.read()
        data = data.split('\n')
        if data[-1]=='':
            data = data[:-1]
    
    data = [float(i) for i in data]
    return data

def parse_means_fixed(instance):
    rewards = None
    probs = None
    with open(instance, 'r') as f:
        data = f.read()
        data = data.split('\n')
        rewards = data[0]
        probs = data[1:]

    rewards = [float(i) for i in rewards.split()]
    probs = [[float(i) for i in a.split()] for a in probs]
    return rewards, probs
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', '-in', type=str, required=True)
    parser.add_argument('--algorithm', '-al', type=str, required=True)
    parser.add_argument('--randomSeed', '-rs', type=int, required=True)
    parser.add_argument('--epsilon', '-ep', type=float, required=False, default=0.02)
    parser.add_argument('--scale', '-c', type=float, required=False, default=2)
    parser.add_argument('--threshold', '-th', type=float, required=False, default=0)
    parser.add_argument('--horizon', '-hz', type=int, required=True)
    args = parser.parse_args()
    
    instance = args.instance
    algorithm = args.algorithm
    randomSeed = args.randomSeed
    epsilon = args.epsilon
    scale = args.scale
    threshold = args.threshold
    horizon = args.horizon

    REG = 0
    HIGH = 0

    if algorithm not in ['alg-t3', 'alg-t4']:
        arm_means = parse_means_normal(instance)
    else:
        support, arm_means = parse_means_fixed(instance)
    
    if algorithm == available_algos[0]:
        REG = epsilon_greedy(arm_means, horizon, randomSeed, epsilon)
    elif algorithm == available_algos[1]:
        REG = ucb(arm_means, horizon, randomSeed)
    elif algorithm == available_algos[2]:
        REG = kl_ucb(arm_means, horizon, randomSeed)
    elif algorithm == available_algos[3]:
        REG = thompson_sampling(arm_means, horizon, randomSeed)
    elif algorithm == available_algos[4]:
        REG = ucb_t2(arm_means, horizon, randomSeed, scale)  
    elif algorithm == available_algos[5]:
        REG = algo_t3(support, arm_means, horizon, randomSeed, 0.3)
    elif algorithm == available_algos[6]:
        HIGH = algo_t4(support, arm_means, horizon, randomSeed, threshold)
     
    output_str = '{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(instance, algorithm, randomSeed, epsilon, scale, threshold, horizon, REG, HIGH)
    print(output_str)
