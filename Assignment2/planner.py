import numpy as np
import argparse
import os
from policyalgos import valueIteration

def parseMDP(mdpfile):
    data = None
    with open(mdpfile, 'r') as f:
        data = f.read()
    
    data = data.split('\n')
    try:
        data.remove('')
    except:
        pass
    
    mdp_dict = {}
    mdp_dict['numStates'] = int(data[0].split(' ')[-1])
    mdp_dict['numActions'] = int(data[1].split(' ')[-1])
    mdp_dict['termStates'] = [int(i) for i in data[2].split(' ')[1:]]
    mdp_dict['type'] = data[-2].split(' ')[-1]
    mdp_dict['gamma'] = float(data[-1].split(' ')[-1])

    rewards = np.zeros((mdp_dict['numStates'], mdp_dict['numActions'], mdp_dict['numStates']))
    tprob = np.zeros((mdp_dict['numStates'], mdp_dict['numActions'], mdp_dict['numStates']))

    for trans in data[3:-2]:
        trans = trans.split(' ')[1:]
        s, a, sd, r, p = int(trans[0]), int(trans[1]), int(trans[2]), float(trans[3]), float(trans[4])
        rewards[s, a, sd] = r
        tprob[s, a, sd] = p

    mdp_dict['rewards'] = rewards
    mdp_dict['tranProb'] = tprob   

    return mdp_dict

def printData(Vs, Pis):
    for s in range(len(Vs)):
        print("{}\t{}\n".format(Vs[s], Pis[s]))
    return

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp", required=True, type=str)
    parser.add_argument("--algorithm", '-a', required=False, type=str, default='all')
    args = parser.parse_args()
    mdpfile = args.mdp
    algo = args.algorithm

    mdp_data = parseMDP(mdpfile)

    if algo=='vi':
        Vs, Pis = valueIteration(mdp_data)
        printData(Vs, Pis)
    elif algo=='hpi':
        print('howard policy iteration')
    elif algo=='lp':
        print('linear programming')
    else:
        print(mdp_data, 'all')
    