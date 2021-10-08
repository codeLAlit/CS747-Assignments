import subprocess, os
import numpy as np
pol_num = 0
max_pol = 0
env_for_p1 = ''
env_for_p2 = ''

p2_state_file = 'data/attt/states/states_file_p2.txt'
p1_state_file = 'data/attt/states/states_file_p1.txt'

def turnP1():
    global env_for_p1, pol_num, env_for_p2
    run(p1_state_file, env_for_p1, 1, pol_num)
    env_for_p2 = 'data/task3_p1_policy_{}.txt'.format(pol_num)
    pol_num+=1
    return 

def turnP2():
    global pol_num, env_for_p1, env_for_p2, max_pol
    if pol_num==0:
        inititalizePolicy(p2_state_file, 2)
        env_for_p1 = 'data/task3_p2_policy_0.txt'
        pol_num+=1
        return 

    elif pol_num==max_pol:
        return 
    else:
        run(p2_state_file, env_for_p2, 2, pol_num)
        env_for_p1 = 'data/task3_p2_policy_{}.txt'.format(pol_num)
        pol_num+=1
        return 

def run(state_file, policy, player, policy_num):
    cmd_encoder = "python","encoder.py","--policy",policy,"--states",state_file
    f = open("data/tmp_mdp", 'w')
    subprocess.call(cmd_encoder, stdout=f)
    f.close()

    cmd_planner = "python","planner.py","--mdp","data/tmp_mdp"
    f = open("data/tmp_policy_value", 'w')
    subprocess.call(cmd_planner, stdout=f)
    f.close()

    cmd_decoder = "python","decoder.py","--value-policy","data/tmp_policy_value","--states",state_file ,"--player-id",str(player)
    f = open("data/task3_p{}_policy_{}.txt".format(player, policy_num), 'w')
    subprocess.call(cmd_decoder, stdout=f)
    f.close()

    os.remove('data/tmp_mdp')
    os.remove('data/tmp_policy_value')

    return

def inititalizePolicy(statesFile, pID):
    with open(statesFile, 'r') as f:
        data = f.readlines()

    with open('data/task3_p{}_policy_0.txt'.format(pID), 'w') as f:
        f.write('{}\n'.format(pID))
        for line in data:
            state = line.split()[0]
            posActs = [i for i in range(9) if state[i]=='0']
            prob = np.zeros(9)
            for act in posActs:
                prob[act] = np.random.randint(1, 1000)
            prob = prob/np.sum(prob)
            f.write(state+' ')            
            f.write(' '.join(map(str, prob))+'\n')
    
    return

if __name__=='__main__':
    # jiski state file jayegi uski policy aayegi
    max_pol = 20
    pol_num = 0
    env_for_p1 = ''
    env_for_p2 = ''
    np.random.seed(8)
    for i in range(max_pol):
        if i%2==0:
            turnP2()
        else:
            turnP1()

            
