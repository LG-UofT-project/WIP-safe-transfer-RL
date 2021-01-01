"""
File containing plotting mechanisms for different visualizations
"""

import gym
import safety_gym
import numpy as np
import os
from scripts.utils import MujocoNormalized
import argparse
from rl_gat.reinforcedgat import GroundedEnv
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import warnings
warnings.filterwarnings("ignore")
from stable_baselines import TRPO, PPO2
  
def calculate_transition_errors(sim_env, real_env, policy, action_tf_policy_list):
    
    deterministic = True
    
    transition_errors = []
    num_episodes = 5
    j = 0
    for action_tf_policy in action_tf_policy_list:
        print(' Begin Iteration ',j)
        l2_error_list = []
        for i in range(num_episodes):
           
            grounded_env = GroundedEnv(sim_env,
                                   action_tf_policy= action_tf_policy,
                                   debug_mode=False,
                                   data_collection_mode=False,
                                   use_deterministic=deterministic)
            
            done = False
            obs = real_env.reset()
            grounded_env.reset_state(obs)
            
            while not done:
                action, _state = policy.predict(obs, deterministic=deterministic)
                obs, _, done, _ = real_env.step(action)
                grounded_env.reset_state(obs)
                grounded_obs, _, done_grounded, _ = grounded_env.step(action)
                l2_error_list.append(np.linalg.norm(grounded_obs - obs))
                if done_grounded:
                    print('Broke at target')
            
        transition_errors.append(np.mean(l2_error_list))
        print('Transition error at this iteration', np.mean(l2_error_list))
        j = j+1
    
    return transition_errors

def main():
    
    parser = argparse.ArgumentParser(description='Plotting mechanisms for GARAT and related modifications')
    parser.add_argument('--sim_env', default = "InvertedPendulum-v2", type=str, help="Name of the simulator/source environment")
    parser.add_argument('--real_env', default = "InvertedPendulumModified-v2", type=str, help="Name of the real/target environment")
    parser.add_argument('--load_policy_path', default = "data/models/TRPO_initial_policy_steps_InvertedPendulum-v2_2000000_.pkl", help="relative path of policy to be used for generating plots")
    parser.add_argument('--load_atp_path', default = "data/models/garat/Single_GAIL_sim2real_TRPO_2000000_1000_50_0/", type=str, help="relative path for stored Action transformation policies")
    parser.add_argument('--seed', default = 0, type=int, help="Random seed")    
    args = parser.parse_args()
    
    #Set seed
    np.random.seed(args.seed)
    
    sim_env = gym.make(args.sim_env)
    real_env = gym.make(args.real_env)
    
    policy = TRPO.load(args.load_policy_path)
    
    action_tf_policy_list = []
    
    num_grounding = 50
    
    print('################## Begin File loading ##################')
    for index in range(num_grounding):
        file_path = os.path.join(args.load_atp_path,"action_transformer_policy1_"+str(index)+".pkl")
        print(file_path)
        action_tf_policy_list.append(PPO2.load(file_path))
            
    
    print('################## File loading Completed ##################')
    
    results = calculate_transition_errors(sim_env, real_env, policy, action_tf_policy_list)

if __name__ == '__main__':
    main()
    os._exit(0)