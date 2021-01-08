"""
File containing plotting mechanisms for different visualizations
"""

import gym
import numpy as np
import os
import argparse
from rl_gat.reinforcedgat import GroundedEnv
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
import warnings
warnings.filterwarnings("ignore")
from stable_baselines import TRPO, PPO2
import seaborn as sns
  
def calculate_transition_errors(sim_env, real_env, policy, action_tf_policy_list):
    
    deterministic = True
    
    transition_errors = []
    num_episodes = 50
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
            grounded_env.reset()

            while not done:
                grounded_env.reset_state(obs)
                action, _state = policy.predict(obs, deterministic=deterministic)
                obs, _, done, _ = real_env.step(action)
                grounded_obs, _, done_grounded, _ = grounded_env.step(action)
                l2_error_list.append(np.linalg.norm(grounded_obs - obs))
                if done_grounded:
                    print('Broke at grounded environment')
        
        transition_errors.append(np.mean(l2_error_list))
        print('Transition error at this iteration', np.mean(l2_error_list))
        j = j+1
    
    return transition_errors

def plot_results(results_dict):
    
    window = 5
    
    for key, result in results_dict.items():
        running_average = np.convolve(result, np.ones(window)*1/window, mode="valid")
        x = np.arange(window, running_average.shape[0]+window)
        sns.lineplot(x, running_average, label=key)

    plt.title('Transition Error between Target and Grounded Environments')
    plt.xlabel('Action Transformation Policy updates')
    plt.ylabel('Average per-step transition error')
    plt.show()

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
    
    action_tf_policy_list_single = []
    action_tf_policy_list_double = []
    action_tf_policy_list_shared_double = []
    action_tf_policy_list_airl = []
    num_grounding = 50
    
    atp_path_single = args.load_atp_path
    atp_path_double = args.load_atp_path.replace('_0','_2')
    atp_path_shared_double = args.load_atp_path.replace('_0','_1')
    atp_path_airl = args.load_atp_path.replace('Single_GAIL_sim2real_TRPO_2000000_1000_50_0','Single_AIRL_sim2real_TRPO_2000000_1000_50_1')
    
    print('################## Begin File loading ##################')
    for index in range(num_grounding):
        file_path_single = os.path.join(atp_path_single,"action_transformer_policy1_"+str(index)+".pkl")
        print(file_path_single)
        action_tf_policy_list_single.append(PPO2.load(file_path_single))
        file_path_double = os.path.join(atp_path_double,"action_transformer_policy1_"+str(index)+".pkl")
        print(file_path_double)
        action_tf_policy_list_double.append(PPO2.load(file_path_double))
        file_path_shared_double = os.path.join(atp_path_shared_double,"action_transformer_policy1_"+str(index)+".pkl")
        print(file_path_shared_double)
        action_tf_policy_list_shared_double.append(PPO2.load(file_path_shared_double))
        #file_path_airl = os.path.join(atp_path_airl,"action_transformer_policy1_"+str(index)+".pkl")
        #print(file_path_airl)
        #action_tf_policy_list_airl.append(PPO2.load(file_path_airl))           
    results_dict = {}
    print('################## File loading Completed ##################')
    
    results_single = calculate_transition_errors(sim_env, real_env, policy, action_tf_policy_list_single)
    
    print('############## Begin Double Discriminator Calculations')    

    results_shared_double = calculate_transition_errors(sim_env, real_env, policy, action_tf_policy_list_shared_double)
    
    results_double = calculate_transition_errors(sim_env, real_env, policy, action_tf_policy_list_double)

    print('############## Begin AIRL Calculations')    
    
    #results_airl = calculate_transition_errors(sim_env, real_env, policy, action_tf_policy_list_airl)
        
    results_dict['GARAT'] = results_single
    results_dict['GARAT Double Discriminator'] = results_double
    results_dict['GARAT Double Discriminator (Generator LR modifications)'] = results_shared_double
    #results_dict['GARAT AIRL'] = results_airl
        
    plot_results(results_dict)
    
    
if __name__ == '__main__':
    main()
    os._exit(0)