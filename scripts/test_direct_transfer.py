import gym
import numpy as np
import os
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import random
# from safe_rl_cmdp.trpo_lagrangian import TRPO_lagrangian
from stable_baselines import TRPO, PPO2, SAC
from rl_gat import *

ALGO = SAC
# set the environment here :
REAL_ENV_NAME = 'HalfCheetahModified-v2'
SIM_ENV_NAME = 'HalfCheetah-v2' # Walker2d, HalfCheetah, Hopper, Walker2dModified, HalfCheetahModified
TIME_STEPS = 3000000
INDICATOR = 'seed1'

def evaluate_policy_on_env(env,
                           model,
                           render=True,
                           iters=1,
                           deterministic=False,
                           constrained=False
                           ):
    # model.set_env(env)
    return_list = []
    cost_list = []
    for i in range(iters):
        return_val = 0
        cost_val = 0
        done = False
        obs = env.reset()
        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = env.step(action)
            return_val+=rewards
            if constrained:
              cost_val += info.get('cost', 0)
            else:
              cost_val += 0
            if render:
                env.render()
                # time.sleep(0.01)

        if not i%15: print('Iteration ', i, ' done.')
        return_list.append(return_val)
        cost_list.append(cost_val)
    print('***** STATS FOR THIS RUN *****')
    print('MEAN : ', np.mean(return_list))
    print('STD : ', np.std(return_list))
    print('COST_MEAN : ', np.mean(cost_list))
    print('COST_STD : ', np.std(cost_list))
    return np.mean(return_list), np.std(return_list)/np.sqrt(len(return_list)), \
           np.mean(cost_list), np.std(cost_list)/np.sqrt(len(cost_list))

def test_direct_transfer(algo=ALGO):
    random.seed(1000)
    np.random.seed(1000)
    sim_policy = 'data/models/' + algo.__name__ + '_initial_policy_steps_' + SIM_ENV_NAME + '_'+str(TIME_STEPS) + '_' + INDICATOR +'_.pkl'

    constrained = False
    if algo.__name__ == 'PPO2':
        algo = PPO2
    elif algo.__name__ == 'TRPO':
        algo = TRPO
    elif algo.__name__ == 'SAC':
        algo = SAC
    elif algo.__name__ == 'TRPO_lagrangian':
        algo = TRPO_lagrangian
        constrained = True

    real_env = gym.make(REAL_ENV_NAME)
    real_env.seed(1000)
    val = evaluate_policy_on_env(real_env,
                                 algo.load(sim_policy),
                                 render=False,
                                 iters=50,
                                 deterministic=False,
                                 constrained=constrained)
    with open("scripts/transfer_test/eval_at_real_"+ REAL_ENV_NAME +"_.txt", "a") as txt_file:
        print(val, file=txt_file)

    sim_env = gym.make(SIM_ENV_NAME)
    sim_env.seed(1000)
    val = evaluate_policy_on_env(sim_env,
                                 algo.load(sim_policy),
                                 render=False,
                                 iters=50,
                                 deterministic=False,
                                 constrained=constrained)
    with open("scripts/transfer_test/eval_at_sim_"+ SIM_ENV_NAME +"_.txt", "a") as txt_file:
        print(val, file=txt_file)

    os._exit(0)

if __name__ == '__main__':
    test_direct_transfer()
    os._exit(0)