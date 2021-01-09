import gym, safety_gym
import numpy as np
import os
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import random
from safe_rl_cmdp.trpo_lagrangian import TRPO_lagrangian
from stable_baselines import TRPO, PPO2
from rl_gat.reinforcedgat import GroundedEnv

ALGO = TRPO_lagrangian
# set the environment here :
# REAL_ENV_NAME = 'Safexp-PointGoal1Heavy-v0'
REAL_ENV_NAME = 'Safexp-PointGoal1slippery-v0'
SIM_ENV_NAME = 'Safexp-PointGoal1-v0'
ATP_NAME = 'Single_TRPO_lagGAIL_sim2real_TRPO_lagrangian_10000000_1000_50_1'
# set this to the parent environment
TIME_STEPS = 10000000 # 10000000, 2000000

def evaluate_policy_on_env(env,
                           model,
                           render=True,
                           iters=1,
                           deterministic=True,
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
    random.seed(1)
    np.random.seed(1)

    # sim_policy = 'data/models/' + algo.__name__ + '_initial_policy_steps_' + SIM_ENV_NAME + '_'+str(TIME_STEPS)+'_.pkl'
    real_policy = 'data/models/' + algo.__name__ + '_initial_policy_steps_' + REAL_ENV_NAME + '_'+str(TIME_STEPS)+'_.pkl'
    atp_policy = 'data/models/garat/' + ATP_NAME + '/action_transformer_policy1_49.pkl'

    sim_env = gym.make(SIM_ENV_NAME)
    atp_environment = PPO2.load(atp_policy)

    use_deterministic = False,
    test_env = GroundedEnv(env=sim_env,
                           action_tf_policy=atp_environment,
                           # action_tf_env=self.atp_environment,
                           debug_mode=False,
                           data_collection_mode=False,
                           use_deterministic=use_deterministic,
                           atp_policy_noise=0.01 if use_deterministic else 0.0,
                           )

    # self.grounded_sim_env = DummyVecEnv([lambda: grnd_env])

    if 'HalfCheetah' in REAL_ENV_NAME or 'Reacher' in REAL_ENV_NAME or 'InvertedPendulum' in REAL_ENV_NAME:
        # sim_policy = sim_policy.replace('10000000_.pkl', '2000000_.pkl')
        real_policy = real_policy.replace('10000000_.pkl', '2000000_.pkl')

    if algo.__name__ == 'PPO2':
        algo = PPO2
    elif algo.__name__ == 'TRPO':
        algo = TRPO
    elif algo.__name__ == 'TRPO_lagrangian':
        algo = TRPO_lagrangian
        constrained = True

    val = evaluate_policy_on_env(test_env,
                                 algo.load(real_policy),
                                 render=False,
                                 iters=50,
                                 deterministic=True,
                                 constrained=constrained)
    with open("scripts/transfer_test/eval_at_grounded.txt", "a") as txt_file:
        print(ATP_NAME, file=txt_file)
        print(val, file=txt_file)

    val = evaluate_policy_on_env(test_env,
                                 algo.load(real_policy),
                                 render=False,
                                 iters=50,
                                 deterministic=False,
                                 constrained=constrained)
    with open("scripts/transfer_test/eval_at_grounded.txt", "a") as txt_file:
        print(val, file=txt_file)

    # # Real policy
    # del algo  # remove the old algo and reload it.
    # if algo.__name__ == 'PPO2':
    #     algo = PPO2
    # elif algo.__name__ == 'TRPO':
    #     algo = TRPO
    # elif algo.__name__ == 'TRPO_lagrangian':
    #     algo = TRPO_lagrangian
    #     constrained = True
    #
    # val = evaluate_policy_on_env(test_env,
    #                              algo.load(real_policy),
    #                              render=False,
    #                              iters=50,
    #                              deterministic=True,
    #                              constrained=constrained)
    # with open("scripts/transfer_test/eval_at_real.txt", "a") as txt_file:
    #     print(val, file=txt_file)
    os._exit(0)

if __name__ == '__main__':
    test_direct_transfer()
    os._exit(0)

# grnd_env = GroundedEnv(env=env,
#                        action_tf_policy=self.action_tf_policy,
#                        # action_tf_env=self.atp_environment,
#                        alpha=alpha,
#                        debug_mode=False,
#                        normalizer=self.target_policy_norm_obs,
#                        data_collection_mode=False,
#                        use_deterministic=use_deterministic,
#                        atp_policy_noise=0.01 if use_deterministic else 0.0,
#                        )
#
# self.grounded_sim_env = DummyVecEnv([lambda: grnd_env])