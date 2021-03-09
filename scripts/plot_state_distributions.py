import gym, safety_gym
import numpy as np
import os
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import random
from safe_rl_cmdp.trpo_lagrangian import TRPO_lagrangian
from stable_baselines import TRPO, PPO2, SAC
from rl_gat.reinforcedgat import GroundedEnv
from scripts.utils import MujocoNormalized
from rl_gat.gat import collect_gym_trajectories
import matplotlib.pyplot as plt
import seaborn as sns

ALGO = TRPO
ROLLOUT_ALGO = TRPO
# set the environment here :
REAL_ENV_NAME = 'HopperFrictionModified-v2'  # HopperFrictionModified-v2, Walker2dModified, InvertedPendulumModified
SIM_ENV_NAME = 'Hopper-v2'  # Hopper-v2, Walker2d, InvertedPendulum
MUJOCO_NORMALIZE = True
ATP_NAME = 'Single_Hopper_authors_correctDisc_GAIL_sim2real_TRPO_0_10000_50_1'
# set this to the parent environment
TIME_STEPS = 1000000  # 2000000
SEED = 1
TOTAL_STEPS = 100000


def plot_state_distributions(algo=ALGO):
    random.seed(SEED)
    np.random.seed(SEED)

    test_policy = 'data/models/' + algo.__name__ + '_initial_policy_steps_' + SIM_ENV_NAME + '_' + str(
        TIME_STEPS) + '_.pkl'

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

    # test_policy = algo.load(test_policy)

    sim_env = gym.make(SIM_ENV_NAME)
    real_env = gym.make(REAL_ENV_NAME)

    if MUJOCO_NORMALIZE:
        sim_env = MujocoNormalized(sim_env)
        real_env = MujocoNormalized(real_env)

    atp_policy = 'data/models/garat/' + ATP_NAME + '/grounding_step_0/action_transformer_policy1_49.pkl'
    # atp_policy = 'data/models/garat/TRPO_on_overleaf/' + ATP_NAME + '/action_transformer_policy1_49.pkl' #/grounding_step_0
    # atp_policy = 'data/models/garat/SAC_on_overleaf/' + ATP_NAME + '/action_transformer_policy1_49.pkl'  # /grounding_step_0
    atp_environment = PPO2.load(atp_policy)

    use_deterministic = False,
    grounded_env = GroundedEnv(env=sim_env,
                               action_tf_policy=atp_environment,
                               debug_mode=False,
                               data_collection_mode=False,
                               use_deterministic=use_deterministic,
                               atp_policy_noise=0.01 if use_deterministic else 0.0,
                               )
    sim_env = DummyVecEnv([lambda: sim_env])
    real_env = DummyVecEnv([lambda: real_env])
    grounded_env = DummyVecEnv([lambda: grounded_env])

    real_Ts = collect_gym_trajectories(env=real_env,
                                       policy=algo.load(test_policy),
                                       limit_trans_count=TOTAL_STEPS,
                                       num=None,
                                       add_noise=0.0,
                                       deterministic=False,
                                       )
    X = []
    for T in real_Ts:  # For each trajectory:
        for i in range(len(T)):
            X.append(T[i][0])
    real_Ts = np.array(X)

    sim_Ts = collect_gym_trajectories(env=sim_env,
                                      policy=algo.load(test_policy),
                                      limit_trans_count=TOTAL_STEPS,
                                      num=None,
                                      add_noise=0.0,
                                      deterministic=False,
                                      )
    X = []
    for T in sim_Ts:  # For each trajectory:
        for i in range(len(T)):
            X.append(T[i][0])
    sim_Ts = np.array(X)

    grounded_Ts = collect_gym_trajectories(env=grounded_env,
                                           policy=algo.load(test_policy),
                                           limit_trans_count=TOTAL_STEPS,
                                           num=None,
                                           add_noise=0.0,
                                           deterministic=False,
                                           )
    X = []
    for T in grounded_Ts:  # For each trajectory:
        for i in range(len(T)):
            X.append(T[i][0])
    grounded_Ts = np.array(X)

    for i in range(np.shape(real_Ts)[1]):
        # for i in range(1):
        p1 = sns.kdeplot(real_Ts[:, i], shade=True, color="r", label="target")
        p1 = sns.kdeplot(sim_Ts[:, i], shade=True, color="b", label="source")
        p1 = sns.kdeplot(grounded_Ts[:, i], shade=True, color="g", label="grounded")
        plt.legend()
        plt.savefig(
            SIM_ENV_NAME + '_' + ROLLOUT_ALGO.__name__ + '_' + ALGO.__name__ + '_dimension' + str(i + 1) + '.png')
        plt.show()
    os._exit(0)


if __name__ == '__main__':
    plot_state_distributions()
    os._exit(0)