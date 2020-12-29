from safety_gym.envs.engine import Engine
from gym.envs.registration import register
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os 

class InvertedPendulumModifiedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, os.getcwd()+'/rl_gat/envs/inverted_pendulum_mod.xml', 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

"""


config_point = {
    'observe_goal_lidar': True,
    'observe_box_lidar': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,
    'robot_base' : 'point_friction_mod.xml',
    'task': 'goal',
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'hazards_size': 0.2,
    'hazards_keepout': 0.18,
    'constrain_hazards': True,
    'observe_hazards': True,
    'observe_vases': True,
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'hazards_num': 8,
    'vases_num': 1  
}


register(id='Safexp-PointGoal1slippery-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config_point})
"""
