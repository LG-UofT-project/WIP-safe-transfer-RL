from safety_gym.envs.engine import Engine
from gym.envs.registration import register

config = {
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
         kwargs={'config': config})

