import numpy as np
from stable_baselines.common.tf_layers import linear
from stable_baselines.common.input import observation_input
import tensorflow as tf 
from abc import ABC

class MLPValue(ABC):
    """
    Class for safety value function
    """
    def __init__(self, sess, ob_space, n_env, n_steps, n_batch, reuse=False, net_arch= [64,64],
                 act_fun=tf.tanh, **kwargs ):
      
      self.sess = sess
      self.ob_space = ob_space
      self.n_env = n_env 
      self.n_steps = n_steps
      self.n_batch = n_batch
      self.reuse = reuse
      
      with tf.variable_scope("input", reuse=False):
            if obs_phs is None:
                self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)
            else:
                self._obs_ph, self._processed_obs = obs_phs
      
      value = tf.layers.flatten(self._processed_obs)
      for idx,vc_layer_size in enumerate(net_arch):
          value = act_fun(linear(latent_policy, "vc_fc{}".format(idx), vc_layer_size, init_scale=np.sqrt(2)))
      
      self._vc = linear(value, 'vc', 1)
    
    @property
    def obs_ph(self):
        return self._obs_ph
    
    @property
    def vc(self):
        return self._vc
         
    def step(self, obs):
      """
      Returns the value for a single step
      """
      return self.sess.run(self.vc, {self.obs_ph: obs})


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]
    
def add_vctarg_and_cadv(seg,cost_gamma,cost_lam):
    """
    Compute target value for cost function using TD(lambda) estimator, and "cost advantage" with GAE(lambda)
    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param cost_gamma: (float) Discount factor for costs
    :param cost_lam: (float) GAE factor for costs
    """
    episode_starts = np.append(seg["episode_starts"],False)
    vcpred = np.append(seg["vcpred"], seg["nextvcpred"]) 
    cost_len = len(seg["costs"])
    seg["cadv"] = np.empty(cost_len, 'float32')
    costs = seg["costs"]
    lastgaelam = 0
    for step in reversed(range(cost_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        cdelta = costs[step] + cost_gamma * vcpred[step + 1] * nonterminal - vcpred[step]
        seg["cadv"][step] = lastgaelam = cdelta + cost_gamma * cost_lam * nonterminal * lastgaelam
    seg["tdlamcost"] = seg["cadv"] + seg["cvpred"]
         
