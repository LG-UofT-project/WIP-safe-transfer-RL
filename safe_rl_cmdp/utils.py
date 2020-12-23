import numpy as np
import warnings
from itertools import zip_longest
from stable_baselines.common.tf_layers import linear
from stable_baselines.common.input import observation_input
import tensorflow as tf 
# from abc import ABC
from stable_baselines.common.policies import FeedForwardPolicy, ActorCriticPolicy, nature_cnn

def mlp_extractor_safe(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network
    const_cost_only_layers = []

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']

            if 'vcf' in layer:
                assert isinstance(layer['vcf'], list), "Error: net_arch[-1]['vcf'] must contain a list of integers."
                const_cost_only_layers = layer['vcf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    latent_vcf = latent

    for idx, (pi_layer_size, vf_layer_size, vcf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers, const_cost_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

        if vcf_layer_size is not None:
            assert isinstance(vcf_layer_size, int), "Error: net_arch[-1]['vcf'] must only contain integers."
            latent_vcf = act_fun(linear(latent_vcf, "vcf_fc{}".format(idx), vcf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value, latent_vcf

class FeedForwardWithSafeValue(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="cnn", **kwargs):
        super(FeedForwardWithSafeValue, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers, vcf=layers)]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = vcf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent, vcf_latent = mlp_extractor_safe(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)
            self._vcf = linear(vcf_latent, 'vcf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()
        self._vcf_flat = self.vcf[:, 0]

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, cost_value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.vcf_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, cost_value, neglogp = self.sess.run([self.action, self.value_flat, self.vcf_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, cost_value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    @property
    def vcf(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, 1)"""
        return self._vcf

    @property
    def vcf_flat(self):
        """tf.Tensor: value estimate, of shape (self.n_batch, )"""
        return self._vcf_flat

    def cost_value(self, obs, state=None, mask=None):
        return self.sess.run(self.vcf_flat, {self.obs_ph: obs})

class MLPWithSafeValue(FeedForwardWithSafeValue):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MLPWithSafeValue, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)

class CnnWithSafeValue(FeedForwardWithSafeValue):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CnnWithSafeValue, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)

# class MLPValue(ABC):
#     """
#     Class for safety value function
#     """
#     def __init__(self, sess, ob_space, n_env, n_steps, n_batch, reuse=False, net_arch= [64,64],
#                  act_fun=tf.tanh, **kwargs ):
#
#       self.sess = sess
#       self.ob_space = ob_space
#       self.n_env = n_env
#       self.n_steps = n_steps
#       self.n_batch = n_batch
#       self.reuse = reuse
#
#       with tf.variable_scope("input", reuse=False):
#             if obs_phs is None:
#                 self._obs_ph, self._processed_obs = observation_input(ob_space, n_batch, scale=scale)
#             else:
#                 self._obs_ph, self._processed_obs = obs_phs
#
#       value = tf.layers.flatten(self._processed_obs)
#       for idx,vc_layer_size in enumerate(net_arch):
#           value = act_fun(linear(latent_policy, "vc_fc{}".format(idx), vc_layer_size, init_scale=np.sqrt(2)))
#
#       self._vc = linear(value, 'vc', 1)
#
#     @property
#     def obs_ph(self):
#         return self._obs_ph
#
#     @property
#     def vc(self):
#         return self._vc
#
#     def step(self, obs):
#       """
#       Returns the value for a single step
#       """
#       return self.sess.run(self.vc, {self.obs_ph: obs})


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
    seg["tdlamcost"] = seg["cadv"] + seg["vcpred"]

def total_episode_reward_cost_logger(rew_acc, rewards, cost_acc, costs, masks, writer, steps):
    """
    calculates the cumulated episode reward, and prints to tensorflow log the output

    :param rew_acc: (np.array float) the total running reward
    :param rewards: (np.array float) the rewards
    :param masks: (np.array bool) the end of episodes
    :param writer: (TensorFlow Session.writer) the writer to log to
    :param steps: (int) the current timestep
    :return: (np.array float) the updated total running reward
    :return: (np.array float) the updated total running reward
    """
    with tf.variable_scope("environment_info", reuse=True):
        for env_idx in range(rewards.shape[0]):
            dones_idx = np.sort(np.argwhere(masks[env_idx]))

            if len(dones_idx) == 0:
                rew_acc[env_idx] += sum(rewards[env_idx])
                cost_acc[env_idx] += sum(costs[env_idx])
            else:
                rew_acc[env_idx] += sum(rewards[env_idx, :dones_idx[0, 0]])
                cost_acc[env_idx] += sum(costs[env_idx, :dones_idx[0, 0]])
                summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                summary = tf.Summary(value=[tf.Summary.Value(tag="episode_cost", simple_value=cost_acc[env_idx])])
                writer.add_summary(summary, steps + dones_idx[0, 0])
                for k in range(1, len(dones_idx[:, 0])):
                    rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[k - 1, 0]:dones_idx[k, 0]])
                    cost_acc[env_idx] = sum(costs[env_idx, dones_idx[k - 1, 0]:dones_idx[k, 0]])
                    summary = tf.Summary(value=[tf.Summary.Value(tag="episode_reward", simple_value=rew_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                    summary = tf.Summary(value=[tf.Summary.Value(tag="episode_cost", simple_value=cost_acc[env_idx])])
                    writer.add_summary(summary, steps + dones_idx[k, 0])
                rew_acc[env_idx] = sum(rewards[env_idx, dones_idx[-1, 0]:])
                cost_acc[env_idx] = sum(costs[env_idx, dones_idx[-1, 0]:])

    return rew_acc, cost_acc


