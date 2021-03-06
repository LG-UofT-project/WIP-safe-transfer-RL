import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from rl_gat.reinforcedgat import ReinforcedGAT
from safe_rl_cmdp.trpo_lagrangian import TRPO_lagrangian
import gym, os, glob, shutil, safety_gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2, TRPO, SAC
import argparse, sys
from termcolor import cprint
from scripts.utils import MujocoNormalized
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import torch
import random
from gym.wrappers import TimeLimit
torch.backends.cudnn.deterministic = True

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)

def check_tmp_folder():
    """
    function to check if the tmp folder is clean, for storing
    trajectories as numpy files in tmp folder.
    :return:python
    """
    if os.path.exists('./data/tmp'):
        print('data/tmp/ directory already exists. Deleting all files. ')
        shutil.rmtree('./data/tmp')

    try:
        os.mkdir('./data/tmp')
        print('Successfully created tmp folder. ')

    except Exception as e:
        print(e)

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

def main():
    # SIM_ENV_NAME = 'InvertedPendulum-v2'
    # REAL_ENV_NAME = 'InvertedPendulumModified-v2_old'

    # expt_label = input('Enter a label for the experiment : ')

    # Set default parameters to follow run_experiments.sh
    parser = argparse.ArgumentParser(description='Reinforced Grounded Action Transformation')
    parser.add_argument('--target_policy_algo', default="SAC", type=str, help="name in str of the agent policy training algorithm")
    parser.add_argument('--action_tf_policy_algo', default="PPO2", type=str, help="name in str of the Action Transformer policy training algorithm")
    parser.add_argument('--load_policy_path', default="data/models/SAC_initial_policy_steps_InvertedPendulum-v2_1000000_.pkl", help="relative path of initial policy trained in sim")
    parser.add_argument('--load_atp_policy_path', default=None, help="relative path of action transformation policy for reuse")
    parser.add_argument('--alpha', default=1.0, type=float, help="Deprecated feature. Ignore")
    parser.add_argument('--beta', default=1.0, type=float, help="Deprecated feature. Ignore")
    parser.add_argument('--n_trainsteps_target_policy', default=1000000, type=int, help="Number of time steps to train the agent policy in the grounded environment")
    parser.add_argument('--n_trainsteps_action_tf_policy', default=1000000, type=int, help="Timesteps to train the Action Transformer policy in the ATPEnvironment")
    # Depreciated; gsim_trans*generator_epochs if not single_batch_test else 5000
    parser.add_argument('--num_cores', default=1, type=int, help="Number of threads to use while collecting real world experience") # was 10
    parser.add_argument('--sim_env', default='InvertedPendulum-v2', help="Name of the simulator environment (Unmodified)")
    parser.add_argument('--real_env', default='InvertedPendulumModified-v2', help="Name of the Real World environment (Modified)")
    parser.add_argument('--n_frames', default=1, type=int, help="Number of previous frames observed by discriminator")
    parser.add_argument('--expt_number', default=1, type=int, help="Expt. number to keep track of multiple experiments")
    parser.add_argument('--n_grounding_steps', default=1, type=int, help="Number of grounding steps. (Outerloop of algorithm ) ")
    parser.add_argument('--n_iters_atp', default=50, type=int, help="Number of GAN iterations")
    parser.add_argument('--discriminator_epochs', default=1, type=int, help="Discriminator epochs per GAN iteration")
    parser.add_argument('--generator_epochs', default=1, type=int, help="ATP epochs per GAN iteration")
    parser.add_argument('--real_trajs', default=1000, type=int, help="Set max amount of real TRAJECTORIES used")
    parser.add_argument('--sim_trajs', default=1000, type=int, help="Set max amount of sim TRAJECTORIES used")
    parser.add_argument('--real_trans', default=10000, type=int, help="amount of real world transitions used")
    # Actual amount of transitions can be bigger than this number, since this waits for the end of episode.
    parser.add_argument('--gsim_trans', default=10000, type=int, help="amount of simulator transitions used")
    parser.add_argument('--debug', action='store_true', help="DEPRECATED")
    parser.add_argument('--eval', action='store_false', help="set to true to evaluate the agent policy in the real environment, after training in grounded environment")
    parser.add_argument('--eval_ref', action='store_true', help="set to true to evaluate reference policies in the real environment")
    parser.add_argument('--use_cuda', action='store_true', help="DEPRECATED. Not using CUDA")
    parser.add_argument('--instance_noise', action='store_true', help="DEPRECATED. Not using instance noise")
    parser.add_argument('--ent_coeff', default=0.01, type=float, help="entropy coefficient for the PPO algorithm, used to train the action transformer policy")
    parser.add_argument('--max_kl', default=3e-4, type=float, help="Set this only if using TRPO for the action transformer policy")
    parser.add_argument('--clip_range', default=0.1, type=float, help="PPO objective clipping factor -> Action transformer policy")
    parser.add_argument('--use_condor', action='store_true', help="UNUSABLE")
    parser.add_argument('--plot', action='store_false', help="visualize the action transformer policy - works well only for simple environments")
    parser.add_argument('--tensorboard', action='store_false', help="visualize training in tensorboard")
    parser.add_argument('--save_atp', action='store_false', help="Saves the action transformer policy")
    parser.add_argument('--save_target_policy', action='store_false', help="saves the agent policy")
    parser.add_argument('--debug_discriminator', action='store_false', help="UNUSED")
    parser.add_argument('--use_eval_callback', action='store_true', help="UNUSED")
    parser.add_argument('--loss_function', default="GAIL", type=str, help="choose from the list: ['GAIL', 'WGAN', 'AIRL', 'FAIRL']")
    parser.add_argument('--reset_disc_only', action='store_true', help="UNUSED")
    parser.add_argument('--namespace', default="TEST1", type=str, help="namespace for the experiments")
    parser.add_argument('--dont_reset', action='store_true', help="UNUSED")
    parser.add_argument('--reset_target_policy', action='store_true', help="UNUSED")
    parser.add_argument('--randomize_target_policy', action='store_true', help="UNUSED")
    parser.add_argument('--compute_grad_penalty', action='store_false', help="set this to true to compute the GP term while training the discriminator")
    parser.add_argument('--single_batch_test', action='store_false', help="performs a single update of the generator and discriminator.")
    parser.add_argument('--folder_namespace', default="None", type=str, help="UNUSED")
    parser.add_argument('--disc_lr', default=3e-3, type=float, help="learning rate for the AdamW optimizer to update the discriminator")
    parser.add_argument('--atp_lr', default=3e-4, type=float, help="learning rate for the Adam optimizer to update the agent policy")
    # parser.add_argument('--atp_lr', default=1e-3, type=float, help="learning rate for the Adam optimizer to update the agent policy")
    parser.add_argument('--nminibatches', default=2, type=int, help="Number of minibatches used by the PPO algorithm to update the action transformer policy")
    parser.add_argument('--noptepochs', default=1, type=int, help="Number of optimization epochs performed per minibatch by the PPO algorithm to update the action transformer policy")
    parser.add_argument('--deterministic', default=0, type=int, help="set to 0 to use the deterministic action transformer policy in the grounded environment")
    parser.add_argument('--single_batch_size', default=512, type=int, help="batch size for the GARAT update")
    parser.add_argument('--double_discriminators', action='store_true', help="set to use separate double discriminators")
    parser.add_argument('--shared_double_discriminators', action='store_false', help="set to use shared double discriminators")
    parser.add_argument('--use_darc', action='store_false', help="set to use reward shaping mechanism from DARC")
    parser.add_argument('--mujoco_norm', action='store_true', help="normalize environment")
    parser.add_argument('--time_limit', action='store_true', help="set maximum episode length")
    parser.add_argument('--discriminate_diff', action='store_false', help="set to use s'-s")

    args = parser.parse_args()

    # set the seeds here for experiments
    random.seed(args.expt_number)
    np.random.seed(args.expt_number)
    torch.manual_seed(args.expt_number)

    # if args.wgan: args.loss_function = 'WGAN'

    # make dummy gym environment
    dummy_env = gym.make(args.real_env)
    if args.mujoco_norm: dummy_env = MujocoNormalized(dummy_env)
    if args.time_limit: dummy_env = TimeLimit(dummy_env)

    if args.dont_reset is True and args.reset_disc_only is True:
        raise ValueError('Cannot have both args dont_reset and reset_disc_only. Choose one.')

    if args.double_discriminators is True and args.shared_double_discriminators is True:
        raise ValueError('Choose one of double discriminator structure')

    if args.target_policy_algo == 'TRPO_lagrangian':
        constrained = True
    else:
        constrained = False


    expt_type = 'sim2sim' if args.sim_env == args.real_env else 'sim2real'
    expt_label = args.namespace + args.loss_function + '_' + expt_type + '_' + args.target_policy_algo + '_' + str(
        args.n_trainsteps_target_policy) + '_' + str(args.real_trans) + '_' + str(args.n_iters_atp) + '_' + str(args.expt_number)

    # create the experiment folder
    if args.use_condor:
        if args.folder_namespace is "None":
            expt_path = '/u/' + args.real_env + '/' + expt_label
        else:
            expt_path = '/u/' + args.folder_namespace + '/' + expt_label
    else:
        expt_path = 'data/models/garat/' + expt_label
    expt_already_running = False

    gatworld = ReinforcedGAT(
        load_policy=args.load_policy_path,
        sim_seed=args.expt_number,
        real_seed=args.expt_number,
        model_seed=args.expt_number,
        num_cores=args.num_cores,
        sim_env_name=args.sim_env,
        real_env_name=args.real_env,
        expt_label=expt_label,
        frames=args.n_frames,
        algo=args.target_policy_algo,
        atp_algo=args.action_tf_policy_algo,
        debug=args.debug,
        real_trajs=args.real_trajs,
        sim_trajs=args.sim_trajs,
        use_cuda=args.use_cuda,
        real_trans=args.real_trans,
        gsim_trans=args.gsim_trans,
        expt_path=expt_path,
        tensorboard=args.tensorboard,
        atp_loss_function=args.loss_function,
        single_batch_size=None if args.single_batch_size == 0 else args.single_batch_size,
        shared_double=args.shared_double_discriminators,
        mujoco_norm=args.mujoco_norm,
        time_limit=args.time_limit,
        discriminate_diff=args.discriminate_diff,
    )

    # checkpointing logic ~~ necessary when deploying script on Condor cluster
    if os.path.exists(expt_path):
        print('~~ Resuming from checkpoint ~~')

        # remove the best_model.zip file if it exists
        if os.path.exists(expt_path+'/best_model.zip'):
            os.remove(expt_path+'/best_model.zip')

        expt_already_running = True
        grounding_step = len(glob.glob(expt_path+'/*.pkl'))
        print('found ',grounding_step,' target policies in disk')
        if grounding_step == args.n_grounding_steps: # training has ended
            raise ValueError('Rerunning same experiment again ! Exiting')
        else:
            if grounding_step>0:
                print('reloading weights of the target policy')
                gatworld.load_model(expt_path+'/target_policy_'+str(grounding_step-1)+'.pkl')
    else:
        print('First time running experiment')
        os.makedirs(expt_path)
        grounding_step = 0

        with open(expt_path + '/commandline_args.txt', 'w') as f:
            f.write('\n'.join(sys.argv[1:]))

    start_grouding_step = grounding_step

    if args.reset_disc_only or args.dont_reset:
        cprint('~~ INITIALIZING DISCRIMINATOR AND ATP POLICY ~~', 'yellow')
        # gatworld._init_rgat_models(algo=args.action_tf_policy_algo,
        #                            ent_coeff=args.ent_coeff,
        #                            max_kl=args.max_kl,
        #                            clip_range=args.clip_range,
        #                            atp_loss_function=args.loss_function,
        #                            disc_lr=args.disc_lr,
        #                            atp_lr=args.atp_lr,
        #                            nminibatches=args.nminibatches,
        #                            noptepochs=args.noptepochs,
        #                            double_discriminators=args.double_discriminators,
        #                            )
        gatworld._init_discriminators(atp_loss_function=args.loss_function,
                                      disc_lr=args.disc_lr,
                                      double_discriminators=args.double_discriminators,
                                      )
        gatworld._init_ATP(algo=args.action_tf_policy_algo,
                           atp_load_policy=args.load_atp_policy_path,
                           atp_loss_function=args.loss_function,
                           ent_coeff=args.ent_coeff,
                           max_kl=args.max_kl,
                           clip_range=args.clip_range,
                           atp_lr=args.atp_lr,
                           nminibatches=args.nminibatches,
                           noptepochs=args.noptepochs,
                           )

    expt_path_base = expt_path
    expt_label_base = expt_label
    for _ in range(args.n_grounding_steps-start_grouding_step):
        grounding_step += 1

        # RESET expt_path and expt_label
        expt_path = expt_path_base + '/grounding_step_' + str(_)
        expt_label = expt_label_base + '_' + str(_)
        gatworld.set_exp_path(expt_path, expt_label)
        os.makedirs(expt_path)

        real_Rs, real_Cs = gatworld.collect_experience_from_real_env(constrained = constrained)
        with open(expt_path + '/real_rewards' + str(grounding_step) + '.txt', 'w') as fr:
            np.savetxt(fr,real_Rs)
            fr.close()
        with open(expt_path + '/real_costs' + str(grounding_step) + '.txt', 'w') as fc:
            np.savetxt(fc,real_Cs)
            fc.close()

        cprint('~~ RESETTING DISCRIMINATOR AND ATP POLICY ~~', 'yellow')
        if args.reset_disc_only:
            gatworld._init_discriminators(atp_loss_function=args.loss_function,
                                          disc_lr=args.disc_lr,
                                          double_discriminators=args.double_discriminators,
                                          )
        else:
            gatworld._init_rgat_models(algo=args.action_tf_policy_algo,
                                       ent_coeff=args.ent_coeff,
                                       max_kl=args.max_kl,
                                       clip_range=args.clip_range,
                                       atp_loss_function=args.loss_function,
                                       disc_lr=args.disc_lr,
                                       atp_lr=args.atp_lr,
                                       nminibatches=args.nminibatches,
                                       noptepochs=args.noptepochs,
                                       double_discriminators=args.double_discriminators,
                                       )

        # ground the environment
        for ii in range(args.n_iters_atp):
            print('################### GROUNDING INNER ITERATION : ', ii, ' ###################')
            for _ in range(args.discriminator_epochs):
                gatworld.train_discriminator(iter_step=ii,
                                             grounding_step=grounding_step,
                                             num_epochs=args.noptepochs*5 if ii <= 10 else args.noptepochs, # warmup
                                             inject_instance_noise=args.instance_noise,
                                             compute_grad_penalty=args.compute_grad_penalty,
                                             nminibatches=args.nminibatches,
                                             single_batch_test=args.single_batch_test,
                                             debug_discriminator=args.debug_discriminator,
                                             )

            gatworld.train_action_transformer_policy(beta=args.beta,
                                                     num_epochs=args.generator_epochs,
                                                     loss_function=args.loss_function,
                                                     single_batch_test=args.single_batch_test,
                                                     )

            # test grounded environment
            if args.plot and dummy_env.action_space.shape[0]<5:
                # action transformer plot
                gatworld.test_grounded_environment(alpha=args.alpha,
                                                   grounding_step=str(grounding_step) + '_' + str(ii),
                                                   )
            else:
                print('Environment has action space > 5. Skipping AT plotting')


            if args.save_atp:
                # save the action transformer policy for further analysis
                gatworld.save_atp(grounding_step=str(grounding_step) + '_' + str(ii))
                # gatworld.save_grounded_env(grounding_step=str(grounding_step) + '_' + str(ii))

        if args.randomize_target_policy:
            gatworld._randomize_target_policy(algo=args.target_policy_algo)

        gatworld.train_target_policy_in_grounded_env(grounding_step=grounding_step,
                                                     alpha=args.alpha,
                                                     time_steps=args.n_trainsteps_target_policy,
                                                     use_eval_callback=args.use_eval_callback,
                                                     save_model=args.save_target_policy,
                                                     use_deterministic=True if args.deterministic == 1 else False,
                                                     use_darc = args.use_darc
                                                     )

        if args.eval:
            cprint('Evaluating target policy in environment .. ', 'red', 'on_blue')
            test_env = gym.make(args.real_env)
            if args.mujoco_norm: test_env = MujocoNormalized(test_env)
            if args.time_limit: test_env = TimeLimit(test_env)

            ##### Original
            # if 'mujoco_norm' in args.load_policy_path:
            #     test_env = MujocoNormalized(test_env)
            # elif 'normalized' in args.load_policy_path:
            #     test_env = DummyVecEnv([lambda: test_env])
            #     test_env = VecNormalize.load('data/models/env_stats/' + args.sim_env + '.pkl',
            #                             venv=test_env)

            # evaluate on the real world.
            try:
                val = evaluate_policy_on_env(test_env,
                                       gatworld.target_policy,
                                       render=False,
                                       iters=50,
                                       deterministic=True, constrained=constrained)

                with open(expt_path+"/output.txt", "a") as txt_file:
                    print(val, file=txt_file)

                val = evaluate_policy_on_env(test_env,
                                             gatworld.target_policy,
                                             render=False,
                                             iters=50,
                                             deterministic=False,
                                             constrained=constrained)

                with open(expt_path + "/stochastic_output.txt", "a") as txt_file:
                    print(val, file=txt_file)
                print(expt_path)
            except Exception as e:
                cprint(e, 'red')

    # expt done, now get the green and red lines
    if args.eval and args.eval_ref:
        # green line
        cprint('**~~vv^^ GETTING GREEN AND RED LINES ^^vv~~**', 'red','on_green')
        test_env = gym.make(args.real_env)
        if args.mujoco_norm: test_env = MujocoNormalized(test_env)
        if args.time_limit: test_env = TimeLimit(test_env)

        ##### Original
        # if 'mujoco_norm' in args.load_policy_path:
        #     test_env = MujocoNormalized(test_env)
        # elif 'normalized' in args.load_policy_path:
        #     test_env = DummyVecEnv([lambda: test_env])
        #     test_env = VecNormalize.load('data/models/env_stats/' + args.sim_env + '.pkl',
        #                                  venv=test_env)

        sim_policy = 'data/models/'+args.target_policy_algo+'_initial_policy_steps_' + args.sim_env + '_1000000_.pkl'
        real_policy = 'data/models/'+args.target_policy_algo+'_initial_policy_steps_' + args.real_env + '_1000000_.pkl'

        if 'HalfCheetah' in args.load_policy_path or 'Reacher' in args.load_policy_path or 'Hopper' in args.load_policy_path or 'Walker' in args.load_policy_path or 'Ant' in args.load_policy_path:
            sim_policy = sim_policy.replace('1000000_.pkl', '2000000_.pkl')
            real_policy = real_policy.replace('1000000_.pkl', '2000000_.pkl')

        # if 'Walker2d' in args.load_policy_path:
        #     sim_policy = sim_policy.replace('1000000_.pkl', '2000000_mujoco_norm_.pkl')
        #     real_policy = real_policy.replace('1000000_.pkl', '2000000_mujoco_norm_.pkl')

        ##### Original
        # if 'mujoco_norm' in args.load_policy_path:
        #     sim_policy = sim_policy.replace('1000000_.pkl', '2000000_mujoco_norm_.pkl')
        #     real_policy = real_policy.replace('1000000_.pkl', '2000000_mujoco_norm_.pkl')
        #
        # elif 'normalized' in args.load_policy_path:
        #     sim_policy = sim_policy.replace('1000000_.pkl', '1000000_normalized_.pkl')
        #     real_policy = real_policy.replace('1000000_.pkl', '1000000_normalized_.pkl')

        if args.target_policy_algo == 'PPO2':
            algo = PPO2
        elif args.target_policy_algo == 'TRPO':
            algo = TRPO
        elif args.target_policy_algo == 'TRPO_lagrangian':
            algo = TRPO_lagrangian
        elif args.target_policy_algo == 'SAC':
            algo = SAC

        val = evaluate_policy_on_env(test_env,
                                     algo.load(sim_policy),
                                     render=False,
                                     iters=50,
                                     deterministic=True,
                                     constrained=constrained)
        with open(expt_path + "/green_red.txt", "a") as txt_file:
            print(val, file=txt_file)

        # red line
        del algo # remove the old algo and reload it.
        if args.target_policy_algo == 'PPO2':
            algo = PPO2
        elif args.target_policy_algo == 'TRPO':
            algo = TRPO
        elif args.target_policy_algo == 'TRPO_lagrangian':
            algo = TRPO_lagrangian
        elif args.target_policy_algo == 'SAC':
            algo = SAC

        val = evaluate_policy_on_env(test_env,
                                     algo.load(real_policy),
                                     render=False,
                                     iters=50,
                                     deterministic=True,
                                     constrained=constrained)
        with open(expt_path + "/green_red.txt", "a") as txt_file:
            print(val, file=txt_file)


    os._exit(0)

if __name__ == '__main__':
    main()
    os._exit(0)