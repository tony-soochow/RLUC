
#
#
# env = NormalizedObservation(gym.make("Ant-v2"))
#
# test_episodes = 100
#
# basic_bm = copy.deepcopy(env.env.env.model.body_mass.copy())
# basic_bf = copy.deepcopy(env.env.env.model.geom_friction.copy())
#
# results = {}
#
# for mass in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
#     if mass not in results:
#         results[mass] = []
#     for _ in range(test_episodes):
#         for idx in range(len(basic_bf)):
#             # 修改质量
#             env.env.env.model.body_mass[idx] = basic_bm[idx] * mass
#
#
# print(basic_bf, len(basic_bf))
# print(basic_bm, len(basic_bm))

import pickle
import sqlite3
from policy_gradients.agent import Trainer
import git
import numpy as np
import gym
import os
import copy
import random
import argparse
from policy_gradients import models
from policy_gradients.torch_utils import ZFilter
import sys
import json
import torch
import torch as ch
import tqdm
import torch.optim as optim
from cox.store import Store, schema_from_dict
from run import main, add_common_parser_opts, override_json_params
from auto_LiRPA.eps_scheduler import LinearScheduler
import logging

logging.disable(logging.INFO)

import pybullet as p


"""class NormalizedObservation(gym.ObservationWrapper):
    def observation(self, observation):
        observation = (observation + 1) / 2  # [-1, 1] => [0, 1]
        observation *= (self.observation_space.high - self.observation_space.low)
        observation += self.observation_space.low
        return observation

    def _observation(self, observation):
        observation = (observation + 1) / 2  # [-1, 1] => [0, 1]
        observation *= (self.observation_space.high - self.observation_space.low)
        observation += self.observation_space.low
        return observation

    def _reverse_action(self, observation):
        observation -= self.observation_space.low
        observation /= (self.observation_space.high - self.observation_space.low)
        observation = observation * 2 - 1
        return observation"""


def main(params, mass, friction):
    override_params = copy.deepcopy(params)
    excluded_params = ['config_path', 'out_dir_prefix', 'num_episodes', 'row_id', 'exp_id',
                       'load_model', 'seed', 'deterministic', 'noise_factor', 'compute_kl_cert', 'use_full_backward',
                       'sqlite_path', 'early_terminate']
    sarsa_params = ['sarsa_enable', 'sarsa_steps', 'sarsa_eps', 'sarsa_reg', 'sarsa_model_path']
    imit_params = ['imit_enable', 'imit_epochs', 'imit_model_path', 'imit_lr']

    # original_params contains all flags in config files that are overridden via command.
    for k in list(override_params.keys()):
        if k in excluded_params:
            del override_params[k]



    # Append a prefix for output path.
    if params['out_dir_prefix']:
        params['out_dir'] = os.path.join(params['out_dir_prefix'], params['out_dir'])
        print(f"setting output dir to {params['out_dir']}")

    if params['config_path']:
        # Load from a pretrained model using existing config.
        # First we need to create the model using the given config file.
        json_params = json.load(open(params['config_path']))

        params = override_json_params(params, json_params, excluded_params + sarsa_params + imit_params)

    temp_envs = []
    if 'load_model' in params and params['load_model']:
        for k, v in zip(params.keys(), params.values()):
            assert v is not None, f"Value for {k} is None"

        # Create the agent from config file.
        p = Trainer.agent_from_params(params, store=None)
        print('Loading pretrained model', params['load_model'])
        pretrained_model = torch.load(params['load_model'])
        if 'policy_model' in pretrained_model:
            p.policy_model.load_state_dict(pretrained_model['policy_model'])
        if 'val_model' in pretrained_model:
            p.val_model.load_state_dict(pretrained_model['val_model'])
        if 'policy_opt' in pretrained_model:
            p.POLICY_ADAM.load_state_dict(pretrained_model['policy_opt'])
        if 'val_opt' in pretrained_model:
            p.val_opt.load_state_dict(pretrained_model['val_opt'])
        # Restore environment parameters, like mean and std.
        if 'envs' in pretrained_model:
            # p.envs = [NormalizedObservation(gym.make(params["game"]))]
            p.envs = pretrained_model['envs']
            temp_envs = p.envs
        for e in p.envs:
            e.normalizer_read_only = True
            e.setup_visualization(params['show_env'], params['save_frames'], params['save_frames_path'])


    rewards = []

    print('Gaussian noise in policy:')
    print(torch.exp(p.policy_model.log_stdev))
    original_stdev = p.policy_model.log_stdev.clone().detach()
    if params['noise_factor'] != 1.0:
        p.policy_model.log_stdev.data[:] += np.log(params['noise_factor'])
    if params['deterministic']:
        print('Policy runs in deterministic mode. Ignoring Gaussian noise.')
        p.policy_model.log_stdev.data[:] = -100
    print('Gaussian noise in policy (after adjustment):')
    print(torch.exp(p.policy_model.log_stdev))


    num_episodes = params['num_episodes']
    all_rewards = []
    all_lens = []

    p.envs = temp_envs
    for i in range(num_episodes):
        print('Episode %d / %d' % (i + 1, num_episodes))
        ep_length, ep_reward = p.eval_model(mass=mass, friction=friction)

        all_rewards.append(ep_reward)
        all_lens.append(ep_length)
    mean_reward, std_reward, min_reward, max_reward = np.mean(all_rewards), np.std(all_rewards), np.min(
        all_rewards), np.max(all_rewards)
    print('all rewards:', all_rewards)
    print(
        'rewards stats:\nmean: {}, std:{}, min:{}, max:{}'.format(mean_reward, std_reward, min_reward, max_reward))
    return mean_reward

def get_parser():
    parser = argparse.ArgumentParser(description='Generate experiments to be run.')
    parser.add_argument('--config-path', type=str, default='', required=False,
                        help='json for this config')
    parser.add_argument('--out-dir-prefix', type=str, default='', required=False,
                        help='prefix for output log path')
    parser.add_argument('--exp-id', type=str, help='experiement id for testing', default='')
    parser.add_argument('--row-id', type=int, help='which row of the table to use', default=-1)
    parser.add_argument('--num-episodes', type=int, help='number of episodes for testing', default=50)
    parser.add_argument('--compute-kl-cert', action='store_true', help='compute KL certificate')
    parser.add_argument('--use-full-backward', action='store_true',
                        help='Use full backward LiRPA bound for computing certificates')
    parser.add_argument('--deterministic', action='store_true', help='disable Gaussian noise in action for evaluation')
    parser.add_argument('--noise-factor', type=float, default=1.0,
                        help='increase the noise (Gaussian std) by this factor.')
    parser.add_argument('--load-model', type=str, help='load a pretrained model file', default='')
    parser.add_argument('--seed', type=int, help='random seed', default=1234)
    # Sarsa training related options.
    parser.add_argument('--sarsa-enable', action='store_true', help='train a sarsa attack model.')
    parser.add_argument('--sarsa-steps', type=int, help='Sarsa training steps.', default=30)
    parser.add_argument('--sarsa-model-path', type=str, help='path to save the sarsa value network.',
                        default='sarsa.model')
    parser.add_argument('--imit-enable', action='store_true', help='train a imit attack model.')
    parser.add_argument('--imit-epochs', type=int, help='Imit training steps.', default=100)
    parser.add_argument('--imit-model-path', type=str, help='path to save the imit policy network.',
                        default='imit.model')
    parser.add_argument('--imit-lr', type=float, help='lr for imitation learning training', default=1e-3)
    parser.add_argument('--sarsa-eps', type=float, help='eps for actions for sarsa training.', default=0.02)
    parser.add_argument('--sarsa-reg', type=float, help='regularization term for sarsa training.', default=0.1)
    # Other configs
    parser.add_argument('--sqlite-path', type=str, help='save results to a sqlite database.', default='')
    parser.add_argument('--early-terminate', action='store_true',
                        help='terminate attack early if low attack reward detected in sqlite.')
    parser = add_common_parser_opts(parser)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.load_model:
        assert args.config_path, "Need to specificy a config file when loading a pretrained model."

    if args.early_terminate:
        assert args.sqlite_path != '', "Need to specify --sqlite-path to terminate early."

    if args.sarsa_enable:
        if args.sqlite_path != '':
            print("When --sarsa-enable is specified, --sqlite-path and --early-terminate will be ignored.")

    params = vars(args)
    seed = params['seed']

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    masses = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    frictions = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]

    vertor = np.zeros((len(masses), len(frictions)))

    for i in range(len(masses)):
        for j in range(len(frictions)):
            print("-" * 80)
            print("mass: {}, friction: {}".format(masses[i], frictions[j]))
            vertor[i][j] = main(params, masses[i], frictions[j])
    # print(vertor)
    # !!!!!! 改变文件名称
    name = "halfcheetah_lip.csv"
    np.savetxt(name, vertor, fmt="%.2f", delimiter=',')
    # print("-"*80)
    get_vect = np.loadtxt(name, delimiter=',')
    print(get_vect)

