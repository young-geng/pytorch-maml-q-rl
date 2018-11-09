import random

import maml_rl.envs
import gym
import numpy as np
import torch
import json

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler

from tensorboardX import SummaryWriter

def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
        'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
        '2DNavigation-v0'])

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
        num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    metalearner = MetaLearner(
        sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau,
        q_inner=args.inner_q == 'true',
        q_residuce_gradient=args.inner_q_residue_gradient == 'true',
        q_soft=args.inner_q_soft == 'true',
        q_soft_temp=args.inner_q_soft_temp,
        device=args.device,
    )

    for batch in range(args.num_batches):
        if args.device.type == 'cuda':
            torch.cuda.empty_cache()
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        episodes, adaptation_info = metalearner.sample(tasks, first_order=args.first_order)
        metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
            cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
            ls_backtrack_ratio=args.ls_backtrack_ratio)

        # Tensorboard
        pre_update_rewards = total_rewards([ep.rewards for ep, _ in episodes])
        post_update_rewards = total_rewards([ep.rewards for _, ep in episodes])
        
        writer.add_scalar('total_rewards/before_update', pre_update_rewards, batch)
        writer.add_scalar('total_rewards/after_update', post_update_rewards, batch)
        writer.add_scalar('total_rewards/rewards_improvement', post_update_rewards - pre_update_rewards, batch)
            
        writer.add_scalar('adaptation/pre_update_inner_loss', adaptation_info.mean_pre_update_loss, batch)
        writer.add_scalar('adaptation/post_update_inner_loss', adaptation_info.mean_post_update_loss, batch)
        writer.add_scalar('adaptation/inner_loss_improvement', adaptation_info.mean_loss_improvment, batch)
        writer.add_scalar('adaptation/weight_change', adaptation_info.mean_weight_change, batch)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--seed', type=int, default=42,
        help='random seed')
    parser.add_argument('--env-name', type=str,
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')
        
    parser.add_argument('--inner-steps', type=int, default=1,
        help='number of inner loop gradient steps')
        
    
    parser.add_argument('--inner-q', choices=('true', 'false'), default='false',
        help='use q learning loss for inner loop')
    
    parser.add_argument('--inner-q-residue-gradient',
        choices=('true', 'false'), default='false',
        help='use residue gradient for inner loop q loss')
        
    parser.add_argument('--inner-q-soft',
        choices=('true', 'false'), default='false',
        help='use soft q learning for inner loop')
        
    parser.add_argument('--inner-q-soft-temp', type=float, default=1.0,
        help='value of the soft q learning temperature for inner loop')
        

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
