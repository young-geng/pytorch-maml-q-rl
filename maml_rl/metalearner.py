import numpy as np

from attrdict import AttrDict

import torch
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient
from maml_rl.policies import CategoricalMLPPolicy


class MetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, inner_steps=1, q_inner=False,
                 q_residuce_gradient=False, q_soft=False, q_soft_temp=1.0,
                 device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        
        self.inner_steps = inner_steps
        
        self.q_inner = q_inner
        self.q_residuce_gradient = q_residuce_gradient
        self.q_soft = q_soft
        self.q_soft_temp = q_soft_temp
        
        self.to(device)
        
    def inner_loss(self, episodes, params=None):
        if self.q_inner:
            return self.inner_q_loss(episodes, params)
        else:
            return self.inner_pg_loss(episodes, params)

    def inner_pg_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, weights=episodes.mask)

        return loss
        
    def inner_q_loss(self, episodes, params=None):
        assert isinstance(self.policy, CategoricalMLPPolicy)
        
        observations, actions, rewards, next_observations, dones = (
            episodes.flattened_transitions
        )
        current_q_values = self.policy.forward_logits(observations, params)
        current_action_q_values = current_q_values[
            torch.arange(current_q_values.shape[0]), actions
        ]
        
        next_q_values = self.policy.forward_logits(next_observations, params)
        
        if self.q_soft:
            def logsumexp(inputs, dim=None, keepdim=False):
                return (
                    (inputs - F.log_softmax(inputs)).mean(dim, keepdim=keepdim)
                )
            
            next_max_q_values = self.q_soft_temp * logsumexp(
                next_q_values / self.q_soft_temp, 1
            )
            
        else:
            next_max_q_values, _ = torch.max(next_q_values, dim=1)
        
        if not self.q_residuce_gradient:
            next_max_q_values = next_max_q_values.detach()
            
        
        target_values = rewards + self.gamma * (1.0 - dones) * next_max_q_values
        
        loss = 0.5 * torch.mean((current_action_q_values - target_values) ** 2)
        
        return loss

    def adapt(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        
        params = None
        
        info = AttrDict()
        loss = self.inner_loss(episodes, params)
        
        info.pre_update_loss = loss.detach().cpu().numpy()
        
        for _ in range(self.inner_steps):
            # Get the new parameters after a one-step gradient update
            params = self.policy.update_params(
                loss, step_size=self.fast_lr, first_order=first_order,
                params=params
            )
            
            # Get the loss on the training episodes
            loss = self.inner_loss(episodes, params)
            
        info.post_update_loss = loss.detach().cpu().numpy()
        
        info.weight_change = torch.norm(
            parameters_to_vector(self.policy.parameters())
            - parameters_to_vector(params.values())
        ).detach().cpu().numpy()

        return params, info

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        episodes = []
        pre_update_losses = []
        post_update_losses = []
        weight_changes = []
        
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                gamma=self.gamma, device=self.device)

            params, adaptation_info = self.adapt(train_episodes, first_order=first_order)

            valid_episodes = self.sampler.sample(self.policy, params=params,
                gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
            
            pre_update_losses.append(adaptation_info.pre_update_loss)
            post_update_losses.append(adaptation_info.post_update_loss)
            weight_changes.append(adaptation_info.weight_change)
            
        info = AttrDict(
            mean_pre_update_loss=np.mean(pre_update_losses),
            mean_post_update_loss=np.mean(post_update_losses),
            mean_weight_change=np.mean(weight_changes)
        )
        
        info.mean_loss_improvment = info.mean_pre_update_loss - info.mean_post_update_loss
            
        return episodes, info

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params, _ = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params, _ = self.adapt(train_episodes)
            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                    weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions)
                    - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages,
                    weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
            damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
            cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device
