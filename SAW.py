import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Dynamics Gradients (D3G)

class Model(nn.Module):
    def __init__(self, state_dim):
        super(Model, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, state_dim)

    def forward(self, state):
        latent = F.relu(self.l1(state))
        latent = F.relu(self.l2(latent))
        next_state = self.l3(latent)

        return next_state


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, is_discrete):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(2 * state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.is_discrete = is_discrete
                

    def forward(self, state, next_state):
        if len(state.shape) == 3:
            ss = torch.cat([state, next_state], 2)
        else:
            ss = torch.cat([state, next_state], 1)
        a = F.relu(self.l1(ss))
        a = F.relu(self.l2(a))

        if self.is_discrete:
            return torch.nn.Softmax()(self.l3(a))
        else:
            return self.max_action * torch.tanh(self.l3(a))

class ValueCritic(nn.Module):
    def __init__(self, state_dim):
        super(ValueCritic, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,1)
    
    def forward(self, state):
        value = F.relu(self.l1(state))
        value = F.relu(self.l2(value))
        value = self.l3(value)
        return value


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(2 * state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(2 * state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, next_state):
        if len(state.shape) == 3:
            ss = torch.cat([state, next_state], 2)
        else:
            ss = torch.cat([state, next_state], 1)

        q1 = F.relu(self.l1(ss))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(ss))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, next_state):
        if len(state.shape) == 3:
            ss = torch.cat([state, next_state], 2)
        else:
            ss = torch.cat([state, next_state], 1)

        q1 = F.relu(self.l1(ss))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1

def loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class SAW(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            is_discrete=False,
            expectile=0.7,
            temperature=10.0,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            num_samples=20,
            v=False,
    ):

        self.actor = Actor(state_dim, action_dim, max_action, is_discrete).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.value = ValueCritic(state_dim).to(device)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=3e-4)

        self.model = Model(state_dim).to(device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_discrete = is_discrete
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.expectile = expectile
        self.temperature = temperature
        self.num_samples = num_samples
        self.v = v
        
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        next_state = state + self.model(state).detach()
        # next_state = self.plan(state).detach() # can also use planning to select state
        action = self.actor(state, next_state).cpu().data.numpy().flatten()

        if self.is_discrete:
            action = np.argmax(action)

        return action

    def select_goal(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        next_state = state + self.model(state).detach()

        return next_state.cpu().data.numpy().flatten()
    
    def plan(self, state):
        # add planning to find the next good state
        pred_state = state + self.model(state)
        noise = torch.randn(self.num_samples, self.state_dim, device=device)*0.005
        
        cycle_state = (pred_state + noise).clamp(-1,1)
        for_state = torch.cat([cycle_state, pred_state], dim=0)
        state = state.repeat(self.num_samples+1, 1)
        value = torch.min(*self.critic(state, for_state)).detach()

        elite_value, elite_idxs = torch.topk(value.squeeze(1), 8, dim=0)
        elite_state = for_state[elite_idxs]

        max_value = elite_value.max(0)[0]
        score = torch.exp(2.0*(elite_value - max_value))
        score /= score.sum(0)

        score = score.cpu().numpy()
        
        output = elite_state[np.random.choice(np.arange(score.shape[0]), p=score)]
        return output.unsqueeze(0)


    def train(self, replay_buffer, online_buffer=None, batch_size=256, dynamics_only=False, logger=None, offline=False):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        self.total_it += 1
        # main loop
        self.update_v(state, next_state)
        self.update_q(reward, not_done, state, next_state, logger)
        self.update_actor_and_model(state, next_state, action, logger)
    
    def update_v(self, state, next_state):
        # update value critic
        with torch.no_grad():
            q1, q2 = self.critic_target(state, next_state)
            q = torch.minimum(q1, q2).detach()
        v = self.value(state)
        
        value_loss = loss(q - v, self.expectile).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
    
    def update_q(self, reward, not_done, state, next_state, logger):
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, next_state)

        target_v = (reward + not_done * self.discount * self.value(next_state)).detach()
        critic_loss = F.mse_loss(current_Q1, target_v) + F.mse_loss(current_Q2, target_v)

        if self.total_it % 1000 == 0:
            logger.log('train/current Q', current_Q1.mean(), self.total_it)
            logger.log('train/critic loss', critic_loss, self.total_it)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # polyak update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_actor_and_model(self, state, next_state, action, logger):
        aug_state = state.unsqueeze(0).repeat(self.num_samples,1,1)
        aug_next_state = next_state.unsqueeze(0).repeat(self.num_samples,1,1)
        action = action.unsqueeze(0).repeat(self.num_samples,1,1)
        noise = torch.randn(self.num_samples, state.shape[0], self.state_dim, device=device)*0.005
        aug_next_state = (aug_next_state + noise).clamp(-1,1)
        q1, q2 = self.critic(aug_state, aug_next_state)
        q = torch.minimum(q1, q2)
        v = self.value(aug_state)
        exp_v = torch.exp((q - v) * self.temperature)
        exp_v = torch.clamp(exp_v, max=100.0).squeeze(-1).detach()
        predicted_action = self.actor(aug_state, aug_next_state) # shape (num_sample, batch_size, action_dim)

        actor_loss = (exp_v.unsqueeze(-1) * (predicted_action - action)**2).mean()

        if self.total_it % 1000 == 0:
            logger.log('train/actor loss', actor_loss, self.total_it)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update model
        # value loss: find the best value
        hat_next_state  = state + self.model(state)
        model_loss = (exp_v.unsqueeze(-1) * (hat_next_state - next_state)**2).mean()

        if self.v:
            cycle_v = self.value(hat_next_state)
            lmbda = 1/cycle_v.abs().mean().detach()

            model_loss -= lmbda * cycle_v.mean()

        if self.total_it % 1000 == 0:
            logger.log('train/model loss', model_loss, self.total_it)
        
        # Optimize the model 
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.model.state_dict(), filename + "_model")
        torch.save(self.model_optimizer.state_dict(), filename + "_model_optimizer")
        torch.save(self.value.state_dict(), filename + "_value")
        torch.save(self.value_optim.state_dict(), filename + "_value_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.model.load_state_dict(torch.load(filename + "_model"))
        self.model_optimizer.load_state_dict(torch.load(filename + "_model_optimizer"))
        self.value.load_state_dict(torch.load(filename + "_value"))
        self.value_optim.load_state_dict(torch.load(filename + "_value_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
