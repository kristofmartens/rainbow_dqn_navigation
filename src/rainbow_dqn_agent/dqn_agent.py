import os
import random
import torch
from torch import optim

from model import QNetwork


class DQNAgent:
    def __init__(self,
                 env,
                 model_name=None,
                 discount_rate=0.99,
                 batch_size=32,
                 learning_rate=0.00001,
                 adam_epsilon=1e-8,
                 multi_step=3,
                 value_dist_size=51,
                 v_min=-10,
                 v_max=10,
                 device='cpu'):
        self.action_space = env.action_space.n
        self.observation_space = env.observation_space
        self.value_dist_size = value_dist_size
        self.Vmin = v_min
        self.Vmax = v_max
        self.support = torch.linspace(v_min, v_max, self.value_dist_size).to(device=device)  # Support (range) of z
        self.delta_z = (v_max - v_min) / (value_dist_size - 1)
        self.batch_size = batch_size
        self.n = multi_step
        self.discount = discount_rate

        self.online_net = QNetwork(env).to(device=device)
        if model_name and os.path.isfile(model_name):
            # Always load tensors onto CPU by default, will shift to GPU if necessary
            self.online_net.load_state_dict(torch.load(model_name, map_location='cpu'))
        self.online_net.train()

        self.target_net = QNetwork(env).to(device=device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=learning_rate, eps=adam_epsilon)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return random.randrange(self.action_space) if random.random() < epsilon else self.act(state)

    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(
                1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(
                self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(
                0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.value_dist_size - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.value_dist_size)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.value_dist_size), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.value_dist_size).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimiser.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path):
        torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
