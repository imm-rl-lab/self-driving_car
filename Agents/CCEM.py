import numpy as np
from copy import deepcopy
import torch
from Agents.Utilities.LinearTransformations import transform_interval


class CCEM(torch.nn.Module):

    def __init__(self, state_dim, action_dim, action_min, action_max, pi_model, noise,
                 percentile_param=70, pi_model_lr=1e-2, tau=1e-2, learning_iter_per_fit=16):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = torch.FloatTensor(action_min)
        self.action_max = torch.FloatTensor(action_max)
        self.noise = noise
        self.pi_model = pi_model
        self.percentile_param = percentile_param
        self.tau = tau
        self.learning_iter_per_fit = learning_iter_per_fit
        self.optimizer = torch.optim.Adam(params=self.pi_model.parameters(), lr=pi_model_lr)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.pi_model(state)
        action = action + torch.FloatTensor(self.noise.get())
        action = transform_interval(action, self.action_min, self.action_max)
        return np.clip(action.detach().numpy(), self.action_min.numpy(), self.action_max.numpy())

    def get_elite_sessions(self, sessions):
        
        total_rewards = [session['total_reward'] for session in sessions]
        reward_threshold = np.percentile(total_rewards, self.percentile_param)
        elite_sessions = []

        for session in sessions:
            if session['total_reward'] >= reward_threshold:
                elite_sessions.append(session)

        return elite_sessions
    
    def update_policy(self, elite_sessions):
        elite_states, elite_actions = [], []
        for session in elite_sessions:
            elite_states.extend(session['states'])
            elite_actions.extend(session['actions'])
        elite_states = torch.FloatTensor(np.array(elite_states))
        elite_actions = torch.FloatTensor(np.array(elite_actions))
        
        for _ in range(self.learning_iter_per_fit):
            predicted_action = transform_interval(self.pi_model(elite_states), self.action_min, self.action_max)
            loss = torch.mean((predicted_action - elite_actions) ** 2)
            self.optimizer.zero_grad()
            old_pi_model = deepcopy(self.pi_model)
            loss.backward()
            self.optimizer.step()

            for new_parameter, old_parameter in zip(self.pi_model.parameters(), old_pi_model.parameters()):
                new_parameter.data.copy_(self.tau * new_parameter + (1 - self.tau) * old_parameter)

        return None

    def fit(self, sessions):
        for session in sessions:
            session['states'] = session['states'][:-1]
            session['total_reward'] = sum(session['rewards'])
        
        elite_sessions = self.get_elite_sessions(sessions)
        if len(sessions) != len(elite_sessions):
            self.update_policy(elite_sessions)
        return None
