import numpy as np
import torch.nn as nn
from Agents.Utilities.SequentialNetwork import SequentialNetwork
from Agents.Utilities.Noises import DiscreteUniformNoise, UniformNoise, OUNoise
from Agents.Utilities.AgentTransformations import get_continuous_agent
from Solvers import TwoAgentSolver as solver
from Utilities.TwoAgentRecorder import TwoAgentRecorder as Recorder


class DGTests:
    def __init__(self, env, u_agent, v_agent, directory, test_attempt_n, 
                 u_action_values, v_action_values, u_action_values_for_BF=None, v_action_values_for_BF=None):
        self.env = env
        self.u_agent = u_agent
        self.v_agent = v_agent
        self.directory = directory
        self.test_attempt_n = test_attempt_n
        self.u_action_values = u_action_values
        self.v_action_values = v_action_values
        self.u_action_values_for_BF = u_action_values_for_BF
        self.v_action_values_for_BF = v_action_values_for_BF
        if u_action_values_for_BF is None:
            self.u_action_values_for_BF = self.u_action_values
        if v_action_values_for_BF is None:
            self.v_action_values_for_BF = self.v_action_values
        self.run_all_tests()
        return None
    
    def run_all_tests(self):
        self.run_one_test(of='u_agent', by='BF')
        self.run_one_test(of='u_agent', by='CCEM')
        self.run_one_test(of='u_agent', by='DCEM')
        self.run_one_test(of='u_agent', by='DQN')
        self.run_one_test(of='u_agent', by='DDPG')
        self.run_one_test(of='v_agent', by='BF')
        self.run_one_test(of='v_agent', by='CCEM')
        self.run_one_test(of='v_agent', by='DCEM')
        self.run_one_test(of='v_agent', by='DQN')
        self.run_one_test(of='v_agent', by='DDPG')
        return None
    
    def run_one_test(self, of='u_agent', by='BF'):
        for attempt in range(self.test_attempt_n):
            self.run_single_test(of=of, by=by, attempt=attempt)
        return None
    
    def run_single_test(self, of='u_agent', by='BF', attempt=0):
        
        #define tasted agent and tesing agent parameters
        if of=='u_agent':
            tested_agent = self.u_agent
            testing_action_min = self.env.v_action_min
            testing_action_max = self.env.v_action_max
            testing_action_dim = self.env.v_action_dim
            testing_action_values = self.v_action_values
            testing_action_values_for_BF = self.v_action_values_for_BF
        elif of=='v_agent':
            tested_agent = self.v_agent
            testing_action_min = self.env.u_action_min
            testing_action_max = self.env.u_action_max
            testing_action_dim = self.env.u_action_dim
            testing_action_values = self.u_action_values
            testing_action_values_for_BF = self.u_action_values_for_BF
        
        #define testing agent
        if by=='BF':
            piece_n = 8
            if len(testing_action_values_for_BF) ** piece_n < 500:
                from Agents.ProgramCounterAgent import ProgramCounterAgent as BF
                testing_learning = False
                episode_n, session_n = 5, 5
                testing_agent = BF(self.env, tested_agent, values=testing_action_values_for_BF, 
                                   piece_n=piece_n, agent_marker=by[0], session_n=5)
            else:
                print('too much for BF')
        elif by=='CCEM':
            from Agents.CCEM import CCEM
            testing_learning = True
            episode_n, session_n = 100, 20
            pi_model = SequentialNetwork([self.env.state_dim, 256, 128, testing_action_dim], nn.ReLU(), nn.Tanh())
            noise = UniformNoise(testing_action_dim, threshold_decrease=1/episode_n)
            testing_agent = CCEM(self.env.state_dim, testing_action_dim, 
                                 testing_action_min, testing_action_max, pi_model, noise, 
                                 percentile_param=70, pi_model_lr=1e-2, tau=1e-2, learning_iter_per_fit=16)
        elif by=='DCEM':
            from Agents.DCEM import DCEM
            testing_learning = True
            episode_n, session_n = 100, 20
            action_n = len(testing_action_values)
            pi_model = SequentialNetwork([self.env.state_dim, 128, 64, action_n], nn.ReLU())
            noise = DiscreteUniformNoise(action_n, threshold_decrease=1/episode_n)
            DCEM = get_continuous_agent(DCEM)
            testing_agent = DCEM(self.env.state_dim, action_n, pi_model, noise, 
                                 action_values=testing_action_values,
                                 percentile_param=70, pi_model_lr=1e-2, tau=1e-2, learning_iter_per_fit=16)
        elif by=='DQN':
            from Agents.DQN import DQN
            testing_learning = True
            episode_n, session_n = 500, 1
            action_n = len(testing_action_values)
            q_model = SequentialNetwork([self.env.state_dim, 128, 64, action_n], nn.ReLU())
            noise = DiscreteUniformNoise(action_n, threshold_decrease=1/episode_n)
            DQN = get_continuous_agent(DQN)
            testing_agent = DQN(self.env.state_dim, action_n, q_model, noise, 
                                action_values=testing_action_values, 
                                q_model_lr=1e-2, gamma=1, batch_size=64, tau=1e-2, learning_iter_per_fit=32)
        elif by=='DDPG':
            from Agents.DDPG import DDPG
            testing_learning = True
            episode_n, session_n = 500, 1
            q_model = SequentialNetwork([self.env.state_dim + testing_action_dim, 256, 128, 1], nn.ReLU())
            pi_model = SequentialNetwork([self.env.state_dim, 256, 128, testing_action_dim], nn.ReLU(), nn.Tanh())
            noise = OUNoise(testing_action_dim, threshold_decrease=1/episode_n)
            testing_agent = DDPG(self.env.state_dim, testing_action_dim, 
                         testing_action_min, testing_action_max, 
                         q_model, pi_model, noise, gamma=1, batch_size=256, tau=1e-2, 
                         q_model_lr=1e-3, pi_model_lr=1e-3, learning_iter_per_fit=16)
            
        #learning start
        #print(f'Testing of {of} by {by}, attempt: {attempt}')
        recorder = Recorder(f'{self.directory}/test_of_{of[0]}_by_{by}/attempt_{attempt}')
        
        if of=='u_agent':
            solver.go(self.env, tested_agent, testing_agent, episode_n=episode_n, session_n=session_n, 
                      show=recorder.record, u_learning=False, v_learning=testing_learning)
        elif of=='v_agent':
            solver.go(self.env, testing_agent, tested_agent, episode_n=episode_n, session_n=session_n, 
                      show=recorder.record, u_learning=testing_learning, v_learning=False)
        
        return None