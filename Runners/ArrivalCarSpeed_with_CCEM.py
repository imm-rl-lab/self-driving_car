import numpy as np
import torch.nn as nn
import argparse, torch, random
from time import time
import os, sys, importlib
sys.path.insert(0, os.path.abspath('..'))

from Agents.CCEM import CCEM
from Agents.Utilities.SequentialNetwork import SequentialNetwork
from Agents.Utilities.Noises import UniformNoise
from Solvers import OneAgentSolver as solver
from Environments.ArrivalCar.ArrivalCarSpeed import ArrivalCarSpeed
from Environments.ArrivalCar.ArrivalCar import ArrivalCar
from Utilities.OneAgentRecorder import OneAgentRecorder as Recorder

def run(directory, env_name, agent_name, dt, p, lr, tau, lrpf, bs, en, attempt):
    
    random.seed(attempt)
    torch.manual_seed(attempt)
    np.random.seed(attempt)
    
    #Environments
    # env_path = 'Environments.' + env_name + '.' + env_name
    # env = getattr(importlib.import_module(env_path), env_name)(dt=dt)
    env = ArrivalCarSpeed(dt=1, inner_dt=0.1)
    env.get_test_trajectory(0.8)


    # Agent
    pi_model = SequentialNetwork([env.state_dim, 128, env.action_dim], nn.ReLU(), nn.Tanh())
    noise = UniformNoise(env.action_dim, threshold_decrease=1/en)
    agent = CCEM(env.state_dim, env.action_dim, env.action_min, env.action_max, pi_model, noise, 
             percentile_param=p,  tau=tau, pi_model_lr=lr, learning_iter_per_fit=lrpf)
    
    # Learning
    print('Learning')
    recorder = Recorder(directory)
    solver.go(env, agent, episode_n=en+1, session_n=3, session_len=101, show=recorder.record)
    
    # Recording
    #print('Recording')
    #torch.save(nu_model, directory + '/nu_model.pt')
    #torch.save(v_model, directory + '/v_model.pt')
    #torch.save(p_model, directory + '/p_model.pt')
    
    # Finish
    print(f'Finish env={env_name}, agent={agent_name}, attempt={attempt}')
    return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='0')
    parser.add_argument('--env_name', type=str, default='0')
    parser.add_argument('--agent_name', type=str, default='0')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--p', type=float, default=60)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--tau', type=float, default=1e-2)
    parser.add_argument('--lrpf', type=int, default=16)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--en', type=int, default=10)
    parser.add_argument('--attempt', type=int, default=10)
    args = parser.parse_args()
    run(args.directory, args.env_name, args.agent_name, args.dt, args.p, args.lr, args.tau, args.lrpf, args.bs, args.en, args.attempt)