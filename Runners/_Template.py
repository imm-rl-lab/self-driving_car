import numpy as np
import torch.nn as nn
import argparse, torch
from time import time
import os, sys
sys.path.insert(0, os.path.abspath('..'))

from Resolvers import OneAgentSolver as solver
from Utilities.OneAgentRecorder import OneAgentRecorder as Recorder

#YOUR IMPORTS HERE

def run(directory, dt, lr, episode_n=201):

    #Environments
    env = #YOUR ENV HERE

    #Agent
    agent = #YOUR AGENT HERE

    #Learning
    print('Learning')
    recorder = Recorder(directory)
    solver.go(env, agent, episode_n=episode_n, show=recorder.record)
    
    #Recording
    print('Recording')
    torch.save(pi_model, directory + '/pi_model.pt')
    
    #Finish
    print('Finish')
    return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str, default='0')
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    run(args.directory, args.dt, args.lr)