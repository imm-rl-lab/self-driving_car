import numpy as np
import torch.nn as nn
from Agents.CCEM import CCEM
from Agents.Utilities.SequentialNetwork import SequentialNetwork
from Agents.Utilities.Noises import UniformNoise
from Solvers import OneAgentSolver as solver
from Environments.ArrivalCar.ArrivalCar import ArrivalCar
from Environments.ArrivalCar.ArrivalCarVisualizer import ArrivalCarVisualizer


if __name__ == "__main__":
    n_experiments = 10
    env = ArrivalCar(dt=1, inner_dt=0.1)
    predicted_mu = []
    for mu in np.linspace(0.75, 1.3, 12):
        print(f'processing mu: {mu}')
        env.get_trajectory(mu)
        mus = []
        for i in range(n_experiments):
            print(i)
            env.disturb_params()
            pi_model = SequentialNetwork([env.state_dim, 32, env.action_dim], nn.ReLU(), nn.Tanh())
            noise = UniformNoise(env.action_dim, threshold_decrease=1/150)
            agent = CCEM(env.state_dim, env.action_dim, env.action_min, env.action_max, pi_model, noise,
                         percentile_param=60,  tau=1e-2, pi_model_lr=1e-2, learning_iter_per_fit=16)

            visualizer = ArrivalCarVisualizer(waiting_for_show=10)
            solver.go(env, agent, episode_n=151, session_n=10, session_len=1, show=visualizer.show)
            mus.append(agent.get_action(env.initial_state)[0])
            env.restore_default_params()

        predicted_mu.append(mus)

    with open(f'results.npy', 'wb') as f:
        np.save(f, predicted_mu)





