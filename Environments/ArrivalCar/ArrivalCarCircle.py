import copy
import numpy as np
import sys, os
from numpy.linalg import norm
sys.path.insert(0, os.path.abspath('..'))
from arrival_model.Environments.ArrivalCar.CarDynamics import CarDynamics


class ArrivalCarCircle:
    def __init__(self, action_min=np.array([0,0,0,0,0]), action_max=np.array([0,0,0,0,0]),
                 initial_state=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), 
                 initial_t=0, initial_x=0, initial_y=-25, initial_psi=0,
                 terminal_time=15, dt=0.05, inner_dt=0.01, circle_part=1/2):

        self.state_dim = 9
        self.action_dim = 2
        self.action_min = action_min
        self.action_max = action_max

        self.initial_state = initial_state
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_dt = inner_dt
        self.inner_step_n = int(dt / inner_dt)
        self.circle_part = circle_part

        self.dynamics = CarDynamics()
        self.mu = 1
        self.initial_info = {'ts': [initial_t], 'xs': [initial_x], 'ys': [initial_y], 'psis': [initial_psi]}
        return None

    def reset(self):
        self.state = self.initial_state
        self.info = copy.deepcopy(self.initial_info)
        self.half = False
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)
        extended_action = np.array([0, 0, action[0], action[0], action[1]])
        
        for _ in range(self.inner_step_n):
            
            self.state = self.runge_kutta_step(extended_action)
        
        self.fill_info(action)
        x, y = self.info['xs'][-1], self.info['ys'][-1]
        
        done = False
        if self.circle_part == 1/2 and x < 0 and y > 0:
            done = True
        if self.circle_part == 3/4 and x < 0 and y < 0:
            done = True
        if self.circle_part == 1:
            if x < 0 and y > 0:
                self.half = True
            if x > 0 and y < 0 and self.half:
                done = True
            
        reward = - self.dt
        if x ** 2 + y ** 2 < 20 ** 2:
            reward = - 2 * self.dt
        
        return self.state, reward, done, self.info
    
    def runge_kutta_step(self, action):
        k1 = self.dynamics.f(self.state, action, self.mu)
        k2 = self.dynamics.f(self.state + k1 * self.inner_dt / 2, action, self.mu)
        k3 = self.dynamics.f(self.state + k2 * self.inner_dt / 2, action, self.mu)
        k4 = self.dynamics.f(self.state + k3 * self.inner_dt, action, self.mu)
        return self.state + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6
    
    def fill_info(self, action):
        lf, lr = 0.87, 1.33
        delta = action[1]
        beta = np.arctan(lr * np.tan(delta) / (lf + lr))
        v = np.sqrt(self.state[0] ** 2 + self.state[1] ** 2)
        
        dpsi = v * np.cos(beta) * np.tan(delta) / (lf + lr)
        psi = self.info['psis'][-1] + dpsi * self.dt
        self.info['psis'].append(psi)
        
        dx = v * np.cos(psi + beta)
        x = self.info['xs'][-1] + dx * self.dt
        self.info['xs'].append(x)
        
        dy = v * np.sin(psi + beta)
        y = self.info['ys'][-1] + dy * self.dt
        self.info['ys'].append(y)
        return None

