import numpy as np
from numpy.linalg import norm
from Environments.ArrivalCar.CarDynamics import CarDynamics


class ArrivalCar:
    def __init__(self, action_min=np.array([0.75]), action_max=np.array([1.3]),
                 initial_state=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                 terminal_time=15, dt=0.05, inner_dt=0.01):

        self.state_dim = len(initial_state)
        self.action_dim = len(action_min)
        self.action_min = action_min
        self.action_max = action_max

        self.initial_state = initial_state
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_dt = inner_dt
        self.inner_step_n = int(dt / inner_dt)

        self.i = 0
        self.state = initial_state
        self.dynamics = CarDynamics()
        self.fx_array = []
        self.fz_array = []
        self.sigma_array = []
        self.trajectory_x = None
        self.trajectory_u = None
        self.scale_param = None
        self.mu = None

    @staticmethod
    def action_scaling(action, action_min, action_max):
        return np.clip(action, action_min, action_max)[0]

    def reset(self):
        self.state = self.initial_state
        return self.state

    def runge_kutta_step(self, action):
        extended_action = self.get_action(action)
        state = self.get_state()
        k1 = self.dynamics.f(state, extended_action, self.mu)
        k2 = self.dynamics.f(state + k1 * self.inner_dt / 2, extended_action, self.mu)
        k3 = self.dynamics.f(state + k2 * self.inner_dt / 2, extended_action, self.mu)
        k4 = self.dynamics.f(state + k3 * self.inner_dt, extended_action, self.mu)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6

    def get_state(self):
        return self.trajectory_x[self.i]

    def get_action(self, action):
        if not self.trajectory_u:
            raise ValueError('Trajectories should be obtained first.')
        self.mu = action
        return self.trajectory_u[self.i]

    def step(self, action):
        action = self.action_scaling(action, self.action_min, self.action_max)
        reward = 0
        for i in range(self.inner_step_n):
            self.i = i
            self.state = self.runge_kutta_step(action)
            reward += self.compute_reward()
            if self.is_terminal():
                self.fx_array.append(self.dynamics.fx)
                self.fz_array.append(self.dynamics.fz)
                self.sigma_array.append(self.dynamics.sigma)
                return self.state, reward, True, None

        return self.state, reward, False, None

    def is_terminal(self):
        return self.i == self.inner_step_n - 1

    def compute_reward(self):
        return -self.inner_dt * norm(self.state - self.trajectory_x[self.i + 1])

    def get_test_trajectory(self, mu):
        self.trajectory_x = [self.__get_test_initial_state()]
        self.trajectory_u = [self.__get_test_control()]*(self.inner_step_n+1)
        for i in range(self.inner_step_n):
            self.i = i
            self.trajectory_x.append(self.runge_kutta_step(mu))

    def get_real_trajectory(self):
        self.trajectory_x, self.trajectory_u = \
            self.dynamics.get_real_trajectory(self.inner_dt)

    def disturb_params(self):
        self.scale_param = self.dynamics.disturb_params()

    def restore_default_params(self):
        self.dynamics.load_constants()

    @staticmethod
    def __get_test_initial_state():
        return np.array([
            30, 0, 0, 0, 0,
            30 / 0.312 * (30 / np.pi * 1e-3), 30 / 0.312 * (30 / np.pi * 1e-3),
            30 / 0.350 * (30 / np.pi * 1e-3), 30 / 0.350 * (30 / np.pi * 1e-3)])

    @staticmethod
    def __get_test_control():
        return np.array([0, 0, 0.035 * 6, 0.035 * 6, 5 * np.pi / 180])
