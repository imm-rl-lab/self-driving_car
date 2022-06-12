import numpy as np
from Environments.ArrivalCar.ArrivalCar import ArrivalCar


class ArrivalCarSpeed(ArrivalCar):
    def __init__(self, action_min=np.array([0.]), action_max=np.array([4.]),
                 initial_state=np.array([1, 0, 0, 0, 0,
                                         1/0.312*30/np.pi*1e-3,
                                         1/0.312*30/np.pi*1e-3,
                                         1/0.350*30/np.pi*1e-3,
                                         1/0.350*30/np.pi*1e-3]),
                 terminal_time=15, dt=1, inner_dt=0.1):

        super().__init__(action_min, action_max, initial_state,
                         terminal_time, dt, inner_dt)

        self.terminal_distance = 200
        self.time = 0
        self.distance = 0
        self.mu = 1

        self.reset()

    def reset(self):
        super().reset()
        self.time = 0
        self.distance = 0
        return self.state

    def get_state(self):
        return self.state

    def get_action(self, action):
        return 0, 0, action, action, 0

    def compute_reward(self):
        self.time += self.inner_dt
        self.distance += self.state[0] * self.inner_dt
        return -self.inner_dt

    def is_terminal(self):
        return (self.time >= self.terminal_time) or \
               (self.distance >= self.terminal_distance)
