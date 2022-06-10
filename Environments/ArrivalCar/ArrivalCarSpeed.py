import numpy as np
from Environments.ArrivalCar.ArrivalCar import ArrivalCar, Parameters


class ArrivalCarSpeed(ArrivalCar):
    def __init__(self, action_min=np.array([0.]), action_max=np.array([25.]),
                 initial_state=np.array([10, 0, 0, 0, 0,
                                         10/0.312*30/np.pi*1e-3,
                                         10/0.312*30/np.pi*1e-3,
                                         10/0.350*30/np.pi*1e-3,
                                         10/0.350*30/np.pi*1e-3]),
                 terminal_time=15, dt=1, inner_dt=0.1):

        super().__init__(action_min, action_max, initial_state,
                         terminal_time, dt, inner_dt)

        self.terminal_distance = 200
        self.time = 0
        self.distance = 0
        self.mu = 1

        self.FxRL = 0
#         self.AlphaRL = 0
        self.FxRL_array = []
#         self.AlphaRL_array = []
        self.FzRL = 0
        self.FzRL_array = []
        self.sigReLe = 0
        self.sigmaReLe_array = []
        self.state_array = []

        self.reset()

    def parse_actions(self, *args):
        return (0, 0, args[0], args[0]), 0

    def parse_states(self, *args):
        VLgt, VLat, YawRate, Yaw, _,  = args[0][:5]
        omFL, omFR, omRL, omRR = args[0][5:] / self.rps2rpm
        return [VLgt, VLat], YawRate, Yaw, [omFL, omFR, omRL, omRR]

    def reset(self):
        super().reset()
        self.time = 0
        self.distance = 0
        self.state_array = []
        return self.state

    def runge_kutta_step(self, action):
        k1 = self.f(self.state, action)
        k2 = self.f(self.state + k1 * self.inner_dt / 2, action)
        k3 = self.f(self.state + k2 * self.inner_dt / 2, action)
        k4 = self.f(self.state + k3 * self.inner_dt, action)
        return self.state + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6

    def f(self, state, action):
        u, steer = self.parse_actions(action)
        V, YawRate, Yaw, omega = self.parse_states(state)
        alpha = self.calculate_slips(steer, V, YawRate)
        Fz = self.calculate_vertical_loads(V, YawRate)
        slip = self.calculate_traction_force(omega, V[0])
        Fx, Fy = self.calculate_forces(alpha, slip, Fz)
        forces_and_moment = self.summarize_forces(Fx, Fy, V[0], steer)
        self.save_params(Fx, Fz, alpha, slip)
        return self.calculate_derivatives(V, YawRate, Yaw, u, Fx, forces_and_moment)

    def save_params(self, Fx, Fz, alpha, slip):
        self.FzRL = Fz[2]
        self.sigReLe = slip[2]
        self.FxRL = Fx[2]
        self.AlphaRL = alpha[2]

    def step(self, action):
        action = self.action_scaling(action, self.action_min, self.action_max)
        reward = 0
        # self.Fx_sigma = []
        # self.Fz_sigma = []
        # self.sigma_Fx = []
        for _ in range(self.inner_step_n):
            self.state = self.runge_kutta_step(action)
            self.time += self.inner_dt
            self.distance += self.state[0] * self.inner_dt
            reward += -self.inner_dt
            # self.Fx_sigma.append(self.FxRL)
            # self.Fz_sigma.append(self.FzRL)
            # self.sigma_Fx.append(self.sigReLe)
            self.state_array.append(self.state)  # Fx(t)
            if (self.time >= self.terminal_time) or (self.distance >= self.terminal_distance):
                self.FxRL_array.append(self.FxRL)
                # self.AlphaRL_array.append(self.AlphaRL)
                self.FzRL_array.append(self.FzRL)
                self.sigmaReLe_array.append(self.sigReLe)
                return self.state, reward, True, None

        return self.state, reward, False, None
