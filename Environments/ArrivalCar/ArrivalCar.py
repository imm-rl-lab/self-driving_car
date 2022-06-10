import numpy as np
import json
from numpy.linalg import norm
from Environments.ArrivalCar.PacejkaTyreModelInterface import PacejkaTyreModelInterface
from scipy.io import loadmat
from scipy.interpolate import interp1d
import os


class Parameters:
    def __init__(self, jsonfile):
        self.__dict__.update((k, v) for k, v in jsonfile.items() if not isinstance(v, dict))
        self.__dict__.update((k, Parameters(v)) for k, v in jsonfile.items() if isinstance(v, dict))


class ArrivalCar:
    def __init__(self, action_min=np.array([0.75]), action_max=np.array([1.3]),
                 initial_state=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                 terminal_time=15, dt=0.05, inner_dt=0.01):
        self.traj_x = []
        self.traj_u = []

        self.state_dim = len(initial_state)
        self.action_dim = len(action_min)
        self.action_min = action_min
        self.action_max = action_max

        self.initial_state = initial_state
        self.terminal_time = terminal_time
        self.dt = dt
        self.inner_dt = inner_dt
        self.inner_step_n = int(dt / inner_dt)

        self.reset()

        self.load_constants()
        self.pacejka = PacejkaTyreModelInterface(
            self.param.PacejkaTyre.MILVehicleModelParameters)

    def load_constants(self):
        self.i = 0
        self.rps2rpm = 30 / np.pi * 1e-3
        self.__g = 9.8015
        with open(os.path.join('Environments','ArrivalCar', 'ma2.json')) as json_file:
            params_dict = json.load(json_file)
            self.param = Parameters(params_dict)
            json_file.close()

    def f(self, state, action):
        u, steer = self.parse_actions(action)
        V, YawRate, Yaw, omega = self.parse_states(state)
        alpha = self.calculate_slips(steer, V, YawRate)
        Fz = self.calculate_vertical_loads(V, YawRate)
        slip = self.calculate_traction_force(omega, V[0])
        Fx, Fy = self.calculate_forces(alpha, slip, Fz)
        # self.param.VehPrmVehMeasdDrgFLgtSpdVec = np.array([0, 9.72222233, 19.4444447,
        #                                                    29.166666, 38.8888893, 48.6111107,
        #                                                    58.3333321, 68.0555573])
        # self.param.VehPrmVehMeasdDrgF = np.array([0, 714.285706, 1014.28571, 1457.14282, 2048.57153,
        #                                           2788.57153, 3674.28564, 4711.42871])
        # DrgF = np.interp(VLgt, self.param.VehPrmVehMeasdDrgFLgtSpdVec, self.param.VehPrmVehMeasdDrgF)
        forces_and_moment = self.summarize_forces(Fx, Fy, V[0], steer)
        return self.calculate_derivatives(V, YawRate, Yaw, u, Fx, forces_and_moment)

    def parse_actions(self, *args):
        self.mu = args[0]
        return self.traj_u[self.i][:4], self.traj_u[self.i][-1]

    def parse_states(self, *args):
        VLgt, VLat, YawRate, Yaw, _, = args[0][:5]
        omFL, omFR, omRL, omRR = args[0][5:] / self.rps2rpm
        return [VLgt, VLat], YawRate, Yaw, [omFL, omFR, omRL, omRR]

    def calculate_slips(self, steer, V, YawRate):
        aFL = steer - np.arctan2(
            V[1] + self.param.VehPrmVehCogDstFromAxleFrnt * YawRate,
            V[0] - self.param.VehPrmVehTrkWidthFrnt * YawRate / 2)
        aFR = steer - np.arctan2(
            V[1] + self.param.VehPrmVehCogDstFromAxleFrnt * YawRate,
            V[0] + self.param.VehPrmVehTrkWidthFrnt * YawRate / 2)
        aRL = np.arctan2(
            -V[1] + self.param.VehPrmVehCogDstFromAxleRe * YawRate,
            V[0] - self.param.VehPrmVehTrkWidthRe * YawRate / 2)
        aRR = np.arctan2(
            -V[1] + self.param.VehPrmVehCogDstFromAxleRe * YawRate,
            V[0] + self.param.VehPrmVehTrkWidthRe * YawRate / 2)
        return (aFL, aFR, aRL, aRR)

    def calculate_vertical_loads(self, V, YawRate):
        u = [0] * 8
        u[0] = (self.param.VehPrmVehM * self.__g *
                self.param.VehPrmVehCogDstFromAxleRe /
                (2 * self.param.VehPrmVehWhlBas))
        u[1] = (self.param.VehPrmVehM * self.__g *
                self.param.VehPrmVehCogDstFromAxleFrnt /
                (2 * self.param.VehPrmVehWhlBas))
        u[2] = (-self.param.VehPrmVehM * YawRate * V[1] *
                self.param.VehPrmVehCogHgt /
                (2 * self.param.VehPrmVehWhlBas))
        u[3] = (0.5 * V[0] ** 2 * self.param.VehPrmVehCogHgt *
                self.param.VehPrmVehAeroDrgCoeff /
                (2 * self.param.VehPrmVehWhlBas))
        u[4] = (YawRate * V[0] * self.param.VehPrmVehM *
                (self.param.VehPrmVehCogDstFromAxleRe *
                 self.param.VehPrmRollCeHgtFrnt /
                 self.param.VehPrmVehWhlBas +
                 self.param.VehPrmVehRollStfnRatFrnt *
                 (self.param.VehPrmVehCogHgt -
                  self.param.VehPrmRollCeHgtFrnt)) /
                self.param.VehPrmVehTrkWidthFrnt)
        u[5] = (YawRate * V[0] * self.param.VehPrmVehM *
                (self.param.VehPrmVehCogDstFromAxleFrnt *
                 self.param.VehPrmRollCeHgtRe /
                 self.param.VehPrmVehWhlBas +
                 (1 - self.param.VehPrmVehRollStfnRatFrnt) *
                 (self.param.VehPrmVehCogHgt -
                  self.param.VehPrmRollCeHgtRe)) /
                self.param.VehPrmVehTrkWidthRe)
        u[6] = 0.5 * self.param.VehPrmLftCoeffFrnt / 2 * V[0] ** 2
        u[7] = 0.5 * self.param.VehPrmLftCoeffRe / 2 * V[0] ** 2
        FzFL = u[0] + u[6] - u[2] - u[3] - u[4]
        FzFR = u[0] + u[6] - u[2] - u[3] + u[4]
        FzRL = u[1] + u[7] + u[2] + u[3] - u[5]
        FzRR = u[1] + u[7] + u[2] + u[3] + u[5]
        return (FzFL, FzFR, FzRL, FzRR)

    def calculate_traction_force(self, omega, Vx):
        slipFL = (self.param.VehPrmTyrEfcRollgRdFrnt * omega[0] - Vx) / Vx
        slipFR = (self.param.VehPrmTyrEfcRollgRdFrnt * omega[1] - Vx) / Vx
        slipRL = (self.param.VehPrmTyrEfcRollgRdRe * omega[2] - Vx) / Vx
        slipRR = (self.param.VehPrmTyrEfcRollgRdRe * omega[3] - Vx) / Vx
        return slipFL, slipFR, slipRL, slipRR

    def calculate_forces(self, alpha, slip, Fz):
        F = [self.pacejka.get(
            self.mu, alpha[i], slip[i], 1 + i // 2, Fz[i], (i + 1) % 2)
            for i in range(len(alpha))]
        return zip(*F)

    def summarize_forces(self, Fx, Fy, Vx, steer):
        FxSum = Fx[2] + Fx[3] - (Fy[0] - Fy[1]) * np.sin(steer) + \
                (Fx[0] + Fx[1]) * np.cos(steer) - \
                Vx ** 2 * self.param.VehPrmVehAeroDrgCoeff
        FySum = (Fy[0] + Fy[1]) * np.cos(steer) + Fy[2] + Fy[3] + \
                (Fx[0] + Fx[1]) * np.sin(steer)
        MzSum = self.param.VehPrmVehCogDstFromAxleFrnt * (
                (Fy[0] + Fy[1]) * np.cos(steer) + (Fx[0] + Fx[1]) * np.sin(steer)) - \
                self.param.VehPrmVehCogDstFromAxleRe * (Fy[2] + Fy[3]) - \
                0.5 * self.param.VehPrmVehTrkWidthFrnt * (
                        (Fy[1] - Fy[0]) * np.sin(steer) + Fx[2] - Fx[3]) + \
                0.5 * self.param.VehPrmVehTrkWidthFrnt * (Fx[1] - Fx[0]) * np.cos(steer)
        return (FxSum, FySum, MzSum)

    def calculate_derivatives(self, V, YawRate, Yaw, u, Fx, Sum):
        r_dot = Sum[2] / self.param.VehPrmVehYawMomJ
        Vx_dot = Sum[0] / self.param.VehPrmVehM + YawRate * V[1]
        Vy_dot = Sum[1] / self.param.VehPrmVehM - YawRate * V[0]
        MapE_dot = V[1] * np.cos(Yaw) + V[0] * np.sin(Yaw)
        omFL_dot = self.rps2rpm * (1000 * u[0] - Fx[0]) * \
                   self.param.VehPrmTyrEfcRollgRdFrnt / (self.param.VehPrmWhlJFrnt)
        omFR_dot = self.rps2rpm * (1000 * u[1] - Fx[1]) * \
                   self.param.VehPrmTyrEfcRollgRdFrnt / (self.param.VehPrmWhlJFrnt)
        omRL_dot = self.rps2rpm * (1000 * u[2] - Fx[2]) * \
                   self.param.VehPrmTyrEfcRollgRdRe / (self.param.VehPrmWhlJRe)
        omRR_dot = self.rps2rpm * (1000 * u[3] - Fx[3]) * \
                   self.param.VehPrmTyrEfcRollgRdRe / (self.param.VehPrmWhlJRe)

        return np.array([Vx_dot, Vy_dot, r_dot, YawRate, MapE_dot,
                         omFL_dot, omFR_dot, omRL_dot, omRR_dot])

    @staticmethod
    def action_scaling(self, action, action_min, action_max):
        return np.clip(action, action_min, action_max)[0]

    def reset(self):
        self.state = self.initial_state
        return self.state

    def runge_kutta_step(self, action):
        k1 = self.f(self.traj_x[self.i], action)
        k2 = self.f(self.traj_x[self.i] + k1 * self.inner_dt / 2, action)
        k3 = self.f(self.traj_x[self.i] + k2 * self.inner_dt / 2, action)
        k4 = self.f(self.traj_x[self.i] + k3 * self.inner_dt, action)
        return self.traj_x[self.i] + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6

    def step(self, action):
        action = self.action_scaling(action, self.action_min, self.action_max)
        reward = 0

        for i in range(self.inner_step_n):
            self.i = i
            self.state = self.runge_kutta_step(action)
            reward += -self.inner_dt * norm(self.state - self.traj_x[i + 1])

        return self.state, reward, True, None

    def get_trajectory(self, mu):
        delta = 1 * 5 * np.pi / 180

        x = np.array([30, 0, 0, 0, 0,
                      30 / 0.312 * self.rps2rpm, 30 / 0.312 * self.rps2rpm,
                      30 / 0.350 * self.rps2rpm, 30 / 0.350 * self.rps2rpm])
        u = np.array([0, 0, 0.035 * 6, 0.035 * 6, delta])

        self.traj_x = [x]
        self.traj_u = [u for _ in range(self.inner_step_n + 1)]

        for i in range(self.inner_step_n):
            self.i = i
            self.traj_x.append(self.runge_kutta_step(mu))

    def get_arrival_trajectory(self):
        ForceAnnots = loadmat('Environments\ArrivalCar\Inputs.mat')
        RealTrjAnnots = loadmat('Environments\ArrivalCar\RealTrj.mat')

        Forces = ForceAnnots['ForwardForce'][:, 1:] - ForceAnnots['BrakingForce'][:, 1:]

        Vx = RealTrjAnnots['Vx']
        Vy = RealTrjAnnots['Vy']
        MapE = np.zeros((len(Vx), 1))

        x = np.hstack((Vx[:, 1:], Vy[:, 1:], RealTrjAnnots['YawRate'][:, 1:],
                       RealTrjAnnots['Yaw'][:, 1:], MapE, RealTrjAnnots['OmWhl'][:, 1:]))
        u = np.hstack([Forces, ForceAnnots['Steer'][:, 1:]])

        dt = np.linspace(Vx[0, 0], Vx[-1, 0], int((Vx[-1, 0] - Vx[0, 0]) / self.inner_dt) + 1)

        self.traj_x = interp1d(Vx[:, 0], x, axis=0)(dt)
        self.traj_u = interp1d(ForceAnnots['Steer'][:, 0], u, axis=0)(dt)

    def disturb_params(self):
        params_to_disturb = ['VehPrmVehM', 'VehPrmVehYawMomJ', 'VehPrmVehBodyMFrntLe',
                             'VehPrmWhlJFrnt', 'VehPrmWhlJRe', 'VehPrmTyrEfcRollgRdFrnt',
                             'VehPrmVehAeroDrgCoeff', 'VehPrmVehRollStfnRatFrnt',
                             'VehPrmVehRollStfnRatFrnt', 'VehPrmTyrStfnLatFrnt',
                             'VehPrmTyrStfnLatRe', 'VehPrmTyrStfnLgtFrnt',
                             'VehPrmTyrStfnLgtRe']

        self.scale_param = np.random.uniform(.9, 1.1, len(params_to_disturb))
        for param, scale_factor in zip(params_to_disturb, self.scale_param):
            self.param.__dict__[param] *= scale_factor

    def get_disturbed_trajectory(self, mu):
        self.disturb_params()
        self.get_trajectory(mu)
        # back to default params
        self.load_constants()
