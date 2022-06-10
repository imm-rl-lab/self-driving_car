import numpy as np
import json
from Environments.ArrivalCar.PacejkaTyreModelInterface import PacejkaTyreModelInterface
import os

class Parameters:

    def __init__(self, jsonfile):
        self.__dict__.update((k, v) for k, v in jsonfile.items() if not isinstance(v, dict))
        self.__dict__.update((k, Parameters(v)) for k, v in jsonfile.items() if isinstance(v, dict))


class ArrivalCarSpeed:
    def __init__(self, action_min=np.array([0.]), action_max=np.array([25.]),
                 initial_state=np.array([10, 0, 0, 0, 0, 10/0.312*30/np.pi*1e-3, 
                                         10/0.312*30/np.pi*1e-3, 10/0.350*30/np.pi*1e-3, 10/0.350*30/np.pi*1e-3]), mu=1,
                 terminal_time=15, terminal_distance=200, dt=1, inner_dt=0.1):

        self.state_dim = len(initial_state)
        self.action_dim = len(action_min)
        self.action_min = action_min
        self.action_max = action_max

        self.initial_state = initial_state
        self.terminal_time = terminal_time
        self.terminal_distance = terminal_distance
        self.dt = dt
        # self.dt = self.terminal_time
        self.inner_dt = inner_dt
        # self.inner_step_n = int(dt / inner_dt)
        self.inner_step_n = int(self.dt / self.inner_dt)
        self.time = 0
        self.distance = 0
        self.mu = mu

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

        self.load_constants()
        self.pacejka = PacejkaTyreModelInterface(
            self.param.PacejkaTyre.MILVehicleModelParameters)

    def load_constants(self):
        self.__rps2rpm = 30 / np.pi * 1e-3
        self.__g = 9.8015
        with open(os.path.join('Environments', 'ArrivalCar', 'Parameters_for_Problem3.json')) as json_file:
            params_dict = json.load(json_file)
            self.param = Parameters(params_dict)
            json_file.close()

    def f(self, state, action):
        # Inputs
        inputForceFL, inputForceFR, inputForceRL, inputForceRR, steer = [0, 0, action, action, 0]

        # States
        VLgt, VLat, YawRate, Yaw, MapE, OmFrntLe, OmFrntRi, OmReLe, OmReRi = state

        # calculate slips
        AlphaFrntLe = steer - np.arctan2(VLat + self.param.VehPrmVehCogDstFromAxleFrnt * YawRate,
                                         VLgt - self.param.VehPrmVehTrkWidthFrnt * YawRate / 2)
        AlphaFrntRi = steer - np.arctan2(VLat + self.param.VehPrmVehCogDstFromAxleFrnt * YawRate,
                                         VLgt + self.param.VehPrmVehTrkWidthFrnt * YawRate / 2)
        AlphaReLe = np.arctan2(-VLat + self.param.VehPrmVehCogDstFromAxleRe * YawRate,
                               VLgt - self.param.VehPrmVehTrkWidthRe * YawRate / 2)
        AlphaReRi = np.arctan2(-VLat + self.param.VehPrmVehCogDstFromAxleRe * YawRate,
                               VLgt + self.param.VehPrmVehTrkWidthRe * YawRate / 2)

        FzFL, FzFR, FzRL, FzRR = self.compute_vertical_loads(VLgt, -YawRate*VLat, YawRate*VLgt)

        self.FzRL = FzRL

        # Compute and limit maximum traction force
        LgtSlipFrntLe = (self.param.VehPrmTyrEfcRollgRdFrnt * OmFrntLe - VLgt) / VLgt
        LgtSlipFrntRi = (self.param.VehPrmTyrEfcRollgRdFrnt * OmFrntRi - VLgt) / VLgt
        LgtSlipReLe = (self.param.VehPrmTyrEfcRollgRdRe * OmReLe - VLgt) / VLgt
        LgtSlipReRi = (self.param.VehPrmTyrEfcRollgRdRe * OmReRi - VLgt) / VLgt


        self.sigReLe = LgtSlipReLe

        # Pacejka full model

        FxFL, FyFL = self.pacejka.get(self.mu, AlphaFrntLe, LgtSlipFrntLe, 1, FzFL, 1)
        FxFR, FyFR = self.pacejka.get(self.mu, AlphaFrntRi, LgtSlipFrntRi, 1, FzFR, 0)
        FxRL, FyRL = self.pacejka.get(self.mu, AlphaReLe, LgtSlipReLe, 2, FzRL, 1)
        FxRR, FyRR = self.pacejka.get(self.mu, AlphaReRi, LgtSlipReRi, 2, FzRR, 0)

        self.FxRL = FxRL
        self.AlphaRL = AlphaReLe
        
#         self.param.VehPrmVehMeasdDrgFLgtSpdVec = np.array([0, 9.72222233, 19.4444447,
#                                                            29.166666, 38.8888893, 48.6111107, 
#                                                            58.3333321, 68.0555573])

#         self.param.VehPrmVehMeasdDrgF = np.array([0, 714.285706, 1014.28571, 1457.14282, 2048.57153,
#                                                   2788.57153, 3674.28564, 4711.42871])
        
#         DrgF = np.interp(VLgt, self.param.VehPrmVehMeasdDrgFLgtSpdVec, self.param.VehPrmVehMeasdDrgF)

        # sum of force evaluation
        FxSum = FxRL + FxRR - (FyFL + FyFR) * np.sin(steer) + (FxFL + FxFR) * np.cos(
            steer) - VLgt ** 2 * self.param.VehPrmVehAeroDrgCoeff
        FySum = (FyFL + FyFR) * np.cos(steer) + FyRL + FyRR + (FxFL + FxFR) * np.sin(steer)
        MzSum = self.param.VehPrmVehCogDstFromAxleFrnt * (
                (FyFL + FyFR) * np.cos(steer) + (FxFL + FxFR) * np.sin(steer)) - self.param.VehPrmVehCogDstFromAxleRe * (
                        FyRL + FyRR) - 0.5 * self.param.VehPrmVehTrkWidthFrnt * (
                        (-FyFL + FyFR) * np.sin(steer) + FxRL - FxRR) + 0.5 * self.param.VehPrmVehTrkWidthFrnt * (
                        FxFR - FxFL) * np.cos(steer)

        # write ODE
        r_dot = MzSum / self.param.VehPrmVehYawMomJ
        Vx_dot = FxSum / self.param.VehPrmVehM + YawRate * VLat
        Vy_dot = FySum / self.param.VehPrmVehM - YawRate * VLgt

#         Yaw_dot = YawRate
        MapE_dot = VLat * np.cos(Yaw) + VLgt * np.sin(Yaw)

        omFL_dot = self.__rps2rpm * (self.param.VehPrmTyrEfcRollgRdFrnt / (self.param.VehPrmWhlJFrnt)) * (1000 * inputForceFL - FxFL)
        omFR_dot = self.__rps2rpm * (self.param.VehPrmTyrEfcRollgRdFrnt / (self.param.VehPrmWhlJFrnt)) * (1000 * inputForceFR - FxFR)
        omRL_dot = self.__rps2rpm * (self.param.VehPrmTyrEfcRollgRdRe / (self.param.VehPrmWhlJRe)) * (1000 * inputForceRL - FxRL)
        omRR_dot = self.__rps2rpm * (self.param.VehPrmTyrEfcRollgRdRe / (self.param.VehPrmWhlJRe)) * (1000 * inputForceRR - FxRR)

        x_out = np.array([Vx_dot, Vy_dot, r_dot, YawRate, MapE_dot, omFL_dot, omFR_dot, omRL_dot, omRR_dot])

        return x_out

    def compute_vertical_loads(self, Vx, Ax, Ay):
        u = [0]*8
        u[0] = self.param.VehPrmVehM * self.__g * self.param.VehPrmVehCogDstFromAxleRe \
               / (2 * self.param.VehPrmVehWhlBas)
        u[1] = self.param.VehPrmVehM * self.__g * self.param.VehPrmVehCogDstFromAxleFrnt \
               / (2 * self.param.VehPrmVehWhlBas)
        u[2] = self.param.VehPrmVehM * self.param.VehPrmVehCogHgt \
               / (2 * self.param.VehPrmVehWhlBas) * Ax
        u[3] = 0.5 * self.param.VehPrmVehAeroDrgCoeff * self.param.VehPrmVehCogHgt \
               / (2 * self.param.VehPrmVehWhlBas) * Vx ** 2
        u[4] = Ay * (self.param.VehPrmVehM *
                     (self.param.VehPrmVehCogDstFromAxleRe * self.param.VehPrmRollCeHgtFrnt
                      / self.param.VehPrmVehWhlBas + self.param.VehPrmVehRollStfnRatFrnt *
                      (self.param.VehPrmVehCogHgt - self.param.VehPrmRollCeHgtFrnt)) \
                      / self.param.VehPrmVehTrkWidthFrnt)
        u[5] = Ay * (self.param.VehPrmVehM *
                     (self.param.VehPrmVehCogDstFromAxleFrnt * self.param.VehPrmRollCeHgtRe \
                      / self.param.VehPrmVehWhlBas + (1 - self.param.VehPrmVehRollStfnRatFrnt) *
                      (self.param.VehPrmVehCogHgt - self.param.VehPrmRollCeHgtRe)) \
                      / self.param.VehPrmVehTrkWidthRe)
        u[6] = 0.5 * self.param.VehPrmLftCoeffFrnt / 2 * Vx ** 2
        u[7] = 0.5 * self.param.VehPrmLftCoeffRe / 2 * Vx ** 2

        FzFL = u[0] + u[6] - u[2] - u[3] - u[4]
        FzFR = u[0] + u[6] - u[2] - u[3] + u[4]
        FzRL = u[1] + u[7] + u[2] + u[3] - u[5]
        FzRR = u[1] + u[7] + u[2] + u[3] + u[5]

        return FzFL, FzFR, FzRL, FzRR

    def action_scaling(self, action):
        return np.clip(action, self.action_min, self.action_max)[0]

    def reset(self):
        self.state = self.initial_state
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

    def step(self, action):
        action = self.action_scaling(action)
        reward = 0
#         self.Fx_sigma = []
#         self.Fz_sigma = []
#         self.sigma_Fx = []

        for _ in range(self.inner_step_n):
            self.state = self.runge_kutta_step(action)
            self.time += self.inner_dt
            self.distance += self.state[0]*self.inner_dt
            reward += -self.inner_dt
#             self.Fx_sigma.append(self.FxRL)
#             self.Fz_sigma.append(self.FzRL)
#             self.sigma_Fx.append(self.sigReLe)
            self.state_array.append(self.state)
            # Fx(t)

            if (self.time >= self.terminal_time) or (self.distance >= self.terminal_distance):
                self.FxRL_array.append(self.FxRL)
#                 self.AlphaRL_array.append(self.AlphaRL)
                self.FzRL_array.append(self.FzRL)
                self.sigmaReLe_array.append(self.sigReLe)

                return self.state, reward, True, None

        return self.state, reward, False, None

