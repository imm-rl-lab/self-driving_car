import numpy as np
import json
from numpy.linalg import norm
import matplotlib.pyplot as plt
from PacejkaTyreModelInterface import PacejkaTyreModelInterface
from scipy.io import loadmat
from scipy.interpolate import interp1d

class Parameters:

    def __init__(self, jsonfile):
        self.__dict__.update((k, v) for k, v in jsonfile.items() if not isinstance(v, dict))
        self.__dict__.update((k, Parameters(v)) for k, v in jsonfile.items() if isinstance(v, dict))

class Structure:

    def __init__(self):
        pass

    def set(self, name, value):
        self.__dict__.update({name: value})


class ArrivalCar:
    def __init__(self, traj_x=0, traj_u=0, action_min=np.array([0.75]), action_max=np.array([1.3]),
                 initial_state=np.array([0,0,0,0,0,0,0,0,0]),
                 terminal_time=15, dt=0.05, inner_dt=0.01):

        self.traj_x = traj_x
        self.traj_u = traj_u

        self.state_dim = len(initial_state)
        self.action_dim = len(action_min)
        self.action_min = action_min
        self.action_max = action_max

        self.initial_state = initial_state
        self.terminal_time = dt
        self.dt = dt
        self.inner_dt = inner_dt
        self.inner_step_n = int(dt / inner_dt)
        self.state = self.reset()

        self.load_constants()
        self.pacejka = PacejkaTyreModelInterface(self.param.PacejkaTyre.MILVehicleModelParameters)

    def load_constants(self):
        '''
        Постоянные в модели посадки самолета
        '''

        with open("ma2.json") as jsonFile:
            params_dict = json.load(jsonFile)
            self.param = Parameters(params_dict)
            jsonFile.close()

    def f(self, x_star, u_star, mu_road):

        rps2rpm = 30 / np.pi * 1e-3

        # Inputs
        inputForceFL, inputForceFR, inputForceRL, inputForceRR, steer = u_star

        # States
        VLgt, VLat, YawRate, Yaw, MapE, OmFrntLe, OmFrntRi, OmReLe, OmReRi = x_star

        OmFrntLe /= rps2rpm
        OmFrntRi /= rps2rpm
        OmReLe /= rps2rpm
        OmReRi /= rps2rpm

        # YawRate *= 0

        # calculate slips
        AlphaFrntLe = (steer - np.arctan2(VLat + self.param.VehPrmVehCogDstFromAxleFrnt * YawRate,
                                          VLgt - self.param.VehPrmVehTrkWidthFrnt * YawRate / 2))
        AlphaFrntRi = (steer - np.arctan2(VLat + self.param.VehPrmVehCogDstFromAxleFrnt * YawRate,
                                          VLgt + self.param.VehPrmVehTrkWidthFrnt * YawRate / 2))
        AlphaReLe = (np.arctan2(-VLat + self.param.VehPrmVehCogDstFromAxleRe * YawRate,
                                VLgt - self.param.VehPrmVehTrkWidthRe * YawRate / 2))
        AlphaReRi = (np.arctan2(-VLat + self.param.VehPrmVehCogDstFromAxleRe * YawRate,
                                VLgt + self.param.VehPrmVehTrkWidthRe * YawRate / 2))

        FzFL, FzFR, FzRL, FzRR = self.vertical_loads(VLgt, -YawRate * VLat, YawRate * VLgt)

        # Compute and limit maximum traction force
        LgtSlipFrntLe = (self.param.VehPrmTyrEfcRollgRdFrnt * OmFrntLe - VLgt) / VLgt
        LgtSlipFrntRi = (self.param.VehPrmTyrEfcRollgRdFrnt * OmFrntRi - VLgt) / VLgt
        LgtSlipReLe = (self.param.VehPrmTyrEfcRollgRdRe * OmReLe - VLgt) / VLgt
        LgtSlipReRi = (self.param.VehPrmTyrEfcRollgRdRe * OmReRi - VLgt) / VLgt

        # Pacejka full model

        FxFL, FyFL = self.pacejka.get(self.param.PacejkaTyre.MILVehicleModelParameters, mu_road, AlphaFrntLe,
                                      LgtSlipFrntLe, 1, FzFL, 1)
        FxFR, FyFR = self.pacejka.get(self.param.PacejkaTyre.MILVehicleModelParameters, mu_road, AlphaFrntRi,
                                      LgtSlipFrntRi, 1, FzFR, 0)
        FxRL, FyRL = self.pacejka.get(self.param.PacejkaTyre.MILVehicleModelParameters, mu_road, AlphaReLe,
                                               LgtSlipReLe, 2, FzRL, 1)
        FxRR, FyRR = self.pacejka.get(self.param.PacejkaTyre.MILVehicleModelParameters, mu_road, AlphaReRi,
                                               LgtSlipReRi, 2, FzRR, 0)

        # self.param.VehPrmVehMeasdDrgFLgtSpdVec = np.array([0, 9.72222233, 19.4444447,
        #                                                    29.166666, 38.8888893, 48.6111107, 58.3333321, 68.0555573])
        #
        # self.param.VehPrmVehMeasdDrgF = np.array([0, 714.285706, 1014.28571, 1457.14282, 2048.57153,
        #                                           2788.57153, 3674.28564, 4711.42871])

        # DrgF = np.interp(VLgt, self.param.VehPrmVehMeasdDrgFLgtSpdVec, self.param.VehPrmVehMeasdDrgF)

        # DrgFx = DrgF*VLgt/np.sqrt(VLgt**2 + VLat**2)
        # DrgFy = DrgF * VLat/np.sqrt(VLgt ** 2 + VLat ** 2)
        # sum of force evaluation
        FxSum = FxRL + FxRR - (FyFL + FyFR) * np.sin(steer) + (FxFL + FxFR) * np.cos(
            steer) - VLgt ** 2 * self.param.VehPrmVehAeroDrgCoeff #DrgF  # in order to computed and desired trajectories match each other
        #VLgt ** 2 * self.param.VehPrmVehAeroDrgCoeff #
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

        Yaw_dot = YawRate
        MapE_dot = VLat * np.cos(Yaw) + VLgt * np.sin(Yaw)

        omFL_dot = rps2rpm * (self.param.VehPrmTyrEfcRollgRdFrnt / (self.param.VehPrmWhlJFrnt)) * (1000 * inputForceFL - FxFL)
        omFR_dot = rps2rpm * (self.param.VehPrmTyrEfcRollgRdFrnt / (self.param.VehPrmWhlJFrnt)) * (1000 * inputForceFR - FxFR)
        omRL_dot = rps2rpm * (self.param.VehPrmTyrEfcRollgRdRe / (self.param.VehPrmWhlJRe)) * (1000 * inputForceRL - FxRL)
        omRR_dot = rps2rpm * (self.param.VehPrmTyrEfcRollgRdRe / (self.param.VehPrmWhlJRe)) * (1000 * inputForceRR - FxRR)

        DiagBus = Structure()

        DiagBus.set('VLgt', VLgt)
        DiagBus.set('steer', steer)
        DiagBus.set('VLat', VLat)
        DiagBus.set('YawRate', YawRate)
        DiagBus.set('Alpha', np.array([AlphaFrntLe, AlphaFrntRi, AlphaReLe, AlphaReRi]) * 180 / np.pi)
        DiagBus.set('Fz', np.array([FzFL, FzFR, FzRL, FzRR]))
        DiagBus.set('Fy', np.array([FyFL, FyFR, FyRL, FyRR]))
        DiagBus.set('Fx', np.array([FxFL, FxFR, FxRL, FxRR]))
        DiagBus.set('SideSlip', np.arctan2(VLat, VLgt)*180/np.pi)
        DiagBus.set('LgtSlip', np.array([LgtSlipFrntLe, LgtSlipFrntRi, LgtSlipReLe, LgtSlipReRi]))

        x_out = np.array([Vx_dot, Vy_dot, r_dot, Yaw_dot, MapE_dot, omFL_dot, omFR_dot, omRL_dot, omRR_dot])

        return x_out

    def vertical_loads(self, Vx, Ax, Ay):
        g = 9.8015
        u = [0]*8
        u[0] = self.param.VehPrmVehM * g * self.param.VehPrmVehCogDstFromAxleRe \
               / (2 * self.param.VehPrmVehWhlBas)
        u[1] = self.param.VehPrmVehM * g * self.param.VehPrmVehCogDstFromAxleFrnt \
               / (2 * self.param.VehPrmVehWhlBas)
        u[2] = 1 * self.param.VehPrmVehM * self.param.VehPrmVehCogHgt \
               / (2 * self.param.VehPrmVehWhlBas) * Ax
        u[3] = 0.5 * 1 * self.param.VehPrmVehAeroDrgCoeff * self.param.VehPrmVehCogHgt \
               / (2 * self.param.VehPrmVehWhlBas) * Vx ** 2
        u[4] = 1 * Ay * (self.param.VehPrmVehM *
                         (self.param.VehPrmVehCogDstFromAxleRe * self.param.VehPrmRollCeHgtFrnt
                          / self.param.VehPrmVehWhlBas + self.param.VehPrmVehRollStfnRatFrnt *
                          (self.param.VehPrmVehCogHgt - self.param.VehPrmRollCeHgtFrnt)) \
                         / self.param.VehPrmVehTrkWidthFrnt)
        u[5] = 1 * Ay * (self.param.VehPrmVehM *
                         (self.param.VehPrmVehCogDstFromAxleFrnt * self.param.VehPrmRollCeHgtRe \
                          / self.param.VehPrmVehWhlBas + (1 - self.param.VehPrmVehRollStfnRatFrnt) *
                          (self.param.VehPrmVehCogHgt - self.param.VehPrmRollCeHgtRe)) \
                          / self.param.VehPrmVehTrkWidthRe)
        u[6] = 0.5 * 1 * self.param.VehPrmLftCoeffFrnt / 2 * Vx ** 2
        u[7] = 0.5 * 1 * self.param.VehPrmLftCoeffRe / 2 * Vx ** 2

        FzFL = u[0] + u[6] - u[2] - u[3] - u[4]
        FzFR = u[0] + u[6] - u[2] - u[3] + u[4]
        FzRL = u[1] + u[7] + u[2] + u[3] - u[5]
        FzRR = u[1] + u[7] + u[2] + u[3] + u[5]

        return FzFL, FzFR, FzRL, FzRR

    def action_scaling(self, action, action_min, action_max):
        action = np.clip(action, action_min, action_max)

        return action[0]

    def reset(self):
        self.state = self.initial_state
        return self.state

    def step(self, action):
        action = self.action_scaling(action, self.action_min, self.action_max)
        reward = 0

        for i in range(self.inner_step_n):
            k1 = self.f(self.traj_x[i], self.traj_u[i], action)
            k2 = self.f(self.traj_x[i] + k1 * self.inner_dt / 2, self.traj_u[i], action)
            k3 = self.f(self.traj_x[i] + k2 * self.inner_dt / 2, self.traj_u[i], action)
            k4 = self.f(self.traj_x[i] + k3 * self.inner_dt, self.traj_u[i], action)
            self.state = self.traj_x[i] + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6
#             self.state = self.traj_x[i] + self.inner_dt*k1
            reward += -self.inner_dt*norm(self.state - self.traj_x[i+1])

        done = True
        

        # if self.state[0] >= self.terminal_time:
        #     reward = -norm(self.state[[1, 3]])
        #     done = True

        return self.state, reward, done, None
    
    def get_trajectory(self, mu):
        rps2rpm = 30/np.pi*1e-3
        delta = 0*5*np.pi/180
        
        x = np.array([30, 0, 0, 0, 0, 30/0.312*rps2rpm, 30/0.312*rps2rpm, 30/0.350*rps2rpm, 30/0.350*rps2rpm])
        u = np.array([0, 0, 0.035*6, 0.035*6, delta])

        traj_x = [x]
        traj_u = [u]
        
        for i in range(self.inner_step_n):
            k1 = self.f(x, u, mu)
            k2 = self.f(x + k1 * self.inner_dt / 2, u, mu)
            k3 = self.f(x + k2 * self.inner_dt / 2, u, mu)
            k4 = self.f(x + k3 * self.inner_dt, u, mu)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6
            traj_x.append(x)
            traj_u.append(u)

        self.traj_x = traj_x
        self.traj_u = traj_u

    def get_arrival_trajectory(self):

        ForceAnnots = loadmat('Inputs.mat')
        RealTrjAnnots = loadmat('RealTrj.mat')

        ForwardForce = ForceAnnots['ForwardForce']
        BrakingForce = ForceAnnots['BrakingForce']
        Steer = ForceAnnots['Steer']
        Forces = ForwardForce[:, 1:] - BrakingForce[:, 1:]

        OmWhl = RealTrjAnnots['OmWhl']
        Vx = RealTrjAnnots['Vx']
        Vy = RealTrjAnnots['Vy']
        Yaw = RealTrjAnnots['Yaw']
        YawRate = RealTrjAnnots['YawRate']
        MapE = np.zeros((len(Vx), 1))

        x = np.hstack((Vx[:, 1:], Vy[:, 1:], YawRate[:, 1:], Yaw[:, 1:], MapE, OmWhl[:, 1:]))
        u = np.hstack([Forces, Steer[:, 1:]])

        dt = np.linspace(Vx[0, 0], Vx[-1, 0], int((Vx[-1, 0] - Vx[0, 0]) / self.inner_dt) + 1)

        self.traj_x = interp1d(Vx[:, 0], x, axis=0)(dt)
        self.traj_u = interp1d(Steer[:, 0], u, axis=0)(dt)

        # self.traj_x = np.interp(dt, Vx[:, 0], x)
        # self.traj_u = np.interp(dt, Steer[:, 0], u)


    def get_disturbed_trajectory(self, mu):
        # disturb params
        params_to_disturb = ['VehPrmVehM', 'VehPrmVehYawMomJ', 'VehPrmVehBodyMFrntLe',
                             'VehPrmWhlJFrnt', 'VehPrmWhlJRe', 'VehPrmTyrEfcRollgRdFrnt',
                             'VehPrmVehAeroDrgCoeff', 'VehPrmVehRollStfnRatFrnt',
                             'VehPrmVehRollStfnRatFrnt', 'VehPrmTyrStfnLatFrnt',
                             'VehPrmTyrStfnLatRe', 'VehPrmTyrStfnLgtFrnt',
                             'VehPrmTyrStfnLgtRe']
        
        self.scale_param = np.random.uniform(.90, 1.1, len(params_to_disturb))
        for param, scale_factor in zip(params_to_disturb, self.scale_param):
            self.param.__dict__[param] *= scale_factor
        
        # get trajectory with disturbed params
        self.get_trajectory(mu)
        
        # back to default params
        self.load_constants()
        
    def plot(self, env) -> None:
        plt.plot([x[0] for x in self.traj_x])
        plt.plot([x[0] for x in env.traj_x])
        return None

        
        