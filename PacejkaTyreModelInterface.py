from copy import deepcopy

import numpy as np

class PacejkaTyreModelInterface:

    def __init__(self, param):
        self.tyrF = param.tyrF
        self.tyrR = param.tyrR
        self.gamma = 0
#         self.vx = 15

    def get(self, mu, sigma, kappa, axisss, Fz_test, LeftRight):
#         Fx = 0  # np.zeros((len(sigma), len(kappa)))
#         Fy = 0  # np.zeros((len(sigma), len(kappa)))

        # for i in range(len(sigma)):
        #     for j in range(len(kappa)):
        #         Fx[[i, j]], Fy[[i, j]], _, _, _ = self.MF51Pacejka(
        #             Fz_test, kappa[j], sigma[i], 0, LeftRight, 15, Tyre, mu)
        # TODO: Распространить на вектора
        
        Tyre = self.tyrF if axisss == 1 else self.tyrR
        return self.MF51Pacejka(Fz_test, kappa, sigma, LeftRight, deepcopy(Tyre), mu)

    def MF51Pacejka(self, Fz, kappa, alpha, flagx, TyrPrms, mu):
        # Magic Formula 5.2 from Pacejka 2002

        # Left - Right
        if flagx == 0:
            TyrPrms.RHX1 = -TyrPrms.RHX1
            TyrPrms.QSX1 = -TyrPrms.QSX1
            TyrPrms.PEY3 = -TyrPrms.PEY3
            TyrPrms.PHY1 = -TyrPrms.PHY1
            TyrPrms.PHY2 = -TyrPrms.PHY2
            TyrPrms.PVY1 = -TyrPrms.PVY1        # flagx: 1 - left
            TyrPrms.PVY2 = -TyrPrms.PVY2        #        0 - right
            TyrPrms.RBY3 = -TyrPrms.RBY3
            TyrPrms.RVY1 = -TyrPrms.RVY1
            TyrPrms.RVY2 = -TyrPrms.RVY2
            TyrPrms.QBZ4 = -TyrPrms.QBZ4
            TyrPrms.QDZ6 = -TyrPrms.QDZ6
            TyrPrms.QDZ7 = -TyrPrms.QDZ7
            TyrPrms.QEZ4 = -TyrPrms.QEZ4
            TyrPrms.QHZ1 = -TyrPrms.QHZ1
            TyrPrms.QHZ2 = -TyrPrms.QHZ2
            TyrPrms.SSZ1 = -TyrPrms.SSZ1
            TyrPrms.QDZ3 = -TyrPrms.QDZ3

        # Friction
        TyrPrms.LMUX *= mu
        TyrPrms.LMUY *= mu

        # FORCES AND TORQUES
        gamma_s = np.sin(self.gamma)
        Fz = np.maximum(0.1, Fz)
        df_z = (Fz - TyrPrms.FNOMIN) / TyrPrms.FNOMIN

        # LONGITUDINAL FORCE
        # Pure slip
        gamma_x = self.gamma * TyrPrms.LGAX
        S_Hx = (TyrPrms.PHX1 + TyrPrms.PHX2 * df_z) * TyrPrms.LHX
        k_x = kappa + S_Hx
        C_x = TyrPrms.PCX1 * TyrPrms.LCX
        mu_x = (TyrPrms.PDX1 + TyrPrms.PDX2 * df_z) * (1 - TyrPrms.PDX3 * gamma_x**2) * TyrPrms.LMUX
        D_x = mu_x * Fz
        K_x = Fz * (TyrPrms.PKX1 + TyrPrms.PKX2 * df_z) * np.exp(TyrPrms.PKX3 * df_z) * TyrPrms.LKX
        E_x = (TyrPrms.PEX1 + TyrPrms.PEX2 * df_z + TyrPrms.PEX3 * df_z**2) * (
               1 - TyrPrms.PEX4 * np.sign(k_x)) * TyrPrms.LEX
        E_x = min([1, E_x])
        B_x = K_x / (C_x * D_x)
        S_Vx = Fz * (TyrPrms.PVX1 + TyrPrms.PVX2 * df_z) * TyrPrms.LVX * TyrPrms.LMUX
        F_x0 = D_x * np.sin(C_x * np.arctan(B_x * k_x - E_x * (B_x * k_x - np.arctan(B_x * k_x)))) + S_Vx

        # Combined slip
        S_Hxalpha = TyrPrms.RHX1
        alpha_s = alpha + S_Hxalpha
        B_xalpha = TyrPrms.RBX1 * np.cos(np.arctan(TyrPrms.RBX2 * kappa)) * TyrPrms.LXAL
        C_xalpha = TyrPrms.RCX1
        E_xalpha = TyrPrms.REX1 + TyrPrms.REX2 * df_z
        E_xalpha = min([1, E_xalpha])
        G_xalpha0 = np.cos(C_xalpha * np.arctan(B_xalpha * S_Hxalpha
                                              - E_xalpha * (B_xalpha * S_Hxalpha - np.arctan(B_xalpha * S_Hxalpha))))
        G_xalpha = np.cos(C_xalpha * np.arctan(B_xalpha * alpha_s
                                             - E_xalpha * (B_xalpha * alpha_s - np.arctan(B_xalpha * alpha_s)))) / G_xalpha0
        Fx = F_x0 * G_xalpha

        # LATERAL FORCE
        # Pure slip
        gamma_y = self.gamma * TyrPrms.LGAY
        S_Hy = (TyrPrms.PHY1 + TyrPrms.PHY2 * df_z) * TyrPrms.LHY + TyrPrms.PHY3 * gamma_y
        alpha_y = alpha + S_Hy
        C_y = TyrPrms.PCY1 * TyrPrms.LCY
        mu_y = (TyrPrms.PDY1 + TyrPrms.PDY2 * df_z) * (1 - TyrPrms.PDY3 * gamma_y**2) * TyrPrms.LMUY
        D_y = mu_y * Fz
        E_y = (TyrPrms.PEY1 + TyrPrms.PEY2 * df_z) * (
                    1 - (TyrPrms.PEY3 + TyrPrms.PEY4 * gamma_y) * np.sign(alpha_y)) * TyrPrms.LEY
        E_y = min([1, E_y])
        K_y0 = TyrPrms.PKY1 * TyrPrms.FNOMIN * np.sin(2 * np.arctan(
            Fz / (TyrPrms.PKY2 * TyrPrms.FNOMIN * TyrPrms.LFZO))) * TyrPrms.LFZO * TyrPrms.LKY
        K_y = K_y0 * (1 - TyrPrms.PKY3 * np.abs(gamma_y))
        B_y = K_y / (C_y * D_y)
        S_Vy = Fz * ((TyrPrms.PVY1 + TyrPrms.PVY2 * df_z) * TyrPrms.LVY + (
                    TyrPrms.PVY3 + TyrPrms.PVY4 * df_z) * gamma_y) * TyrPrms.LMUY
        F_y0 = D_y * np.sin(C_y * np.arctan(B_y * alpha_y - E_y * (B_y * alpha_y - np.arctan(B_y * alpha_y)))) + S_Vy

        # Combined slip
        S_Hyk = TyrPrms.RHY1 + TyrPrms.RHY2 * df_z
        k_s = kappa + S_Hyk
        B_yk = TyrPrms.RBY1 * np.cos(np.arctan(TyrPrms.RBY2 * (alpha - TyrPrms.RBY3))) * TyrPrms.LYKA
        C_yk = TyrPrms.RCY1
        E_yk = TyrPrms.REY1 + TyrPrms.REY2 * df_z
        E_yk = min([1, E_yk])
        D_Vyk = TyrPrms.LMUY * Fz * (TyrPrms.RVY1 + TyrPrms.RVY2 * df_z + TyrPrms.RVY3 * self.gamma) * np.cos(
            np.arctan(TyrPrms.RVY4 * alpha))
        S_Vyk = D_Vyk * np.sin(TyrPrms.RVY5 * np.arctan(TyrPrms.RVY6 * kappa)) * TyrPrms.LVYKA
        G_yk = np.cos(C_yk * np.arctan(B_yk * k_s - E_yk * (
                B_yk * k_s - np.arctan(B_yk * k_s)))) / np.cos(C_yk * np.arctan(B_yk * S_Hyk - E_yk * (
                B_yk * S_Hyk - np.arctan(B_yk * S_Hyk))))
        Fy = F_y0 * G_yk + S_Vyk
        Fy = -Fy

#         # ALIGNING TORQUE
#         # Pure slip
#         gamma_z = self.gamma * TyrPrms.LGAZ
#         S_Ht = TyrPrms.QHZ1 + TyrPrms.QHZ2 * df_z + (TyrPrms.QHZ3 + TyrPrms.QHZ4 * df_z) * gamma_z
#         alpha_t = alpha + S_Ht
#         B_t = (TyrPrms.QBZ1 + TyrPrms.QBZ2 * df_z + TyrPrms.QBZ3 * df_z**2) * (
#                     1 + TyrPrms.QBZ4 * gamma_z + TyrPrms.QBZ5 * np.abs(gamma_z)) * TyrPrms.LKY / TyrPrms.LMUY
#         C_t = TyrPrms.QCZ1
#         D_t = Fz * (TyrPrms.QDZ1 + TyrPrms.QDZ2 * df_z) * (
#                     1 + TyrPrms.QDZ3 * gamma_z + TyrPrms.QDZ4 * gamma_z**2) *\
#               TyrPrms.UNLOADED_RADIUS / TyrPrms.FNOMIN * TyrPrms.LTR
#         E_t = (TyrPrms.QEZ1 + TyrPrms.QEZ2 * df_z + TyrPrms.QEZ3 * df_z**2) * (
#                     1 + (TyrPrms.QEZ4 + TyrPrms.QEZ5 * gamma_z) * ((2 / np.pi) * np.arctan(B_t * C_t * alpha_t)))
#         E_t = min([1, E_t])
#         B_r = TyrPrms.QBZ9 * TyrPrms.LKY / TyrPrms.LMUY + TyrPrms.QBZ10 * B_y * C_y
#         D_r = Fz * ((TyrPrms.QDZ6 + TyrPrms.QDZ7 * df_z) * TyrPrms.LRES + (
#                     TyrPrms.QDZ8 + TyrPrms.QDZ9 * df_z) * gamma_z) * TyrPrms.UNLOADED_RADIUS * TyrPrms.LMUY
#         S_Hf = S_Hy + S_Vy / K_y
#         alpha_r = alpha + S_Hf

#         # Combined slip
#         alpha_t_eq = np.arctan(np.sqrt(np.tan(alpha_t)**2 + (K_x / K_y)**2 * kappa**2)) * np.sign(alpha_t)
#         alpha_r_eq = np.arctan(np.sqrt(np.tan(alpha_r)**2 + (K_x / K_y)**2 * kappa**2)) * np.sign(alpha_r)
#         t = D_t * np.cos(C_t * np.arctan(B_t * alpha_t_eq - E_t * (
#                 B_t * alpha_t_eq - np.arctan(B_t * alpha_t_eq)))) * np.cos(alpha)
#         F_yg = Fy - S_Vyk
#         M_zr = D_r * np.cos(np.arctan(B_r * alpha_r_eq)) * np.cos(alpha)
#         s = (TyrPrms.SSZ1 + TyrPrms.SSZ2 * (Fy / TyrPrms.FNOMIN) + (
#                 TyrPrms.SSZ3 + TyrPrms.SSZ4 * df_z) * self.gamma) * TyrPrms.LS
#         Mz = -t * F_yg + M_zr + s * Fx

#         # OVERTURNING TORQUE
#         # Pure and combined slip
#         Mx = TyrPrms.UNLOADED_RADIUS * Fz * (
#                     TyrPrms.QSX1 - TyrPrms.QSX2 * gamma_s + TyrPrms.QSX3 * Fy / TyrPrms.FNOMIN) * TyrPrms.LMX
#         # ROLLING RESISTANCE TORQUE
#         # Pure and combined slip
#         My = TyrPrms.UNLOADED_RADIUS * Fz * (TyrPrms.QSY1 + TyrPrms.QSY2 * Fx / TyrPrms.FNOMIN + TyrPrms.QSY3 * np.abs(
#             vx / TyrPrms.LONGVL) + TyrPrms.QSY3 * (vx / TyrPrms.LONGVL)**4)

        return Fx, Fy #, Mx, My, Mz

