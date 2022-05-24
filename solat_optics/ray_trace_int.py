"""
Functions to produce electric fields on the aperture planes from various sources.
Author: Grace E. Chesmore
December 2021
"""
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.interpolate import interp2d
from scipy.spatial import distance

import solat_optics
import solat_optics.ot_geo as ot_geo
import solat_optics.ray_trace as ray_trace
from solat_optics.ot_geo import *


def snell_vec(n_1, n_2, n_surf, s_1):
    """
    Vectorized snell's law, returing outgoing angle of ray through a medium.
    s_1 is the incoming vector, pointing from the light source to the surface.
    n_surf is the normal of the surface.
    """
    s_2 = (n_1 / n_2) * np.cross(n_surf, (np.cross(-n_surf, s_1))) - n_surf * np.sqrt(
        1 - (n_1 / n_2) ** 2 * np.dot((np.cross(n_surf, s_1)), (np.cross(n_surf, s_1)))
    )

    return s_2


class aperature_fields:
    def __init__(self, P_rx, tele_geo, plotOpts):

        if plotOpts.plot == 1:

            col = plotOpts.color
            ax = plotOpts.ax
            alph = plotOpts.alpha

        y_ap = tele_geo.y_ap
        th_fwhp = tele_geo.th_fwhp
        n_vac = tele_geo.n_vac
        n_si = tele_geo.n_si
        N_linear = tele_geo.n_scan

        # Step 1:  grid the plane of rays shooting out of receiver feed
        theta = np.linspace(-(np.pi / 2) - 0.25, -(np.pi / 2) + 0.25, N_linear)
        phi = np.linspace((np.pi / 2) - 0.25, (np.pi / 2) + 0.25, N_linear)

        theta, phi = np.meshgrid(theta, phi)
        theta = np.ravel(theta)
        phi = np.ravel(phi)

        # Step 2: calculate the position + local surface normal for the dish
        n_pts = len(theta)
        out = np.zeros((17, n_pts))

        # class instance variables
        self.tele_geo = tele_geo
        self.theta = theta
        self.phi = phi
        self.th_fwhp = th_fwhp  # Full width half power [rad] of source
        self.y_ap = y_ap
        self.n_pts = n_pts
        self.P_rx = P_rx
        self.out = out

    def trace_rays(self, ii):

        tele_geo = self.tele_geo
        n_vac = tele_geo.n_vac
        n_si = tele_geo.n_si
        th_fwhp = tele_geo.th_fwhp
        phi = self.phi
        theta = self.theta
        P_rx = self.P_rx
        y_ap = self.y_ap

        th = theta[ii]
        ph = phi[ii]

        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # Receiver feed position [mm] (in telescope reference frame):
        x_0 = P_rx[0]
        y_0 = P_rx[1]
        z_0 = P_rx[2]

        def root_z3a(t):

            x = x_0 + alpha * t
            y = y_0 + beta * t
            z = z_0 + gamma * t

            xm3a, ym3a, zm3a = tele_into_m3a(
                x, y, z
            )  # Convert ray's endpoint into M2 coordinates

            z_m3a = z3a(xm3a, ym3a)  # Z of mirror in M2 coordinates
            if np.isnan(z_m3a) == True:
                z_m3a = 0
            root = zm3a - z_m3a
            return root

        t_m3a = optimize.brentq(root_z3a, 2, 1600)

        # Location of where ray hits M2
        x_m3a = x_0 + alpha * t_m3a
        y_m3a = y_0 + beta * t_m3a
        z_m3a = z_0 + gamma * t_m3a
        P_m3a = np.array([x_m3a, y_m3a, z_m3a])

        if x_m3a ** 2 + x_m3a ** 2 >= (392 / 2) ** 2:
            pass
        ###### in M2 coordinates ##########################
        x_m3a_temp, y_m3a_temp, z_m3a_temp = tele_into_m3a(
            x_m3a, y_m3a, z_m3a
        )  # P_m2 temp
        x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m3a(x_0, y_0, z_0)  # P_rx temp
        norm = d_z3a(x_m3a_temp, y_m3a_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        vec_rx_m3a = np.array([x_m3a_temp, y_m3a_temp, z_m3a_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_m3a = np.sqrt(np.sum(vec_rx_m3a ** 2))
        tan_rx_m3a = vec_rx_m3a / dist_rx_m3a

        # Use Snell's Law to find angle of outgoing ray:

        tan_og_si = snell_vec(n_vac, n_si, N_hat, tan_rx_m3a)

        # Transform back to telescope cordinates ############

        N_hat_t = np.zeros(3)
        N_hat_t[0] = N_hat[0]
        N_hat_t[1] = N_hat[1] * np.cos(th1_l3) - N_hat[2] * np.sin(th1_l3)
        N_hat_t[2] = N_hat[1] * np.sin(th1_l3) + N_hat[2] * np.cos(th1_l3)

        tan_rx_m3a_t = np.zeros(3)
        tan_rx_m3a_t[0] = tan_rx_m3a[0]
        tan_rx_m3a_t[1] = tan_rx_m3a[1] * np.cos(th1_l3) - tan_rx_m3a[2] * np.sin(
            th1_l3
        )
        tan_rx_m3a_t[2] = tan_rx_m3a[1] * np.sin(th1_l3) + tan_rx_m3a[2] * np.cos(
            th1_l3
        )

        tan_og_t = np.zeros(3)
        tan_og_t[0] = tan_og_si[0]
        tan_og_t[1] = tan_og_si[1] * np.cos(th1_l3) - tan_og_si[2] * np.sin(th1_l3)
        tan_og_t[2] = tan_og_si[1] * np.sin(th1_l3) + tan_og_si[2] * np.cos(th1_l3)

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_z3b(t):
            x = P_m3a[0] + alpha * t
            y = P_m3a[1] + beta * t
            z = P_m3a[2] + gamma * t
            xm3b, ym3b, zm3b = tele_into_m3b(
                x, y, z
            )  # take ray end coordinates and convert to M1 coordinates
            z_m3b = z3b(xm3b, ym3b)  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_m3b) == True:
                z_m3b = 0
            root = zm3b - z_m3b
            return root

        t_m3b = optimize.brentq(root_z3b, 5, 200)

        # Location of where ray hits M1
        x_m3b = P_m3a[0] + alpha * t_m3b
        y_m3b = P_m3a[1] + beta * t_m3b
        z_m3b = P_m3a[2] + gamma * t_m3b
        P_m3b = np.array([x_m3b, y_m3b, z_m3b])

        ###### in M1 cordinates ##########################
        x_m3b_temp, y_m3b_temp, z_m3b_temp = tele_into_m3b(
            x_m3b, y_m3b, z_m3b
        )  # P_m1b temp
        x_m3a_temp, y_m3a_temp, z_m3a_temp = tele_into_m3b(
            P_m3a[0], P_m3a[1], P_m3a[2]
        )  # P_1a temp
        norm = d_z3b(x_m3b_temp, y_m3b_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_m3a_m3b = np.array([x_m3b_temp, y_m3b_temp, z_m3b_temp]) - np.array(
            [x_m3a_temp, y_m3a_temp, z_m3a_temp]
        )
        dist_m3a_m3b = np.sqrt(np.sum(vec_m3a_m3b ** 2))
        tan_m3a_m3b = vec_m3a_m3b / dist_m3a_m3b

        tan_og_vac = snell_vec(n_si, n_vac, N_hat, tan_m3a_m3b)

        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)
        tan_m3a_m3b_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        N_hat_t[0] = N_hat[0]
        N_hat_t[1] = N_hat[1] * np.cos(th2_l3) - N_hat[2] * np.sin(th2_l3)
        N_hat_t[2] = N_hat[1] * np.sin(th2_l3) + N_hat[2] * np.cos(th2_l3)

        tan_m3a_m3b_t[0] = tan_m3a_m3b[0]
        tan_m3a_m3b_t[1] = tan_m3a_m3b[1] * np.cos(th2_l3) - tan_m3a_m3b[2] * np.sin(
            th2_l3
        )
        tan_m3a_m3b_t[2] = tan_m3a_m3b[1] * np.sin(th2_l3) + tan_m3a_m3b[2] * np.cos(
            th2_l3
        )

        tan_og_t[0] = tan_og_vac[0]
        tan_og_t[1] = tan_og_vac[1] * np.cos(th2_l3) - tan_og_vac[2] * np.sin(th2_l3)
        tan_og_t[2] = tan_og_vac[1] * np.sin(th2_l3) + tan_og_vac[2] * np.cos(th2_l3)

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_z2a(t):
            x = P_m3b[0] + alpha * t
            y = P_m3b[1] + beta * t
            z = P_m3b[2] + gamma * t

            xm2a, ym2a, zm2a = tele_into_m2a(
                x, y, z
            )  # take ray end coordinates and convert to M1 coordinates
            z_m2a = z2a(xm2a, ym2a)  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_m2a) == True:
                z_m2a = 0
            root = zm2a - z_m2a
            return root

        t_m2a = optimize.brentq(root_z2a, 10, 2000)

        # Location of where ray hits M1
        x_m2a = P_m3b[0] + alpha * t_m2a
        y_m2a = P_m3b[1] + beta * t_m2a
        z_m2a = P_m3b[2] + gamma * t_m2a
        P_m2a = np.array([x_m2a, y_m2a, z_m2a])

        ###### in M1 cordinates ##########################
        x_m2a_temp, y_m2a_temp, z_m2a_temp = tele_into_m2a(
            x_m2a, y_m2a, z_m2a
        )  # P_m2a temp
        x_m3b_temp, y_m3b_temp, z_m3b_temp = tele_into_m2a(
            P_m3b[0], P_m3b[1], P_m3b[2]
        )  # P_1b temp
        norm = d_z2a(x_m2a_temp, y_m2a_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_m3b_m2a = np.array([x_m2a_temp, y_m2a_temp, z_m2a_temp]) - np.array(
            [x_m3b_temp, y_m3b_temp, z_m3b_temp]
        )
        dist_m3b_m2a = np.sqrt(np.sum(vec_m3b_m2a ** 2))
        tan_m3b_m2a = vec_m3b_m2a / dist_m3b_m2a

        # Outgoing ray
        tan_og_si = snell_vec(n_vac, n_si, N_hat, tan_m3b_m2a)

        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)
        tan_m3b_m2a_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        N_hat_t[0] = N_hat[0]
        N_hat_t[1] = N_hat[1] * np.cos(th2_l2) - N_hat[2] * np.sin(th2_l2)
        N_hat_t[2] = N_hat[1] * np.sin(th2_l2) + N_hat[2] * np.cos(th2_l2)

        tan_m3b_m2a_t[0] = tan_m3b_m2a[0]
        tan_m3b_m2a_t[1] = tan_m3b_m2a[1] * np.cos(th2_l2) - tan_m3b_m2a[2] * np.sin(
            th2_l2
        )
        tan_m3b_m2a_t[2] = tan_m3b_m2a[1] * np.sin(th2_l2) + tan_m3b_m2a[2] * np.cos(
            th2_l2
        )

        tan_og_t[0] = tan_og_si[0]
        tan_og_t[1] = tan_og_si[1] * np.cos(th2_l2) - tan_og_si[2] * np.sin(th2_l2)
        tan_og_t[2] = tan_og_si[1] * np.sin(th2_l2) + tan_og_si[2] * np.cos(th2_l2)

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_z2b(t):
            x = P_m2a[0] + alpha * t
            y = P_m2a[1] + beta * t
            z = P_m2a[2] + gamma * t
            xm2b, ym2b, zm2b = tele_into_m2b(
                x, y, z
            )  # take ray end coordinates and convert to M1 coordinates
            z_m2b = z2b()  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_m2b) == True:
                z_m2b = 0
            root = zm2b - z_m2b
            return root

        t_m2b = optimize.brentq(root_z2b, 5, 400)

        # Location of where ray hits M1
        x_m2b = P_m2a[0] + alpha * t_m2b
        y_m2b = P_m2a[1] + beta * t_m2b
        z_m2b = P_m2a[2] + gamma * t_m2b
        P_m2b = np.array([x_m2b, y_m2b, z_m2b])
        ###### in M1 cordinates ##########################
        x_m2b_temp, y_m2b_temp, z_m2b_temp = tele_into_m2b(
            x_m2b, y_m2b, z_m2b
        )  # P_m1b temp
        x_m2a_temp, y_m2a_temp, z_m2a_temp = tele_into_m2b(
            P_m2a[0], P_m2a[1], P_m2a[2]
        )  # P_1a temp
        norm = d_z2b()
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_m2a_m2b = np.array([x_m2b_temp, y_m2b_temp, z_m2b_temp]) - np.array(
            [x_m2a_temp, y_m2a_temp, z_m2a_temp]
        )
        dist_m2a_m2b = np.sqrt(np.sum(vec_m2a_m2b ** 2))
        tan_m2a_m2b = vec_m2a_m2b / dist_m2a_m2b

        # Outgoing ray
        tan_og_vac = snell_vec(n_si, n_vac, N_hat, tan_m2a_m2b)

        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)
        tan_m2a_m2b_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        N_hat_t[0] = N_hat[0]
        N_hat_t[1] = N_hat[1] * np.cos(th1_l2) - N_hat[2] * np.sin(th1_l2)
        N_hat_t[2] = N_hat[1] * np.sin(th1_l2) + N_hat[2] * np.cos(th1_l2)

        tan_m2a_m2b_t[0] = tan_m2a_m2b[0]
        tan_m2a_m2b_t[1] = tan_m2a_m2b[1] * np.cos(th1_l2) - tan_m2a_m2b[2] * np.sin(
            th1_l2
        )
        tan_m2a_m2b_t[2] = tan_m2a_m2b[1] * np.sin(th1_l2) + tan_m2a_m2b[2] * np.cos(
            th1_l2
        )

        tan_og_t[0] = tan_og_vac[0]
        tan_og_t[1] = tan_og_vac[1] * np.cos(th1_l2) - tan_og_vac[2] * np.sin(th1_l2)
        tan_og_t[2] = tan_og_vac[1] * np.sin(th1_l2) + tan_og_vac[2] * np.cos(th1_l2)

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_z1a(t):

            x = x_m2b + alpha * t
            y = y_m2b + beta * t
            z = z_m2b + gamma * t
            xm1a, ym1a, zm1a = tele_into_m1a(
                x, y, z
            )  # Convert ray's endpoint into M2 coordinates
            z_m1a = z1a(xm1a, ym1a)  # Z of mirror in M2 coordinates
            if np.isnan(z_m1a) == True:
                z_m1a = 0
            root = zm1a - z_m1a
            return root

        t_m1a = optimize.brentq(root_z1a, 5, 500)

        # Location of where ray hits M2
        x_m1a = x_m2b + alpha * t_m1a
        y_m1a = y_m2b + beta * t_m1a
        z_m1a = z_m2b + gamma * t_m1a
        P_m1a = np.array([x_m1a, y_m1a, z_m1a])

        ###### in M2 coordinates ##########################
        x_m1a_temp, y_m1a_temp, z_m1a_temp = tele_into_m1a(
            x_m1a, y_m1a, z_m1a
        )  # P_m2 temp
        x_m2b_temp, y_m2b_temp, z_m2b_temp = tele_into_m1a(
            x_m2b, y_m2b, z_m2b
        )  # P_rx temp
        norm = d_z1a(x_m1a_temp, y_m1a_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_m2b_m1a = np.array([x_m1a_temp, y_m1a_temp, z_m1a_temp]) - np.array(
            [x_m2b_temp, y_m2b_temp, z_m2b_temp]
        )
        dist_m2b_m1a = np.sqrt(np.sum(vec_m2b_m1a ** 2))
        tan_m2b_m1a = vec_m2b_m1a / dist_m2b_m1a

        # Outgoing ray
        tan_og_si = snell_vec(n_vac, n_si, N_hat, tan_m2b_m1a)

        # Outgoing
        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)
        tan_m2b_m1a_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        N_hat_t[0] = N_hat[0]
        N_hat_t[1] = N_hat[1] * np.cos(th2_l1) - N_hat[2] * np.sin(th2_l1)
        N_hat_t[2] = N_hat[1] * np.sin(th2_l1) + N_hat[2] * np.cos(th2_l1)

        tan_m2b_m1a_t[0] = tan_m2b_m1a[0]
        tan_m2b_m1a_t[1] = tan_m2b_m1a[1] * np.cos(th2_l1) - tan_m2b_m1a[2] * np.sin(
            th2_l1
        )
        tan_m2b_m1a_t[2] = tan_m2b_m1a[1] * np.sin(th2_l1) + tan_m2b_m1a[2] * np.cos(
            th2_l1
        )

        tan_og_t[0] = tan_og_si[0]
        tan_og_t[1] = tan_og_si[1] * np.cos(th2_l1) - tan_og_si[2] * np.sin(th2_l1)
        tan_og_t[2] = tan_og_si[1] * np.sin(th2_l1) + tan_og_si[2] * np.cos(th2_l1)
        ##################################################

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_z1b(t):
            x = P_m1a[0] + alpha * t
            y = P_m1a[1] + beta * t
            z = P_m1a[2] + gamma * t
            xm1b, ym1b, zm1b = tele_into_m1b(
                x, y, z
            )  # take ray end coordinates and convert to M1 coordinates
            z_m1b = z1b()  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_m1b) == True:
                z_m1b = 0
            root = zm1b - z_m1b
            return root

        t_m1b = optimize.brentq(root_z1b, 5, 200)

        # Location of where ray hits M1
        x_m1b = P_m1a[0] + alpha * t_m1b
        y_m1b = P_m1a[1] + beta * t_m1b
        z_m1b = P_m1a[2] + gamma * t_m1b
        P_m1b = np.array([x_m1b, y_m1b, z_m1b])

        ###### in M1 cordinates ##########################
        x_m1b_temp, y_m1b_temp, z_m1b_temp = tele_into_m1b(
            x_m1b, y_m1b, z_m1b
        )  # P_m1b temp
        x_m1a_temp, y_m1a_temp, z_m1a_temp = tele_into_m1b(
            P_m1a[0], P_m1a[1], P_m1a[2]
        )  # P_1a temp
        norm = d_z1b()
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_m1a_m1b = np.array([x_m1b_temp, y_m1b_temp, z_m1b_temp]) - np.array(
            [x_m1a_temp, y_m1a_temp, z_m1a_temp]
        )
        dist_m1a_m1b = np.sqrt(np.dot(vec_m1a_m1b, vec_m1a_m1b))
        tan_m1a_m1b = vec_m1a_m1b / dist_m1a_m1b

        # Outgoing ray
        tan_og_vac = snell_vec(n_si, n_vac, N_hat, -tan_m1a_m1b)

        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)
        tan_m1a_m1b_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        N_hat_t[0] = N_hat[0]
        N_hat_t[1] = N_hat[1] * np.cos(th1_l1) - N_hat[2] * np.sin(th1_l1)
        N_hat_t[2] = N_hat[1] * np.sin(th1_l1) + N_hat[2] * np.cos(th1_l1)

        tan_m1a_m1b_t[0] = tan_m1a_m1b[0]
        tan_m1a_m1b_t[1] = tan_m1a_m1b[1] * np.cos(th1_l1) - tan_m1a_m1b[2] * np.sin(
            th1_l1
        )
        tan_m1a_m1b_t[2] = tan_m1a_m1b[1] * np.sin(th1_l1) + tan_m1a_m1b[2] * np.cos(
            th1_l1
        )

        tan_og_t[0] = tan_og_vac[0]
        tan_og_t[1] = tan_og_vac[1] * np.cos(th1_l1) - tan_og_vac[2] * np.sin(th1_l1)
        tan_og_t[2] = tan_og_vac[1] * np.sin(th1_l1) + tan_og_vac[2] * np.cos(th1_l1)

        ################################################
        dist_m1b_ap = abs((y_ap - P_m1b[1]) / tan_og_t[1])
        total_path_length = (
            dist_rx_m3a
            + dist_m3a_m3b * (n_si)
            + dist_m3b_m2a
            + dist_m2a_m2b * (n_si)
            + dist_m2b_m1a
            + dist_m1a_m1b * (n_si)
            + dist_m1b_ap
        )

        pos_ap = P_m1b - dist_m1b_ap * tan_og_t

        # Estimate theta
        de_ve = np.arctan(tan_rx_m3a_t[0] / (-tan_rx_m3a_t[1]))
        de_ho = np.arctan(
            tan_rx_m3a_t[2] / np.sqrt(tan_rx_m3a_t[0] ** 2 + tan_rx_m3a_t[1] ** 2)
        )

        #         ################################################
        #         if plotOpts.plot == 1:
        #             if np.mod(ii, 21) == 0:
        #                 if ii == 21:
        #                     ot_geo.plot_lenses()

        #                 ax.plot([y_0, y_m3a], [z_0, z_m3a], "-", color=col, alpha=alph)

        #                 ax.plot([y_m3a, y_m3b], [z_m3a, z_m3b], "-", color=col, alpha=alph)
        #                 ax.plot([y_m3b, y_m2a], [z_m3b, z_m2a], "-", color=col, alpha=alph)

        #                 ax.plot([y_m2a, y_m2b], [z_m2a, z_m2b], "-", color=col, alpha=alph)
        #                 ax.plot([y_m2b, y_m1a], [z_m2b, z_m1a], "-", color=col, alpha=alph)

        #                 ax.plot([y_m1a, y_m1b], [z_m1a, z_m1b], "-", color=col, alpha=alph)
        #                 ax.plot(
        #                     [y_m1b, pos_ap[1]], [z_m1b, pos_ap[2]], "-", color=col, alpha=alph
        #                 )

        amp = np.exp(
            (-0.5)
            * (de_ho ** 2 + de_ve ** 2)
            / (th_fwhp / (np.sqrt(8 * np.log(2)))) ** 2
        )
        return [
            pos_ap[0],
            pos_ap[1],
            pos_ap[2],
            x_m1b,
            y_m1b,
            z_m1b,
            x_m1a,
            y_m1a,
            z_m1a,
            x_m2b,
            y_m2b,
            z_m2b,
            x_m2a,
            y_m2a,
            z_m2a,
            x_m3b,
            y_m3b,
            z_m3b,
            x_m3a,
            y_m3a,
            z_m3a,  # 21
            x_0,
            y_0,
            z_0,
            N_hat[0],
            N_hat[1],
            N_hat[2],
            tan_og_t[0],
            tan_og_t[1],
            tan_og_t[2],  # 30
            total_path_length,
            amp,  # to determine phase, amp
        ]

    #         return [0]*len(self.out)

    def output(self):

        result = []
        for ii in range(self.n_pts):
            result.append(self.trace_rays(ii))
        self.out = np.array(result).transpose()

        return self.out


class rx_to_lyot_model:
    """
    Produces electric fields on aperture plane ray-traced from the receiver feed.
    """

    def __init__(self, p_rx, tele_geo, plotOpts):

        y_ap = tele_geo.y_ap
        th_fwhp = tele_geo.th_fwhp
        n_vac = tele_geo.n_vac
        n_si = tele_geo.n_si
        n_linear = tele_geo.n_scan

        # Step 1:  grid the plane of rays shooting out of receiver feed
        theta = np.linspace(-(np.pi / 2) - 0.28, -(np.pi / 2) + 0.28, n_linear)
        phi = np.linspace((np.pi / 2) - 0.28, (np.pi / 2) + 0.28, n_linear)

        theta, phi = np.meshgrid(theta, phi)
        theta = np.ravel(theta)
        phi = np.ravel(phi)

        # Step 2: calculate the position + local surface normal for the dish
        n_pts = len(theta)
        out = np.zeros((17, n_pts))

        # class instance variables
        self.tele_geo = tele_geo
        self.theta = theta
        self.phi = phi
        self.th_fwhp = th_fwhp  # Full width half power [rad] of source
        self.y_ap = y_ap
        self.n_pts = n_pts
        self.p_rx = p_rx
        self.out = out

    def trace_rays(self, ii):

        tele_geo = self.tele_geo
        n_vac = tele_geo.n_vac
        n_si = tele_geo.n_si
        th_fwhp = tele_geo.th_fwhp
        phi = self.phi
        theta = self.theta
        p_rx = self.p_rx
        y_ap = self.y_ap

        th = theta[ii]
        ph = phi[ii]

        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # Receiver feed position [mm] (in telescope reference frame):
        x_0 = p_rx[0]
        y_0 = p_rx[1]
        z_0 = p_rx[2]

        def root_z3a(t):

            x = x_0 + alpha * t
            y = y_0 + beta * t
            z = z_0 + gamma * t
            xm3a, ym3a, zm3a = tele_into_m3a(
                x, y, z
            )  # Convert ray's endpoint into M2 coordinates
            z_m3a = z3a(xm3a, ym3a)  # Z of mirror in M2 coordinates
            if np.isnan(z_m3a) == True:
                z_m3a = 0
            root = zm3a - z_m3a
            return root

        t_m3a = optimize.brentq(root_z3a, 40, 700)

        # Location of where ray hits M2
        x_m3a = x_0 + alpha * t_m3a
        y_m3a = y_0 + beta * t_m3a
        z_m3a = z_0 + gamma * t_m3a
        p_m3a = np.array([x_m3a, y_m3a, z_m3a])

        if np.sqrt(x_m3a ** 2 + z_m3a ** 2) >= ((374 - 8) / 2):
            pass

        ###### in M2 coordinates ##########################
        x_m3a_temp, y_m3a_temp, z_m3a_temp = tele_into_m3a(
            x_m3a, y_m3a, z_m3a
        )  # P_m2 temp
        x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m3a(x_0, y_0, z_0)  # p_rx temp
        norm = d_z3a(x_m3a_temp, y_m3a_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        n_hat = norm_temp / np.sqrt(np.sum(norm_temp ** 2))
        vec_rx_m3a = np.array([x_m3a_temp, y_m3a_temp, z_m3a_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_m3a = np.sqrt(np.sum(vec_rx_m3a ** 2))
        tan_rx_m3a = vec_rx_m3a / dist_rx_m3a

        # Use Snell's Law to find angle of outgoing ray:
        tan_og_si = snell_vec(n_vac, n_si, n_hat, tan_rx_m3a)

        # Transform back to telescope cordinates ############

        n_hat_t = np.zeros(3)
        n_hat_t[0] = n_hat[0]
        n_hat_t[1] = n_hat[1] * np.cos(th1_l3) - n_hat[2] * np.sin(th1_l3)
        n_hat_t[2] = n_hat[1] * np.sin(th1_l3) + n_hat[2] * np.cos(th1_l3)

        tan_rx_m3a_t = np.zeros(3)
        tan_rx_m3a_t[0] = tan_rx_m3a[0]
        tan_rx_m3a_t[1] = tan_rx_m3a[1] * np.cos(th1_l3) - tan_rx_m3a[2] * np.sin(
            th1_l3
        )
        tan_rx_m3a_t[2] = tan_rx_m3a[1] * np.sin(th1_l3) + tan_rx_m3a[2] * np.cos(
            th1_l3
        )

        tan_og_t = np.zeros(3)
        tan_og_t[0] = tan_og_si[0]
        tan_og_t[1] = tan_og_si[1] * np.cos(th1_l3) - tan_og_si[2] * np.sin(th1_l3)
        tan_og_t[2] = tan_og_si[1] * np.sin(th1_l3) + tan_og_si[2] * np.cos(th1_l3)

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_z3b(t):
            x = p_m3a[0] + alpha * t
            y = p_m3a[1] + beta * t
            z = p_m3a[2] + gamma * t
            xm3b, ym3b, zm3b = tele_into_m3b(
                x, y, z
            )  # take ray end coordinates and convert to M1 coordinates
            z_m3b = z3b(xm3b, ym3b)  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_m3b) == True:
                z_m3b = 0
            root = zm3b - z_m3b
            return root

        t_m3b = optimize.brentq(root_z3b, 5, 200)

        # Location of where ray hits M1
        x_m3b = p_m3a[0] + alpha * t_m3b
        y_m3b = p_m3a[1] + beta * t_m3b
        z_m3b = p_m3a[2] + gamma * t_m3b
        p_m3b = np.array([x_m3b, y_m3b, z_m3b])

        ###### in M1 cordinates ##########################
        x_m3b_temp, y_m3b_temp, z_m3b_temp = tele_into_m3b(
            x_m3b, y_m3b, z_m3b
        )  # p_m1b temp
        x_m3a_temp, y_m3a_temp, z_m3a_temp = tele_into_m3b(
            p_m3a[0], p_m3a[1], p_m3a[2]
        )  # P_1a temp
        norm = d_z3b(x_m3b_temp, y_m3b_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        n_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_m3a_m3b = np.array([x_m3b_temp, y_m3b_temp, z_m3b_temp]) - np.array(
            [x_m3a_temp, y_m3a_temp, z_m3a_temp]
        )
        dist_m3a_m3b = np.sqrt(np.sum(vec_m3a_m3b ** 2))
        tan_m3a_m3b = vec_m3a_m3b / dist_m3a_m3b

        tan_og_vac = snell_vec(n_si, n_vac, n_hat, tan_m3a_m3b)

        # Transform back to telescope cordinates
        n_hat_t = np.zeros(3)
        tan_m3a_m3b_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        n_hat_t[0] = n_hat[0]
        n_hat_t[1] = n_hat[1] * np.cos(th2_l3) - n_hat[2] * np.sin(th2_l3)
        n_hat_t[2] = n_hat[1] * np.sin(th2_l3) + n_hat[2] * np.cos(th2_l3)

        tan_m3a_m3b_t[0] = tan_m3a_m3b[0]
        tan_m3a_m3b_t[1] = tan_m3a_m3b[1] * np.cos(th2_l3) - tan_m3a_m3b[2] * np.sin(
            th2_l3
        )
        tan_m3a_m3b_t[2] = tan_m3a_m3b[1] * np.sin(th2_l3) + tan_m3a_m3b[2] * np.cos(
            th2_l3
        )

        tan_og_t[0] = tan_og_vac[0]
        tan_og_t[1] = tan_og_vac[1] * np.cos(th2_l3) - tan_og_vac[2] * np.sin(th2_l3)
        tan_og_t[2] = tan_og_vac[1] * np.sin(th2_l3) + tan_og_vac[2] * np.cos(th2_l3)

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_lyot(t):
            x = p_m3b[0] + alpha * t
            y = p_m3b[1] + beta * t
            z = p_m3b[2] + gamma * t

            xlyot, ylyot, zlyot = tele_into_lyot(
                x, y, z
            )  # take ray end coordinates and convert to M1 coordinates
            z_ly = z_lyot()  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_ly) == True:
                z_ly = 0
            root = zlyot - z_ly
            return root

        t_lyot = optimize.brentq(root_lyot, 1, 2000)

        # Location of where ray hits M1
        x_ly = p_m3b[0] + alpha * t_lyot
        y_ly = p_m3b[1] + beta * t_lyot
        z_ly = p_m3b[2] + gamma * t_lyot

        ###### in M1 cordinates ##########################
        x_lyot_temp, y_lyot_temp, z_lyot_temp = tele_into_lyot(
            x_ly, y_ly, z_ly
        )  # p_m1b temp
        x_m3b_temp, y_m3b_temp, z_m3b_temp = tele_into_lyot(
            x_m3b, y_m3b, z_m3b
        )  # P_1a temp
        norm = d_zlyot()
        norm_temp = np.array([-norm[0], -norm[1], 1])
        n_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_m3b_lyot = np.array([x_m3b_temp, y_m3b_temp, z_m3b_temp]) - np.array(
            [x_lyot_temp, y_lyot_temp, z_lyot_temp]
        )
        dist_m3b_lyot = np.sqrt(np.sum(vec_m3b_lyot ** 2))
        tan_m3b_lyot = vec_m3b_lyot / dist_m3b_lyot

        # Transform back to telescope cordinates
        n_hat_t = np.zeros(3)
        tan_m3b_lyot_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        n_hat_t[0] = n_hat[0]
        n_hat_t[1] = n_hat[1] * np.cos(th2_l3) - n_hat[2] * np.sin(th2_l3)
        n_hat_t[2] = n_hat[1] * np.sin(th2_l3) + n_hat[2] * np.cos(th2_l3)

        tan_m3b_lyot_t[0] = tan_m3b_lyot[0]
        tan_m3b_lyot_t[1] = tan_m3b_lyot[1] * np.cos(th2_l3) - tan_m3b_lyot[2] * np.sin(
            th2_l3
        )
        tan_m3b_lyot_t[2] = tan_m3b_lyot[1] * np.sin(th2_l3) + tan_m3b_lyot[2] * np.cos(
            th2_l3
        )

        tan_og_t[0] = tan_og_vac[0]
        tan_og_t[1] = tan_og_vac[1] * np.cos(th2_l3) - tan_og_vac[2] * np.sin(th2_l3)
        tan_og_t[2] = tan_og_vac[1] * np.sin(th2_l3) + tan_og_vac[2] * np.cos(th2_l3)

        #         #################################################
        #         if plotOpts.plot == 1:
        #             if np.mod(ii, 21) == 0:
        #                 if ii == 21:
        #                     ot_geo.plot_lenses()

        #                 ax.plot(
        #                     [y_0, y_m3a], [z_0, z_m3a], "-", color=plotOpts.color, alpha=alph
        #                 )
        #                 ax.plot(
        #                     [y_m3a, y_m3b],
        #                     [z_m3a, z_m3b],
        #                     "-",
        #                     color=plotOpts.color,
        #                     alpha=alph,
        #                 )
        #                 ax.plot(
        #                     [y_m3b, y_ly], [z_m3b, z_ly], "-", color=plotOpts.color, alpha=alph
        #                 )

        total_path_length = (dist_m3a_m3b * n_si) + (dist_rx_m3a) + dist_m3b_lyot
        amp = np.exp(
            (-0.5)
            * ((th - np.mean(theta)) ** 2 + (ph - np.mean(phi)) ** 2)
            / (th_fwhp / (np.sqrt(8 * np.log(2)))) ** 2
        )

        return [
            x_ly,
            y_ly,
            z_ly,
            total_path_length,
            amp,  # to determine phase, amp
            n_hat_t[0],
            n_hat_t[1],
            n_hat_t[2],
            tan_og_t[0],
            tan_og_t[1],
            tan_og_t[2],  # 23
            x_m3b,
            y_m3b,
            z_m3b,  # 13
            x_m3a,
            y_m3a,
            z_m3a,
            x_0,
            y_0,
            z_0,
        ]

    def output(self):

        result = []
        for ii in range(self.n_pts):
            result.append(self.trace_rays(ii))
        self.out = np.array(result).transpose()

        return self.out


class source_to_lyot_model:
    """
    Produces electric fields on aperture plane ray-traced from the source-feed.
    """

    def __init__(self, p_source, tele_geo, plotOpts):

        if plotOpts.plot == 1:

            col = plotOpts.color
            ax = plotOpts.ax
            alph = plotOpts.alpha

        y_ap = tele_geo.y_ap
        th_fwhp = tele_geo.th_fwhp
        n_vac = tele_geo.n_vac
        n_si = tele_geo.n_si
        n_linear = tele_geo.n_scan

        # Step 1:  grid the plane of rays shooting out of receiver feed
        theta = np.linspace((np.pi / 2) - 0.22, (np.pi / 2) + 0.22, n_linear)
        phi = np.linspace((np.pi / 2) - 0.22, (np.pi / 2) + 0.22, n_linear)

        theta, phi = np.meshgrid(theta, phi)
        theta = np.ravel(theta)
        phi = np.ravel(phi)

        # Step 2: calculate the position + local surface normal for the dish
        n_pts = len(theta)
        out = np.zeros((17, n_pts))

        # class instance variables
        self.tele_geo = tele_geo
        self.theta = theta
        self.phi = phi
        self.th_fwhp = th_fwhp  # Full width half power [rad] of source
        self.y_ap = y_ap
        self.n_pts = n_pts
        self.p_source = p_source
        self.out = out

    def trace_rays(self, ii):

        tele_geo = self.tele_geo
        n_vac = tele_geo.n_vac
        n_si = tele_geo.n_si
        th_fwhp = tele_geo.th_fwhp
        phi = self.phi
        theta = self.theta
        p_source = self.p_source
        y_ap = self.y_ap

        th = theta[ii]
        ph = phi[ii]

        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # Receiver feed position [mm] (in telescope reference frame):
        x_0 = p_source[0]
        y_0 = p_source[1]
        z_0 = p_source[2]

        def root_z1b(t):
            x = p_source[0] + alpha * t
            y = p_source[1] + beta * t
            z = p_source[2] + gamma * t
            xm1b, ym1b, zm1b = tele_into_m1b(
                x, y, z
            )  # take ray end coordinates and convert to M1 coordinates
            z_m1b = z1b()  # Z of mirror 1 in M1 coordinates
            if np.isnan(z_m1b) == True:
                z_m1b = 0
            root = zm1b - z_m1b
            return root

        t_m1b = optimize.brentq(root_z1b, 5, 14000)

        # Location of where ray hits M1
        x_m1b = p_source[0] + alpha * t_m1b
        y_m1b = p_source[1] + beta * t_m1b
        z_m1b = p_source[2] + gamma * t_m1b
        p_m1b = np.array([x_m1b, y_m1b, z_m1b])

        ###### in M2 coordinates ##########################
        x_m1b_temp, y_m1b_temp, z_m1b_temp = tele_into_m1b(
            x_m1b, y_m1b, z_m1b
        )  # P_m2 temp
        x_source_temp, y_source_temp, z_source_temp = tele_into_m1b(
            x_0, y_0, z_0
        )  # p_rx temp
        norm = d_z1b()
        norm_temp = np.array([-norm[0], -norm[1], 1])
        n_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_source_m1b = np.array([x_m1b_temp, y_m1b_temp, z_m1b_temp]) - np.array(
            [x_source_temp, y_source_temp, z_source_temp]
        )
        dist_source_m1b = np.sqrt(np.sum(vec_source_m1b ** 2))
        tan_source_m1b = vec_source_m1b / dist_source_m1b

        # Use Snell's Law to find angle of outgoing ray:
        tan_og = snell_vec(n_vac, n_si, n_hat, tan_source_m1b)

        # Transform back to telescope cordinates
        n_hat_t = np.zeros(3)
        tan_source_m1b_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        n_hat_t[0] = n_hat[0]
        n_hat_t[1] = n_hat[1] * np.cos(th1_l1) - n_hat[2] * np.sin(th1_l1)
        n_hat_t[2] = n_hat[1] * np.sin(th1_l1) + n_hat[2] * np.cos(th1_l1)

        tan_source_m1b_t[0] = tan_source_m1b[0]
        tan_source_m1b_t[1] = tan_source_m1b[1] * np.cos(th1_l1) - tan_source_m1b[
            2
        ] * np.sin(th1_l1)
        tan_source_m1b_t[2] = tan_source_m1b[1] * np.sin(th1_l1) + tan_source_m1b[
            2
        ] * np.cos(th1_l1)

        tan_og_t[0] = tan_og[0]
        tan_og_t[1] = tan_og[1] * np.cos(th1_l1) - tan_og[2] * np.sin(th1_l1)
        tan_og_t[2] = tan_og[1] * np.sin(th1_l1) + tan_og[2] * np.cos(th1_l1)

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        def root_z1a(t):

            x = x_m1b + alpha * t
            y = y_m1b + beta * t
            z = z_m1b + gamma * t
            xm1a, ym1a, zm1a = tele_into_m1a(
                x, y, z
            )  # Convert ray's endpoint into M2 coordinates
            z_m1a = z1a(xm1a, ym1a)  # Z of mirror in M2 coordinates
            if np.isnan(z_m1a) == True:
                z_m1a = 0
            root = zm1a - z_m1a
            return root

        t_m1a = optimize.brentq(root_z1a, 1, 900)

        # Location of where ray hits M2
        x_m1a = x_m1b + alpha * t_m1a
        y_m1a = y_m1b + beta * t_m1a
        z_m1a = z_m1b + gamma * t_m1a
        p_m1a = np.array([x_m1a, y_m1a, z_m1a])

        ###### in M2 coordinates ##########################
        x_m1a_temp, y_m1a_temp, z_m1a_temp = tele_into_m1a(
            x_m1a, y_m1a, z_m1a
        )  # P_m2 temp
        x_m1b_temp, y_m1b_temp, z_m1b_temp = tele_into_m1a(
            x_m1b, y_m1b, z_m1b
        )  # p_rx temp
        norm = d_z1a(x_m1a_temp, y_m1a_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        n_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
        vec_m1b_m1a = np.array([x_m1a_temp, y_m1a_temp, z_m1a_temp]) - np.array(
            [x_m1b_temp, y_m1b_temp, z_m1b_temp]
        )
        dist_m1b_m1a = np.sqrt(np.sum(vec_m1b_m1a ** 2))
        tan_m1b_m1a = vec_m1b_m1a / dist_m1b_m1a

        # Outgoing ray
        tan_og = snell_vec(n_si, n_vac, -n_hat, tan_m1b_m1a)

        # Outgoing
        # Transform back to telescope cordinates
        n_hat_t = np.zeros(3)
        tan_m1b_m1a_t = np.zeros(3)
        tan_og_t = np.zeros(3)

        n_hat_t[0] = n_hat[0]
        n_hat_t[1] = n_hat[1] * np.cos(th2_l1) - n_hat[2] * np.sin(th2_l1)
        n_hat_t[2] = n_hat[1] * np.sin(th2_l1) + n_hat[2] * np.cos(th2_l1)

        tan_m1b_m1a_t[0] = tan_m1b_m1a[0]
        tan_m1b_m1a_t[1] = tan_m1b_m1a[1] * np.cos(th2_l1) - tan_m1b_m1a[2] * np.sin(
            th2_l1
        )
        tan_m1b_m1a_t[2] = tan_m1b_m1a[1] * np.sin(th2_l1) + tan_m1b_m1a[2] * np.cos(
            th2_l1
        )

        tan_og_t[0] = tan_og[0]
        tan_og_t[1] = tan_og[1] * np.cos(th2_l1) - tan_og[2] * np.sin(th2_l1)
        tan_og_t[2] = tan_og[1] * np.sin(th2_l1) + tan_og[2] * np.cos(th2_l1)

        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        if np.sqrt(x_m1a ** 2 + z_m1a ** 2) >= (392 / 2):
            return [
                0,
                0,
                0,
                0,
                0,  # to determine phase, amp
                0,
                0,
                0,
                0,
                0,
                0,  # 23
                0,
                0,
                0,  # 13
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        else:

            def root_z2b(t):
                x = p_m1a[0] + alpha * t
                y = p_m1a[1] + beta * t
                z = p_m1a[2] + gamma * t
                xm2b, ym2b, zm2b = tele_into_m2b(
                    x, y, z
                )  # take ray end coordinates and convert to M1 coordinates
                z_m2b = z2b()  # Z of mirror 1 in M1 coordinates
                if np.isnan(z_m2b) == True:
                    z_m2b = 0
                root = zm2b - z_m2b
                return root

            t_m2b = optimize.brentq(root_z2b, 5, 600)
            # Location of where ray hits M1
            x_m2b = p_m1a[0] + alpha * t_m2b
            y_m2b = p_m1a[1] + beta * t_m2b
            z_m2b = p_m1a[2] + gamma * t_m2b
            p_m2b = np.array([x_m2b, y_m2b, z_m2b])

            ###### in M1 cordinates ##########################
            x_m2b_temp, y_m2b_temp, z_m2b_temp = tele_into_m2b(
                x_m2b, y_m2b, z_m2b
            )  # p_m1b temp
            x_m1a_temp, y_m1a_temp, z_m1a_temp = tele_into_m2b(
                p_m1a[0], p_m1a[1], p_m1a[2]
            )  # P_1a temp
            norm = d_z2b()
            norm_temp = np.array([-norm[0], -norm[1], 1])
            n_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
            vec_m1a_m2b = np.array([x_m2b_temp, y_m2b_temp, z_m2b_temp]) - np.array(
                [x_m1a_temp, y_m1a_temp, z_m1a_temp]
            )
            dist_m1a_m2b = np.sqrt(np.sum(vec_m1a_m2b ** 2))
            tan_m1a_m2b = vec_m1a_m2b / dist_m1a_m2b

            # Outgoing ray
            tan_og = snell_vec(n_vac, n_si, -n_hat, tan_m1a_m2b)

            # Transform back to telescope cordinates
            n_hat_t = np.zeros(3)
            tan_m1a_m2b_t = np.zeros(3)
            tan_og_t = np.zeros(3)

            n_hat_t[0] = n_hat[0]
            n_hat_t[1] = n_hat[1] * np.cos(th1_l2) - n_hat[2] * np.sin(th1_l2)
            n_hat_t[2] = n_hat[1] * np.sin(th1_l2) + n_hat[2] * np.cos(th1_l2)

            tan_m1a_m2b_t[0] = tan_m1a_m2b[0]
            tan_m1a_m2b_t[1] = tan_m1a_m2b[1] * np.cos(th1_l2) - tan_m1a_m2b[
                2
            ] * np.sin(th1_l2)
            tan_m1a_m2b_t[2] = tan_m1a_m2b[1] * np.sin(th1_l2) + tan_m1a_m2b[
                2
            ] * np.cos(th1_l2)

            tan_og_t[0] = tan_og[0]
            tan_og_t[1] = tan_og[1] * np.cos(th1_l2) - tan_og[2] * np.sin(th1_l2)
            tan_og_t[2] = tan_og[1] * np.sin(th1_l2) + tan_og[2] * np.cos(th1_l2)

            if np.sqrt(x_m2b ** 2 + z_m2b ** 2) >= (352 / 2):
                return [
                    0,
                    0,
                    0,
                    0,
                    0,  # to determine phase, amp
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,  # 23
                    0,
                    0,
                    0,  # 13
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]

            else:

                alpha = tan_og_t[0]
                beta = tan_og_t[1]
                gamma = tan_og_t[2]

                def root_z2a(t):
                    x = p_m2b[0] + alpha * t
                    y = p_m2b[1] + beta * t
                    z = p_m2b[2] + gamma * t

                    xm2a, ym2a, zm2a = tele_into_m2a(
                        x, y, z
                    )  # take ray end coordinates and convert to M1 coordinates
                    z_m2a = z2a(xm2a, ym2a)  # Z of mirror 1 in M1 coordinates
                    if np.isnan(z_m2a) == True:
                        z_m2a = 0
                    root = zm2a - z_m2a
                    # print(xm2a,ym2a,zm2a,z_m2a,root)
                    return root

                t_m2a = optimize.brentq(root_z2a, 0, 100)

                # Location of where ray hits M1
                x_m2a = p_m2b[0] + alpha * t_m2a
                y_m2a = p_m2b[1] + beta * t_m2a
                z_m2a = p_m2b[2] + gamma * t_m2a
                p_m2a = np.array([x_m2a, y_m2a, z_m2a])

                ###### in M1 cordinates ##########################
                x_m2a_temp, y_m2a_temp, z_m2a_temp = tele_into_m2a(
                    x_m2a, y_m2a, z_m2a
                )  # p_m2a temp
                x_m2b_temp, y_m2b_temp, z_m2b_temp = tele_into_m2a(
                    p_m2b[0], p_m2b[1], p_m2b[2]
                )  # P_1b temp
                norm = d_z2a(x_m2a_temp, y_m2a_temp)
                norm_temp = np.array([-norm[0], -norm[1], 1])
                n_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
                vec_m2b_m2a = np.array([x_m2a_temp, y_m2a_temp, z_m2a_temp]) - np.array(
                    [x_m2b_temp, y_m2b_temp, z_m2b_temp]
                )
                dist_m2b_m2a = np.sqrt(np.sum(vec_m2b_m2a ** 2))
                tan_m2b_m2a = vec_m2b_m2a / dist_m2b_m2a

                # Outgoing ray

                tan_og = snell_vec(n_si, n_vac, -n_hat, tan_m2b_m2a)

                # Transform back to telescope cordinates
                n_hat_t = np.zeros(3)
                tan_m2b_m2a_t = np.zeros(3)
                tan_og_t = np.zeros(3)

                n_hat_t[0] = n_hat[0]
                n_hat_t[1] = n_hat[1] * np.cos(th2_l2) - n_hat[2] * np.sin(th2_l2)
                n_hat_t[2] = n_hat[1] * np.sin(th2_l2) + n_hat[2] * np.cos(th2_l2)

                tan_m2b_m2a_t[0] = tan_m2b_m2a[0]
                tan_m2b_m2a_t[1] = tan_m2b_m2a[1] * np.cos(th2_l2) - tan_m2b_m2a[
                    2
                ] * np.sin(th2_l2)
                tan_m2b_m2a_t[2] = tan_m2b_m2a[1] * np.sin(th2_l2) + tan_m2b_m2a[
                    2
                ] * np.cos(th2_l2)

                tan_og_t[0] = tan_og[0]
                tan_og_t[1] = tan_og[1] * np.cos(th2_l2) - tan_og[2] * np.sin(th2_l2)
                tan_og_t[2] = tan_og[1] * np.sin(th2_l2) + tan_og[2] * np.cos(th2_l2)

                alpha = tan_og_t[0]
                beta = tan_og_t[1]
                gamma = tan_og_t[2]

                def root_lyot(t):
                    x = p_m2a[0] + alpha * t
                    y = p_m2a[1] + beta * t
                    z = p_m2a[2] + gamma * t

                    xlyot, ylyot, zlyot = tele_into_lyot(
                        x, y, z
                    )  # take ray end coordinates and convert to M1 coordinates
                    z_ly = z_lyot()  # Z of mirror 1 in M1 coordinates
                    if np.isnan(z_ly) == True:
                        z_ly = 0
                    root = zlyot - z_ly
                    return root

                t_lyot = optimize.brentq(root_lyot, 10, 900)

                # Location of where ray hits M1
                x_ly = p_m2a[0] + alpha * t_lyot
                y_ly = p_m2a[1] + beta * t_lyot
                z_ly = p_m2a[2] + gamma * t_lyot

                ###### in M1 cordinates ##########################
                x_lyot_temp, y_lyot_temp, z_lyot_temp = tele_into_lyot(
                    x_ly, y_ly, z_ly
                )  # p_m1b temp
                x_m2a_temp, y_m2a_temp, z_m2a_temp = tele_into_lyot(
                    x_m2a, y_m2a, z_m2a
                )  # P_1a temp
                norm = d_zlyot()
                norm_temp = np.array([-norm[0], -norm[1], 1])
                n_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
                vec_m2a_lyot = np.array(
                    [x_m2a_temp, y_m2a_temp, z_m2a_temp]
                ) - np.array([x_lyot_temp, y_lyot_temp, z_lyot_temp])
                dist_m2a_lyot = np.sqrt(np.sum(vec_m2a_lyot ** 2))
                tan_m2a_lyot = vec_m2a_lyot / dist_m2a_lyot

                tan_og = snell_vec(n_si, n_vac, n_hat, tan_m2a_lyot)

                # Transform back to telescope cordinates
                n_hat_t = np.zeros(3)
                tan_m2a_lyot_t = np.zeros(3)
                tan_og_t = np.zeros(3)

                n_hat_t[0] = n_hat[0]
                n_hat_t[1] = n_hat[1] * np.cos(th1_l3) - n_hat[2] * np.sin(th1_l3)
                n_hat_t[2] = n_hat[1] * np.sin(th1_l3) + n_hat[2] * np.cos(th1_l3)

                tan_m2a_lyot_t[0] = tan_m2a_lyot[0]
                tan_m2a_lyot_t[1] = tan_m2a_lyot[1] * np.cos(th1_l3) - tan_m2a_lyot[
                    2
                ] * np.sin(th1_l3)
                tan_m2a_lyot_t[2] = tan_m2a_lyot[1] * np.sin(th1_l3) + tan_m2a_lyot[
                    2
                ] * np.cos(th1_l3)

                tan_og_t[0] = tan_og[0]
                tan_og_t[1] = tan_og[1] * np.cos(th1_l3) - tan_og[2] * np.sin(th1_l3)
                tan_og_t[2] = tan_og[1] * np.sin(th1_l3) + tan_og[2] * np.cos(th1_l3)

                total_path_length = (
                    (t_m1b) + (t_m1a * n_si) + (t_m2b) + (t_m2a * n_si) + t_lyot
                )

                amp = 1
                return [
                    x_ly,
                    y_ly,
                    z_ly,
                    total_path_length,
                    amp,  # to determine phase, amp
                    n_hat_t[0],
                    n_hat_t[1],
                    n_hat_t[2],
                    tan_og_t[0],
                    tan_og_t[1],
                    tan_og_t[2],  # 23
                    x_m2b,
                    y_m2b,
                    z_m2b,  # 13
                    x_m2a,
                    y_m2a,
                    z_m2a,
                    x_m1b,
                    y_m1b,
                    z_m1b,
                    x_m1a,
                    y_m1a,
                    z_m1a,
                    x_0,
                    y_0,
                    z_0,
                ]

    def output(self):

        result = []
        for ii in range(self.n_pts):
            result.append(self.trace_rays(ii))
        self.out = np.array(result).transpose()

        return self.out


def regrid(beam1, beam2):

    res_so = int(np.sqrt(len(beam2[0, :])))
    res_rx = int(np.sqrt(len(beam1[0, :])))

    x_mod = np.reshape(beam2[0, :], (res_so, res_so))
    y_mod = np.reshape(beam2[1, :], (res_so, res_so))
    z_mod = np.reshape(beam2[2, :], (res_so, res_so))

    x_1 = np.reshape(beam1[0, :], (res_rx, res_rx))
    y_1 = np.reshape(beam1[1, :], (res_rx, res_rx))
    z_1 = np.reshape(beam1[2, :], (res_rx, res_rx))

    Z_phi_rx = np.reshape(beam1[3, :], (res_rx, res_rx))
    Z_amp_rx = np.reshape(beam1[4, :], (res_rx, res_rx))
    Z_phi_so = np.reshape(beam2[3, :], (res_so, res_so))
    Z_amp_so = np.reshape(beam2[4, :], (res_so, res_so))

    xx = np.linspace(-70, 70, len(x_mod))  # x_mod[:,int(len(x_mod)/2)] # so
    zz = np.linspace(-70, 70, len(x_mod))  # z_mod[int(len(x_mod)/2),:] # so
    #     print(xx,zz)
    X, Z = np.meshgrid(xx, zz)

    xx_1 = np.linspace(-70, 70, res_rx)  # rx
    zz_1 = np.linspace(-70, 70, res_rx)  # rx
    #     print(xx_1,zz_1)
    X1, Z1 = np.meshgrid(xx_1, zz_1)

    x2 = np.linspace(-70, 70, res_so)
    z2 = np.linspace(-70, 70, res_so)
    X2, Z2 = np.meshgrid(x2, z2)

    f_amp_so = interp2d(xx, zz, Z_amp_so, kind="cubic")
    f_phi_so = interp2d(xx, zz, Z_phi_so, kind="cubic")
    f_amp_rx = interp2d(xx_1, zz_1, Z_amp_rx, kind="cubic")
    f_phi_rx = interp2d(xx_1, zz_1, Z_phi_rx, kind="cubic")

    Z_amp_so_new = f_amp_so(x2, z2)
    Z_phi_so_new = f_phi_so(x2, z2)
    Z_amp_rx_new = f_amp_rx(x2, z2)
    Z_phi_rx_new = f_phi_rx(x2, z2)

    beam1_new = np.zeros(np.shape(beam2))
    beam2_new = np.zeros(np.shape(beam2))

    beam1_new[0, :] = np.ravel(X2)
    beam1_new[2, :] = np.ravel(Z2)
    beam1_new[3, :] = np.ravel(Z_phi_so_new)
    beam1_new[4, :] = np.ravel(Z_amp_so_new)

    beam2_new[0, :] = np.ravel(X2)
    beam2_new[2, :] = np.ravel(Z2)
    beam2_new[3, :] = np.ravel(Z_phi_rx_new)
    beam2_new[4, :] = np.ravel(Z_amp_rx_new)

    return beam1_new, beam2_new
