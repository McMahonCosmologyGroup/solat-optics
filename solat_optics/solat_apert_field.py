import multiprocessing as mp
import sys

path_to_package = "/home/chesmore/Desktop/Code/solat-optics"
sys.path.append(path_to_package)

# Holosim packages
import ap_fitting as afit
import far_field as ff
import numpy as np
import pan_mod as pm
from pan_mod import *
from scipy import optimize

import solat_optics.latrt_geo as latrt_geo
from solat_optics.latrt_geo import *

th2 = (-np.pi / 2) - initialize_telescope_geometry.th_2

y_cent_m1 = -7201.003729431267

adj_pos_m1, adj_pos_m2 = pm.get_single_vert_adj_positions()


class RayMirrorPts:
    def __init__(self, tele_geo, theta, phi, el, az):
        theta, phi = np.meshgrid(theta, phi)
        theta = np.ravel(theta)
        phi = np.ravel(phi)

        # find center of focal plane
        x = np.linspace(-4000, 4000, 100)  # [mm]
        y = np.linspace(-4000, 4000, 100)  # [mm]
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X ** 2 + Y ** 2)
        Zf = z_focal(X, Y)
        Zf = np.where(r < 3600, Zf, np.nan)
        Xtf, Ytf, Ztf = foc_into_tele(X, Y, Zf, el, az)
        xf_mean = np.mean(
            Xtf[int(len(Xtf) / 2), :][
                np.where(np.isnan(Xtf[int(len(Xtf) / 2), :]) != True)
            ]
        )
        yf_mean = np.mean(
            Ytf[:, int(len(Ytf) / 2)][
                np.where(np.isnan(Ytf[:, int(len(Ytf) / 2)]) != True)
            ]
        )
        zf_mean = np.mean(
            Ztf[:, int(len(Ytf) / 2)][
                np.where(np.isnan(Ytf[:, int(len(Ytf) / 2)]) != True)
            ]
        )

        # output array
        n_pts = len(theta)
        out = np.zeros((9, n_pts))
        out[6, 0] = xf_mean
        out[7, 0] = yf_mean
        out[8, 0] = zf_mean

        # class instance variables
        self.tele_geo = tele_geo
        self.theta = theta
        self.phi = phi
        self.el = el
        self.az = az
        self.n_pts = n_pts
        self.out = out

    #     def plot_setup(self):
    #         el = self.el
    #         az = self.az

    #         x = np.linspace(-4000,4000,100) # [mm]
    #         y = np.linspace(-4000,4000,100) # [mm]
    #         X,Y = np.meshgrid(x,y)
    #         r = np.sqrt(X**2 + Y**2)
    #         Z1 = z1(X,Y)
    #         Z1 = np.where(r<3600,Z1,np.nan)

    #         Xt1,Yt1,Zt1 = m1_into_tele(X,Y,Z1)
    #         X1,Y1,Z1 = rotate_az_el(Xt1[:,int(len(Yt1)/2)],Yt1[:,int(len(Yt1)/2)],Zt1[:,int(len(Yt1)/2)],el,az)
    #         plt.plot(Y1,Z1, linewidth=4, color = 'b')

    #         x = np.linspace(-4000,4000,100) # [mm]
    #         y = np.linspace(-4000,4000,100) # [mm]
    #         X,Y = np.meshgrid(x,y)
    #         r = np.sqrt(X**2 + Y**2)
    #         Z2 = z2(X,Y)
    #         Z2 = np.where(r<3600,Z2,np.nan)

    #         Xt2,Yt2,Zt2 = m2_into_tele(X,Y,Z2)
    #         X2,Y2,Z2 = rotate_az_el(Xt2[:,int(len(Yt2)/2)],Yt2[:,int(len(Yt2)/2)],Zt2[:,int(len(Yt2)/2)],el,az)
    #         plt.plot(Y2,Z2, linewidth=4, color = 'b')

    #         x = np.linspace(-4000,4000,100) # [mm]
    #         y = np.linspace(-4000,4000,100) # [mm]
    #         X,Y = np.meshgrid(x,y)
    #         r = np.sqrt(X**2 + Y**2)
    #         Zf = z_focal(X,Y)
    #         Zf = np.where(r<3600,Zf,np.nan)
    #         Xtf,Ytf,Ztf = foc_into_tele(X,Y,Zf,el,az)
    #         plt.plot(Ytf[:,int(len(Yt2)/2)],Ztf[:,int(len(Yt2)/2)], linewidth=4, color = 'b')

    def trace_rays(self, ii):
        tele_geo = self.tele_geo
        theta = self.theta
        phi = self.phi
        el = self.el
        az = self.az

        # Define the outgoing ray's direction
        th = theta[ii]
        ph = phi[ii]
        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # Receiver feed position [mm] in telescope r.f.
        x_0 = tele_geo.x_tow * 1e3
        y_0 = tele_geo.y_tow * 1e3
        z_0 = tele_geo.z_tow * 1e3

        def root_ap(t):

            # Endpoint of ray:
            x = x_0 + alpha * t
            y = y_0 + beta * t
            z = z_0 + gamma * t

            # Convert to M1 r.f.
            xap, yap, zap = tele_into_apert(x, y, z, -el, -az)

            # Z of mirror in M1 r.f.
            z_apt = z_ap(xap, yap)

            return zap - z_apt

        d_0ap = np.sqrt(x_0 ** 2 + (y_0 + 7200) ** 2 + (z_0 - 4000) ** 2)
        lb = max(0, d_0ap - 0.5e5)
        ub = d_0ap + 0.5e5
        t_ap = optimize.brentq(root_ap, lb, ub)

        x_a = x_0 + alpha * t_ap
        y_a = y_0 + beta * t_ap
        z_a = z_0 + gamma * t_ap
        P_a = [x_a, y_a, z_a]

        if x_a ** 2 + (y_a - y_cent_m1) ** 2 > 3e3 ** 2:
            return [0, 0, 0, 0, 0, 0]
        else:

            def root_z1(t):

                # Endpoint of ray:
                x = P_a[0] + alpha * t
                y = P_a[1] + beta * t
                z = P_a[2] + gamma * t

                # Convert to M1 r.f.
                xm1, ym1, zm1 = tele_into_m1(x, y, z, -el, -az)

                # Z of mirror in M1 r.f.
                z_m1 = z1(xm1, ym1)

                return zm1 - z_m1

            # print(P_a)
            # x = np.linspace(0, 2e4, 200)
            # y = root_z1(x)
            # plt.plot(x, y)
            # plt.show()
            t_m1 = optimize.brentq(root_z1, 0, 2e4)

            # Endpoint of ray:
            x_m1 = P_a[0] + alpha * t_m1
            y_m1 = P_a[1] + beta * t_m1
            z_m1 = P_a[2] + gamma * t_m1
            P_m1 = [x_m1, y_m1, z_m1]

            ########## M1 r.f ###########################################################

            x_m1_temp, y_m1_temp, z_m1_temp = tele_into_m1(
                P_m1[0], P_m1[1], P_m1[2], -el, -az
            )
            x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m1(x_0, y_0, z_0, -el, -az)

            # Normal vector of ray on M2
            norm = d_z1(x_m1_temp, y_m1_temp)
            norm_temp = np.array([-norm[0], -norm[1], 1])
            N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

            # Normalized vector from RX to M2
            vec_rx_m1 = np.array([x_m1_temp, y_m1_temp, z_m1_temp]) - np.array(
                [x_rx_temp, y_rx_temp, z_rx_temp]
            )
            dist_rx_m1 = np.sqrt(np.sum(vec_rx_m1 ** 2))
            tan_rx_m1 = vec_rx_m1 / dist_rx_m1

            # Vector of outgoing ray

            tan_og = tan_rx_m1 - 2 * np.dot(np.sum(np.dot(tan_rx_m1, N_hat)), N_hat)

            # Transform back to telescope cordinates
            N_hat_t = np.zeros(3)

            th1 = initialize_telescope_geometry.th_1

            N_hat_x_temp = N_hat[0] * np.cos(np.pi) + N_hat[2] * np.sin(np.pi)
            N_hat_y_temp = N_hat[1]
            N_hat_z_temp = -N_hat[2] * np.sin(np.pi) + N_hat[2] * np.cos(np.pi)

            N_hat_t[0] = N_hat_x_temp
            N_hat_t[1] = N_hat_y_temp * np.cos(th1) - N_hat_z_temp * np.sin(th1)
            N_hat_t[2] = N_hat_y_temp * np.sin(th1) + N_hat_z_temp * np.cos(th1)

            tan_rx_m1_t = np.zeros(3)
            tan_rx_m1_x_temp = tan_rx_m1[0] * np.cos(np.pi) + tan_rx_m1[2] * np.sin(
                np.pi
            )
            tan_rx_m1_y_temp = tan_rx_m1[1]
            tan_rx_m1_z_temp = -tan_rx_m1[2] * np.sin(np.pi) + tan_rx_m1[2] * np.cos(
                np.pi
            )

            tan_rx_m1_t[0] = tan_rx_m1_x_temp
            tan_rx_m1_t[1] = tan_rx_m1_y_temp * np.cos(th1) - tan_rx_m1_z_temp * np.sin(
                th1
            )
            tan_rx_m1_t[2] = tan_rx_m1_y_temp * np.sin(th1) + tan_rx_m1_z_temp * np.cos(
                th1
            )

            tan_og_t = np.zeros(3)
            tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
            tan_og_y_temp = tan_og[1]
            tan_og_z_temp = -tan_og[2] * np.sin(np.pi) + tan_og[2] * np.cos(np.pi)

            tan_og_t[0] = tan_og_x_temp
            tan_og_t[1] = tan_og_y_temp * np.cos(th1) - tan_og_z_temp * np.sin(th1)
            tan_og_t[2] = tan_og_y_temp * np.sin(th1) + tan_og_z_temp * np.cos(th1)

            # rotate el
            x_temp = tan_og_t[0]
            y_temp = np.cos(el) * tan_og_t[1] - np.sin(el) * tan_og_t[2]
            z_temp = np.sin(el) * tan_og_t[1] + np.cos(el) * tan_og_t[2]

            # rotate az
            tan_og_t[0] = np.cos(az) * x_temp + np.sin(az) * z_temp
            tan_og_t[1] = y_temp
            tan_og_t[2] = -np.sin(az) * x_temp + np.cos(az) * z_temp

            ########## Tele. r.f ###########################################################

            # Vector of outgoing ray:
            alpha = tan_og_t[0]
            beta = tan_og_t[1]
            gamma = tan_og_t[2]

            # Use a root finder to find where the ray intersects with M2
            def root_z2(t):

                # Endpoint of ray:
                x = x_m1 + alpha * t
                y = y_m1 + beta * t
                z = z_m1 + gamma * t

                # Convert to M2 r.f.
                xm2, ym2, zm2 = tele_into_m2(x, y, z, -el, -az)

                # Z of mirror in M2 r.f.
                z_m2 = z2(xm2, ym2)
                return zm2 - z_m2

            # Endpoint of ray:
            t_m2 = optimize.brentq(root_z2, 0, 13e3)
            x_m2 = x_m1 + alpha * t_m2
            y_m2 = y_m1 + beta * t_m2
            z_m2 = z_m1 + gamma * t_m2

            return [x_m2, y_m2, z_m2, x_m1, y_m1, z_m1]

    def output(self):
        if (
            self.n_pts > 3000
        ):  # a threshold to use parallelism (it's a empirical number tested with Ryzen 5 3600)
            pool = mp.Pool(max(1, mp.cpu_count() - 1))
            # result = list(pool.imap(self.trace_rays, range(self.n_pts)))
            result = pool.map(self.trace_rays, range(self.n_pts))
            pool.close()
            pool.join()
        else:
            result = []
            for ii in range(self.n_pts):
                result.append(self.trace_rays(ii))

        self.out[0:6] = np.array(result).transpose()
        return self.out


def ray_mirror_pts(tele_geo, theta, phi, el, az):
    color = "c"
    alph = 0.6
    theta, phi = np.meshgrid(theta, phi)
    theta = np.ravel(theta)
    phi = np.ravel(phi)

    # Read in telescope geometry values
    th2 = tele_geo.th2

    horn_fwhp = tele_geo.th_fwhp
    N_linear = tele_geo.N_scan

    n_pts = len(theta)
    out = np.zeros((9, n_pts))

    x = np.linspace(-4000, 4000, 100)  # [mm]
    y = np.linspace(-4000, 4000, 100)  # [mm]
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)
    Zf = z_focal(X, Y)
    Zf = np.where(r < 3600, Zf, np.nan)
    Xtf, Ytf, Ztf = foc_into_tele(X, Y, Zf, el, az)

    xf_mean = np.mean(
        Xtf[int(len(Xtf) / 2), :][np.where(np.isnan(Xtf[int(len(Xtf) / 2), :]) != True)]
    )
    yf_mean = np.mean(
        Ytf[:, int(len(Ytf) / 2)][np.where(np.isnan(Ytf[:, int(len(Ytf) / 2)]) != True)]
    )
    zf_mean = np.mean(
        Ztf[:, int(len(Ytf) / 2)][np.where(np.isnan(Ytf[:, int(len(Ytf) / 2)]) != True)]
    )

    for ii in range(n_pts):

        # Define the outgoing ray's direction
        th = theta[ii]
        ph = phi[ii]
        r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

        alpha = r_hat[0]
        beta = r_hat[1]
        gamma = r_hat[2]

        # Receiver feed position [mm] in telescope r.f.
        x_0 = tele_geo.x_tow * 1e3
        y_0 = tele_geo.y_tow * 1e3
        z_0 = tele_geo.z_tow * 1e3

        def root_ap(t):

            # Endpoint of ray:
            x = x_0 + alpha * t
            y = y_0 + beta * t
            z = z_0 + gamma * t

            # Convert to M1 r.f.
            xap, yap, zap = tele_into_apert(x, y, z, -el, -az)

            # Z of mirror in M1 r.f.
            z_apt = z_ap(xap, yap)

            return zap - z_apt

        t_ap = optimize.brentq(root_ap, 1e4, 1e9)
        x_a = x_0 + alpha * t_ap
        y_a = y_0 + beta * t_ap
        z_a = z_0 + gamma * t_ap
        P_a = [x_a, y_a, z_a]

        if x_a ** 2 + (y_a - y_cent_m1) ** 2 > 3e3 ** 2:
            continue

        def root_z1(t):

            # Endpoint of ray:
            x = P_a[0] + alpha * t
            y = P_a[1] + beta * t
            z = P_a[2] + gamma * t

            # Convert to M1 r.f.
            xm1, ym1, zm1 = tele_into_m1(x, y, z, -el, -az)

            # Z of mirror in M1 r.f.
            z_m1 = z1(xm1, ym1)

            return zm1 - z_m1

        t_m1 = optimize.brentq(root_z1, 0, 5e4)

        # Endpoint of ray:
        x_m1 = P_a[0] + alpha * t_m1
        y_m1 = P_a[1] + beta * t_m1
        z_m1 = P_a[2] + gamma * t_m1
        P_m1 = [x_m1, y_m1, z_m1]

        ########## M1 r.f ###########################################################

        x_m1_temp, y_m1_temp, z_m1_temp = tele_into_m1(
            P_m1[0], P_m1[1], P_m1[2], -el, -az
        )
        x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m1(x_0, y_0, z_0, -el, -az)

        # Normal vector of ray on M2
        norm = d_z1(x_m1_temp, y_m1_temp)
        norm_temp = np.array([-norm[0], -norm[1], 1])
        N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

        # Normalized vector from RX to M2
        vec_rx_m1 = np.array([x_m1_temp, y_m1_temp, z_m1_temp]) - np.array(
            [x_rx_temp, y_rx_temp, z_rx_temp]
        )
        dist_rx_m1 = np.sqrt(np.sum(vec_rx_m1 ** 2))
        tan_rx_m1 = vec_rx_m1 / dist_rx_m1

        # Vector of outgoing ray

        tan_og = tan_rx_m1 - 2 * np.dot(np.sum(np.dot(tan_rx_m1, N_hat)), N_hat)

        # Transform back to telescope cordinates
        N_hat_t = np.zeros(3)

        th1 = initialize_telescope_geometry.th_1

        N_hat_x_temp = N_hat[0] * np.cos(np.pi) + N_hat[2] * np.sin(np.pi)
        N_hat_y_temp = N_hat[1]
        N_hat_z_temp = -N_hat[2] * np.sin(np.pi) + N_hat[2] * np.cos(np.pi)

        N_hat_t[0] = N_hat_x_temp
        N_hat_t[1] = N_hat_y_temp * np.cos(th1) - N_hat_z_temp * np.sin(th1)
        N_hat_t[2] = N_hat_y_temp * np.sin(th1) + N_hat_z_temp * np.cos(th1)

        tan_rx_m1_t = np.zeros(3)
        tan_rx_m1_x_temp = tan_rx_m1[0] * np.cos(np.pi) + tan_rx_m1[2] * np.sin(np.pi)
        tan_rx_m1_y_temp = tan_rx_m1[1]
        tan_rx_m1_z_temp = -tan_rx_m1[2] * np.sin(np.pi) + tan_rx_m1[2] * np.cos(np.pi)

        tan_rx_m1_t[0] = tan_rx_m1_x_temp
        tan_rx_m1_t[1] = tan_rx_m1_y_temp * np.cos(th1) - tan_rx_m1_z_temp * np.sin(th1)
        tan_rx_m1_t[2] = tan_rx_m1_y_temp * np.sin(th1) + tan_rx_m1_z_temp * np.cos(th1)

        tan_og_t = np.zeros(3)
        tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
        tan_og_y_temp = tan_og[1]
        tan_og_z_temp = -tan_og[2] * np.sin(np.pi) + tan_og[2] * np.cos(np.pi)

        tan_og_t[0] = tan_og_x_temp
        tan_og_t[1] = tan_og_y_temp * np.cos(th1) - tan_og_z_temp * np.sin(th1)
        tan_og_t[2] = tan_og_y_temp * np.sin(th1) + tan_og_z_temp * np.cos(th1)

        # rotate el
        x_temp = tan_og_t[0]
        y_temp = np.cos(el) * tan_og_t[1] - np.sin(el) * tan_og_t[2]
        z_temp = np.sin(el) * tan_og_t[1] + np.cos(el) * tan_og_t[2]

        # rotate az
        tan_og_t[0] = np.cos(az) * x_temp + np.sin(az) * z_temp
        tan_og_t[1] = y_temp
        tan_og_t[2] = -np.sin(az) * x_temp + np.cos(az) * z_temp

        ########## Tele. r.f ###########################################################

        # Vector of outgoing ray:
        alpha = tan_og_t[0]
        beta = tan_og_t[1]
        gamma = tan_og_t[2]

        # Use a root finder to find where the ray intersects with M2
        def root_z2(t):

            # Endpoint of ray:
            x = x_m1 + alpha * t
            y = y_m1 + beta * t
            z = z_m1 + gamma * t

            # Convert to M2 r.f.
            xm2, ym2, zm2 = tele_into_m2(x, y, z, -el, -az)

            # Z of mirror in M2 r.f.
            z_m2 = z2(xm2, ym2)
            return zm2 - z_m2

        t_m2 = optimize.brentq(root_z2, 0, 20e3)

        #         if np.mod(ii,30)==0:

        #             x = np.linspace(-4000,4000,100) # [mm]
        #             y = np.linspace(-4000,4000,100) # [mm]
        #             X,Y = np.meshgrid(x,y)
        #             r = np.sqrt(X**2 + Y**2)
        #             Z1 = z1(X,Y)
        #             Z1 = np.where(r<3600,Z1,np.nan)

        #             Xt1,Yt1,Zt1 = m1_into_tele(X,Y,Z1)
        #             X1,Y1,Z1 = rotate_az_el(Xt1[:,int(len(Yt1)/2)],Yt1[:,int(len(Yt1)/2)],Zt1[:,int(len(Yt1)/2)],el,az)
        #             plt.plot(Y1,Z1,color = 'b')

        #             x = np.linspace(-4000,4000,100) # [mm]
        #             y = np.linspace(-4000,4000,100) # [mm]
        #             X,Y = np.meshgrid(x,y)
        #             r = np.sqrt(X**2 + Y**2)
        #             Z2 = z2(X,Y)
        #             Z2 = np.where(r<3600,Z2,np.nan)

        #             Xt2,Yt2,Zt2 = m2_into_tele(X,Y,Z2)
        #             X2,Y2,Z2 = rotate_az_el(Xt2[:,int(len(Yt2)/2)],Yt2[:,int(len(Yt2)/2)],Zt2[:,int(len(Yt2)/2)],el,az)
        #             plt.plot(Y2,Z2,color = 'b')

        #             x = np.linspace(-4000,4000,100) # [mm]
        #             y = np.linspace(-4000,4000,100) # [mm]
        #             X,Y = np.meshgrid(x,y)
        #             r = np.sqrt(X**2 + Y**2)
        #             Zf = z_focal(X,Y)
        #             Zf = np.where(r<3600,Zf,np.nan)
        #             Xtf,Ytf,Ztf = foc_into_tele(X,Y,Zf,el,az)
        #             plt.plot(Ytf[:,int(len(Yt2)/2)],Ztf[:,int(len(Yt2)/2)],color = 'b')

        #             ### X #######
        #             x = np.linspace(-4000,4000,100) # [mm]
        #             y = np.linspace(-4000,4000,100) # [mm]
        #             X,Y = np.meshgrid(x,y)
        #             r = np.sqrt(X**2 + Y**2)
        #             Z1 = z1(X,Y)
        #             Z1 = np.where(r<3600,Z1,np.nan)

        #             Xt1,Yt1,Zt1 = m1_into_tele(X,Y,Z1)
        #             X1,Y1,Z1 = rotate_az_el(Xt1[int(len(Yt1)/2),:],Yt1[int(len(Yt1)/2),:],Zt1[int(len(Yt1)/2),:],el,az)
        #             plt.plot(X1,Y1,color = 'b')

        #             x = np.linspace(-4000,4000,100) # [mm]
        #             y = np.linspace(-4000,4000,100) # [mm]
        #             X,Y = np.meshgrid(x,y)
        #             r = np.sqrt(X**2 + Y**2)
        #             Z2 = z2(X,Y)
        #             Z2 = np.where(r<3600,Z2,np.nan)

        #             Xt2,Yt2,Zt2 = m2_into_tele(X,Y,Z2)
        #             X2,Y2,Z2 = rotate_az_el(Xt2[int(len(Yt2)/2),:],Yt2[int(len(Yt2)/2),:],Zt2[int(len(Yt2)/2),:],el,az)
        #             plt.plot(X2,Y2,color = 'b')

        # Endpoint of ray:
        x_m2 = x_m1 + alpha * t_m2
        y_m2 = y_m1 + beta * t_m2
        z_m2 = z_m1 + gamma * t_m2
        P_m2 = np.array([x_m2, y_m2, z_m2])

        # Write out
        out[0, ii] = x_m2
        out[1, ii] = y_m2
        out[2, ii] = z_m2

        out[3, ii] = x_m1
        out[4, ii] = y_m1
        out[5, ii] = z_m1

    out[6, 0] = xf_mean
    out[7, 0] = yf_mean
    out[8, 0] = zf_mean

    return out


def aperature_fields_from_panel_model(
    panel_model1, panel_model2, tele_geo, theta, phi, rxmirror, el, az
):

    alph = 0.5
    color = "lime"
    theta, phi = np.meshgrid(theta, phi)
    theta = np.ravel(theta)
    phi = np.ravel(phi)

    th2 = tele_geo.th2

    y_ap = tele_geo.rx_y * 1e3
    horn_fwhp = tele_geo.th_fwhp
    focal = tele_geo.F_2
    # Step 1:  grid the plane of rays shooting out of receiver feed
    N_linear = tele_geo.N_scan
    col_m2 = adj_pos_m2[0]
    row_m2 = adj_pos_m2[1]
    x_adj_m2 = adj_pos_m2[4]
    y_adj_m2 = adj_pos_m2[3]
    col_m1 = adj_pos_m1[0]
    row_m1 = adj_pos_m1[1]
    x_adj_m1 = adj_pos_m1[2]
    y_adj_m1 = adj_pos_m1[3]

    x_panm_m2 = np.reshape(
        rxmirror[0, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )
    y_panm_m2 = np.reshape(
        rxmirror[2, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )
    x_panm_m1 = np.reshape(
        rxmirror[3, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )
    y_panm_m1 = np.reshape(
        rxmirror[4, :], (int(np.sqrt(len(phi))), int(np.sqrt(len(phi))))
    )

    pan_id_m2 = identify_panel(x_panm_m2, y_panm_m2, x_adj_m2, y_adj_m2, col_m2, row_m2)
    pan_id_m1 = identify_panel(
        x_panm_m1, y_panm_m1 - y_cent_m1, x_adj_m1, y_adj_m1, col_m1, row_m1
    )

    row_panm_m2 = np.ravel(pan_id_m2[0, :, :])
    col_panm_m2 = np.ravel(pan_id_m2[1, :, :])
    row_panm_m1 = np.ravel(pan_id_m1[0, :, :])
    col_panm_m1 = np.ravel(pan_id_m1[1, :, :])

    # Step 2: calculate the position + local surface normal for the dish
    n_pts = len(theta)
    out = np.zeros((17, n_pts))

    for ii in range(n_pts):
        i_row = row_panm_m1[ii]
        i_col = col_panm_m1[ii]
        i_panm = np.where((panel_model1[0, :] == i_row) & (panel_model1[1, :] == i_col))

        if len(i_panm[0]) != 0:

            th = theta[ii]
            ph = phi[ii]
            r_hat = [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]

            alpha = r_hat[0]
            beta = r_hat[1]
            gamma = r_hat[2]

            # Source feed position [mm] (in telescope reference frame):
            x_0 = tele_geo.x_tow * 1e3  # convert to [mm]
            y_0 = tele_geo.y_tow * 1e3  # convert to [mm]
            z_0 = tele_geo.z_tow * 1e3  # convert to [mm]

            def root_ap(t):

                # Endpoint of ray:
                x = x_0 + alpha * t
                y = y_0 + beta * t
                z = z_0 + gamma * t

                # Convert to M1 r.f.
                xap, yap, zap = tele_into_apert(x, y, z, -el, -az)

                # Z of mirror in M1 r.f.
                z_apt = z_ap(xap, yap)
                return zap - z_apt

            t_ap = optimize.brentq(root_ap, 1e4, 1e9)

            x_a = x_0 + alpha * t_ap
            y_a = y_0 + beta * t_ap
            z_a = z_0 + gamma * t_ap
            P_a = [x_a, y_a, z_a]

            if x_a ** 2 + (y_a - y_cent_m1) ** 2 > 3e3 ** 2:
                continue

            a = panel_model1[2, i_panm]
            b = panel_model1[3, i_panm]
            c = panel_model1[4, i_panm]
            d = panel_model1[5, i_panm]
            e = panel_model1[6, i_panm]
            f = panel_model1[7, i_panm]
            x0 = panel_model1[8, i_panm]
            y0 = panel_model1[9, i_panm]

            def root_z1(t):
                x = P_a[0] + alpha * t
                y = P_a[1] + beta * t
                z = P_a[2] + gamma * t
                xm1, ym1, zm1 = tele_into_m1(x, y, z, -el, -az)
                # take ray end coordinates and convert to M1 coordinates

                xm1_err, ym1_err, zm1_err = tele_into_m1(x, y, z, -el, -az)

                x_temp = xm1_err * np.cos(np.pi) + zm1_err * np.sin(np.pi)
                y_temp = ym1_err
                z_temp = -xm1_err * np.sin(np.pi) + zm1_err * np.cos(np.pi)

                xpc = x_temp - x0
                ypc = y_temp - y0

                z_err = (
                    a
                    + b * xpc
                    + c * (ypc)
                    + d * (xpc ** 2 + ypc ** 2)
                    + e * (xpc * ypc)
                )

                z_err = z_err[0][0]
                z_m1 = z1(xm1, ym1)  # Z of mirror 1 in M1 coordinates

                root = zm1 - (z_m1 + z_err)
                return root

            t_m1 = optimize.brentq(root_z1, 0, 1e4)

            # Endpoint of ray:
            x_m1 = P_a[0] + alpha * t_m1
            y_m1 = P_a[1] + beta * t_m1
            z_m1 = P_a[2] + gamma * t_m1
            P_m1 = [x_m1, y_m1, z_m1]

            x_m1_temp, y_m1_temp, z_m1_temp = tele_into_m1(
                P_m1[0], P_m1[1], P_m1[2], -el, -az
            )
            x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m1(x_0, y_0, z_0, -el, -az)

            # Normal vector of ray on M2
            norm = d_z1(x_m1_temp, y_m1_temp)
            norm_temp = np.array([-norm[0], -norm[1], 1])
            N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))

            # Normalized vector from RX to M2
            vec_rx_m1 = np.array([x_m1_temp, y_m1_temp, z_m1_temp]) - np.array(
                [x_rx_temp, y_rx_temp, z_rx_temp]
            )
            dist_rx_m1 = np.sqrt(np.sum(vec_rx_m1 ** 2))
            tan_rx_m1 = vec_rx_m1 / dist_rx_m1

            # Vector of outgoing ray

            tan_og = tan_rx_m1 - 2 * np.dot(np.sum(np.dot(tan_rx_m1, N_hat)), N_hat)

            # Transform back to telescope cordinates
            N_hat_t = np.zeros(3)

            th1 = initialize_telescope_geometry.th_1

            N_hat_x_temp = N_hat[0] * np.cos(np.pi) + N_hat[2] * np.sin(np.pi)
            N_hat_y_temp = N_hat[1]
            N_hat_z_temp = -N_hat[2] * np.sin(np.pi) + N_hat[2] * np.cos(np.pi)

            N_hat_t[0] = N_hat_x_temp
            N_hat_t[1] = N_hat_y_temp * np.cos(th1) - N_hat_z_temp * np.sin(th1)
            N_hat_t[2] = N_hat_y_temp * np.sin(th1) + N_hat_z_temp * np.cos(th1)
            # rotate el
            x_temp = N_hat_t[0]
            y_temp = np.cos(el) * N_hat_t[1] - np.sin(el) * N_hat_t[2]
            z_temp = np.sin(el) * N_hat_t[1] + np.cos(el) * N_hat_t[2]

            # rotate az
            N_hat_t[0] = np.cos(az) * x_temp + np.sin(az) * z_temp
            N_hat_t[1] = y_temp
            N_hat_t[2] = -np.sin(az) * x_temp + np.cos(az) * z_temp

            tan_rx_m1_t = np.zeros(3)

            tan_rx_m1_x_temp = tan_rx_m1[0] * np.cos(np.pi) + tan_rx_m1[2] * np.sin(
                np.pi
            )
            tan_rx_m1_y_temp = tan_rx_m1[1]
            tan_rx_m1_z_temp = -tan_rx_m1[2] * np.sin(np.pi) + tan_rx_m1[2] * np.cos(
                np.pi
            )

            tan_rx_m1_t[0] = tan_rx_m1_x_temp
            tan_rx_m1_t[1] = tan_rx_m1_y_temp * np.cos(th1) - tan_rx_m1_z_temp * np.sin(
                th1
            )
            tan_rx_m1_t[2] = tan_rx_m1_y_temp * np.sin(th1) + tan_rx_m1_z_temp * np.cos(
                th1
            )

            tan_og_t = np.zeros(3)
            tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
            tan_og_y_temp = tan_og[1]
            tan_og_z_temp = -tan_og[2] * np.sin(np.pi) + tan_og[2] * np.cos(np.pi)

            tan_og_t[0] = tan_og_x_temp
            tan_og_t[1] = tan_og_y_temp * np.cos(th1) - tan_og_z_temp * np.sin(th1)
            tan_og_t[2] = tan_og_y_temp * np.sin(th1) + tan_og_z_temp * np.cos(th1)
            # rotate el
            x_temp = tan_og_t[0]
            y_temp = np.cos(el) * tan_og_t[1] - np.sin(el) * tan_og_t[2]
            z_temp = np.sin(el) * tan_og_t[1] + np.cos(el) * tan_og_t[2]

            # rotate az
            tan_og_t[0] = np.cos(az) * x_temp + np.sin(az) * z_temp
            tan_og_t[1] = y_temp
            tan_og_t[2] = -np.sin(az) * x_temp + np.cos(az) * z_temp

            #             if np.mod(ii,71)==0:
            #                 plt.plot([y_m1,y_m1+200*tan_og_t[1]],[z_m1,z_m1+200*tan_og_t[2]],color = 'r')
            #                 plt.plot([y_m1,y_m1+200*N_hat_t[1]],[z_m1,z_m1+200*N_hat_t[2]],color = 'b')
            #                 plt.plot([y_m1,P_a[1]],[z_m1,P_a[2]],color = color,alpha = alph)
            #                 plt.plot([y_a,y_0],[z_a,z_0],color = 'm',alpha = alph)

            # Vector of outgoing ray:
            alpha = tan_og_t[0]
            beta = tan_og_t[1]
            gamma = tan_og_t[2]

            i_row = row_panm_m2[ii]
            i_col = col_panm_m2[ii]
            i_panm = np.where(
                (panel_model2[0, :] == i_row) & (panel_model2[1, :] == i_col)
            )
            if len(i_panm[0]) != 0:

                a = panel_model2[2, i_panm]
                b = panel_model2[3, i_panm]
                c = panel_model2[4, i_panm]
                d = panel_model2[5, i_panm]
                e = panel_model2[6, i_panm]
                f = panel_model2[7, i_panm]
                x0 = panel_model2[8, i_panm]
                y0 = panel_model2[9, i_panm]

                def root_z2(t):
                    x = x_m1 + alpha * t
                    y = y_m1 + beta * t
                    z = z_m1 + gamma * t
                    xm2, ym2, zm2 = tele_into_m2(
                        x, y, z, -el, -az
                    )  # Convert ray's endpoint into M2 coordinates

                    if z_0 != 0:
                        z /= np.cos(np.arctan(1 / 3))
                    xm2_err, ym2_err, zm2_err = tele_into_m2(
                        x, y, z, -el, -az
                    )  # Convert ray's endpoint into M2 coordinates

                    x_temp = xm2_err * np.cos(np.pi) + zm2_err * np.sin(np.pi)
                    y_temp = ym2_err
                    z_temp = -xm2_err * np.sin(np.pi) + zm2_err * np.cos(np.pi)

                    xpc = x_temp - x0
                    ypc = y_temp - y0

                    z_err = (
                        a
                        + b * xpc
                        + c * (ypc)
                        + d * (xpc ** 2 + ypc ** 2)
                        + e * (xpc * ypc)
                    )
                    z_err = z_err[0][0]

                    z_m2 = z2(xm2, ym2)  # Z of mirror in M2 coordinates

                    root = zm2 - (z_m2 + z_err)
                    return root

                t_m2 = optimize.brentq(root_z2, 1e3, 13e3)

                # Location of where ray hits M2
                x_m2 = x_m1 + alpha * t_m2
                y_m2 = y_m1 + beta * t_m2
                z_m2 = z_m1 + gamma * t_m2

                # Using x and y in M2 coordiantes, find the z err:

                P_m2 = np.array([x_m2, y_m2, z_m2])

                ###### in M2 coordinates ##########################
                x_m2_temp, y_m2_temp, z_m2_temp = tele_into_m2(
                    x_m2, y_m2, z_m2, -el, -az
                )  # P_m2 temp
                x_rx_temp, y_rx_temp, z_rx_temp = tele_into_m2(
                    x_m1, y_m1, z_m1, -el, -az
                )  # P_rx temp
                norm = d_z2(x_m2_temp, y_m2_temp)
                norm_temp = np.array([-norm[0], -norm[1], 1])
                N_hat = norm_temp / np.sqrt(sum(norm_temp ** 2))
                vec_rx_m2 = np.array([x_m2_temp, y_m2_temp, z_m2_temp]) - np.array(
                    [x_rx_temp, y_rx_temp, z_rx_temp]
                )
                dist_rx_m2 = np.sqrt(np.sum(vec_rx_m2 ** 2))
                tan_rx_m2 = vec_rx_m2 / dist_rx_m2

                # Outgoing ray
                tan_og = tan_rx_m2 - 2 * np.dot(np.sum(np.dot(tan_rx_m2, N_hat)), N_hat)

                # Transform back to telescope cordinates
                N_hat_t = np.zeros(3)

                N_hat_x_temp = N_hat[0] * np.cos(np.pi) - N_hat[1] * np.sin(np.pi)
                N_hat_y_temp = N_hat[0] * np.sin(np.pi) + N_hat[1] * np.cos(np.pi)
                N_hat_z_temp = N_hat[2]

                N_hat_t[0] = N_hat_x_temp
                N_hat_t[1] = N_hat_y_temp * np.cos(th2) - N_hat_z_temp * np.sin(th2)
                N_hat_t[2] = N_hat_y_temp * np.sin(th2) + N_hat_z_temp * np.cos(th2)
                # rotate el
                x_temp = N_hat_t[0]
                y_temp = np.cos(el) * N_hat_t[1] - np.sin(el) * N_hat_t[2]
                z_temp = np.sin(el) * N_hat_t[1] + np.cos(el) * N_hat_t[2]

                # rotate az
                N_hat_t[0] = np.cos(az) * x_temp + np.sin(az) * z_temp
                N_hat_t[1] = y_temp
                N_hat_t[2] = -np.sin(az) * x_temp + np.cos(az) * z_temp

                tan_rx_m2_t = np.zeros(3)

                tan_rx_m2_x_temp = tan_rx_m2[0] * np.cos(np.pi) - tan_rx_m2[1] * np.sin(
                    np.pi
                )
                tan_rx_m2_y_temp = tan_rx_m2[0] * np.sin(np.pi) + tan_rx_m2[1] * np.cos(
                    np.pi
                )
                tan_rx_m2_z_temp = tan_rx_m2[2]

                tan_rx_m2_t[0] = tan_rx_m2_x_temp
                tan_rx_m2_t[1] = tan_rx_m2_y_temp * np.cos(
                    th2
                ) - tan_rx_m2_z_temp * np.sin(th2)
                tan_rx_m2_t[2] = tan_rx_m2_y_temp * np.sin(
                    th2
                ) + tan_rx_m2_z_temp * np.cos(th2)

                tan_og_t = np.zeros(3)
                tan_og_x_temp = tan_og[0] * np.cos(np.pi) - tan_og[1] * np.sin(np.pi)
                tan_og_y_temp = tan_og[0] * np.sin(np.pi) + tan_og[1] * np.cos(np.pi)
                tan_og_z_temp = tan_og[2]

                tan_og_t[0] = tan_og_x_temp
                tan_og_t[1] = tan_og_y_temp * np.cos(th2) - tan_og_z_temp * np.sin(th2)
                tan_og_t[2] = tan_og_y_temp * np.sin(th2) + tan_og_z_temp * np.cos(th2)

                # rotate el
                x_temp = tan_og_t[0]
                y_temp = np.cos(el) * tan_og_t[1] - np.sin(el) * tan_og_t[2]
                z_temp = np.sin(el) * tan_og_t[1] + np.cos(el) * tan_og_t[2]

                # rotate az
                tan_og_t[0] = np.cos(az) * x_temp + np.sin(az) * z_temp
                tan_og_t[1] = y_temp
                tan_og_t[2] = -np.sin(az) * x_temp + np.cos(az) * z_temp
                ##################################################

                alpha = tan_og_t[0]
                beta = tan_og_t[1]
                gamma = tan_og_t[2]

                def root_focal(t):
                    x = x_m2 + alpha * t
                    y = y_m2 + beta * t
                    z = z_m2 + gamma * t
                    xfoc, yfoc, zfoc = tele_into_foc(
                        x, y, z, el, az
                    )  # Convert ray's endpoint into M2 coordinates

                    z_foc = (
                        z_focal(xfoc, yfoc) - y_ap
                    )  # Z of focal plane in focal plane coordinates

                    root = zfoc - z_foc
                    return root

                t_foc = optimize.brentq(root_focal, 1e3, 15e3)

                # Location of where ray hits M2
                x_foc = x_m2 + alpha * t_foc
                y_foc = y_m2 + beta * t_foc
                z_foc = z_m2 + gamma * t_foc
                pos_ap = [x_foc, y_foc, z_foc]

                total_path_length = t_ap + t_m1 + t_m2 + t_foc

                #                 if np.mod(ii,21)==0:
                #                     plt.plot([y_m1,P_a[1]],[z_m1,P_a[2]],color = color,alpha = alph)
                #                     plt.plot([y_a,y_0],[z_a,z_0],color = color,alpha = alph)
                #                     plt.plot([y_m2,y_m1],[z_m2,z_m1],color = color,alpha = alph)
                #                     plt.plot([y_m2,pos_ap[1]],[z_m2,pos_ap[2]],color = color,alpha = alph)
                #                     plt.plot([y_m2,y_foc],[z_m2,z_foc],color = color,alpha = alph)
                #                 if np.mod(ii,71)==0 :
                #                     plt.plot([x_m2,x_m1],[y_m2,y_m1],color = color,alpha = alph)
                #                     plt.plot([x_m2,x_foc],[y_m2,y_foc],color = color,alpha = alph)
                #                     plt.plot([y_m2,y_foc],[z_m2,z_foc],color = color,alpha = alph)
                #                     plt.plot([y_m2,y_m1],[z_m2,z_m1],color = color,alpha = alph)

                # Write out
                out[0, ii] = x_m2
                out[1, ii] = y_m2
                out[2, ii] = z_m2

                out[3, ii] = x_m1
                out[4, ii] = y_m1
                out[5, ii] = z_m1

                out[6, ii] = N_hat_t[0]
                out[7, ii] = N_hat_t[1]
                out[8, ii] = N_hat_t[2]

                out[9, ii] = (
                    pos_ap[0] - rxmirror[6, 0]
                )  # need to re-center this on focal plane
                out[10, ii] = pos_ap[1] - rxmirror[7, 0]
                out[11, ii] = pos_ap[2] - rxmirror[8, 0]

                out[12, ii] = tan_og_t[0]
                out[13, ii] = tan_og_t[1]
                out[14, ii] = tan_og_t[2]

                out[15, ii] = total_path_length
                out[16, ii] = 1
    return out
