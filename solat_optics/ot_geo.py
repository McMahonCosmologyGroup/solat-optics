"""
Optics tube geometry definitions including surfaces of lenses,
and an example for coding a binary filter from a jpeg.
Grace E. Chesmore
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image, ImageOps


class LatGeo:
    """
    LAT geometry settings.
    """

    def __init__(self):

        self.n_si = 3.4
        self.n_vac = 1.0

        self.th_fwhp = 35 * np.pi / 180
        self.n_scan = 100

        self.de_ang = 0.5 / 60 * np.pi / 180  # arcsec_val= 1/60 degree
        self.lambda_ = (30.0 / 150.0) * 0.01  # [m]
        self.k = 2 * np.pi / self.lambda_

        self.lyot_y = 453.62

        self.x_ap = 0
        self.y_ap = 0
        self.z_ap = 0

    def set_fwhp(self, fwhp):
        """
        Half width of sweep in 1D
        """
        self.th_fwhp = fwhp

    def set_nscan(self, n_val):
        """
        Half width of sweep in 1D
        """
        self.n_scan = n_val

    def set_res(self, res):
        """
        Half width of sweep in 1D
        """
        self.de_ang = res

    def set_wavelength(self, wavelength):
        """
        Half width of sweep in 1D
        """
        self.lambda_ = wavelength
        self.k = 2 * np.pi / wavelength


th1_l1 = np.pi / 2
th2_l1 = -np.pi / 2
th1_l2 = -np.pi / 2
th2_l2 = -np.pi / 2
th1_l3 = -np.pi / 2
th2_l3 = -np.pi / 2

LENS1_Y = 975.08
LENS2_Y = 592.76
LENS3_Y = 180.8 - ((2.5 + 1.513) - (35.16 - 32.19))
LYOT_Y = 453.62


def z1b():
    """
    Surface shape of lens 1 side B.
    """
    return 0


def z1a(x_arr, y_arr):
    """
    Surface shape of lens 1 side A.
    """
    r_val = -1510.61
    c_val = 1 / r_val
    k = 12.0966
    a_1 = -5.876941e-5
    a_2 = 1.769198e-9
    a_3 = -1.310183e-15
    a_4 = 1.994766e-19
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)
    if 1 == 0:  # 1-((1+k)*c**2 * r**2) <0:
        amp = 0
    else:
        amp = (c_val * r_arr ** 2) / (
            1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))
        )
        amp += a_1 * r_arr ** 2 + a_2 * r_arr ** 4 + a_3 * r_arr ** 6 + a_4 * r_arr ** 8
    return amp


def d_z1b():
    """
    Normal vector of surface of lens 1 side B.
    """
    amp_x = 0
    amp_y = 0
    return amp_x, amp_y


def d_z1a(x_arr, y_arr):
    """
    Normal vector of surface of lens 1 side A.
    """
    r_val = -1510.61
    c_val = 1 / r_val
    k = 12.0966
    a_1 = -5.876941e-5
    a_2 = 1.769198e-9
    a_3 = -1.310183e-15
    a_4 = 1.994766e-19
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)

    coeff_1 = (
        (a_1 * 2)
        + (a_2 * 4 * r_arr ** 2)
        + (a_3 * 6 * r_arr ** 4)
        + (a_4 * 8 * r_arr ** 6)
    )
    coeff_2 = (c_val * 2) / (1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2)))
    coeff_3 = (c_val ** 3 * (k + 1) * r_arr ** 2) / (
        np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))
        * (1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))) ** 2
    )
    amp_x = x_arr * (coeff_1 + coeff_2 + coeff_3)
    amp_y = y_arr * (coeff_1 + coeff_2 + coeff_3)

    return amp_x, amp_y


def m1b_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transform from lens 1 side B into telescope reference frame.
    """
    x_rot1 = x_arr
    y_rot1 = y_arr * np.cos(th1_l1) - z_arr * np.sin(th1_l1) - LENS1_Y
    z_rot1 = y_arr * np.sin(th1_l1) + z_arr * np.cos(th1_l1)
    return x_rot1, y_rot1, z_rot1


def m1a_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transform from lens 1 side A into telescope reference frame.
    """
    x_rot2 = x_arr
    y_rot2 = (y_arr * np.cos(th2_l1) - z_arr * np.sin(th2_l1)) - (LENS1_Y - 19.32)
    z_rot2 = y_arr * np.sin(th2_l1) + z_arr * np.cos(th2_l1)
    return x_rot2, y_rot2, z_rot2


def tele_into_m1b(x_arr, y_arr, z_arr):
    """
    Coordinate transform from telescope reference frame into lens 1 side B reference frame.
    """
    y_arr += LENS1_Y
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th1_l1) - z_arr * np.sin(-th1_l1)
    z_temp = y_arr * np.sin(-th1_l1) + z_arr * np.cos(-th1_l1)

    return x_temp, y_temp, z_temp


def tele_into_m1a(x_arr, y_arr, z_arr):
    """
    Coordinate transform from telescope reference frame into lens 1 side  reference frame.
    """
    y_arr += LENS1_Y - 19.32
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th2_l1) - z_arr * np.sin(-th2_l1)
    z_temp = y_arr * np.sin(-th2_l1) + z_arr * np.cos(-th2_l1)

    return x_temp, y_temp, z_temp


def z2b():
    """
    Surface shape of lens 2 side B.
    """
    return 0


def z2a(x_arr, y_arr):
    """
    Surface shape of lens 2 side A.
    """
    r_val = -871.25
    c_val = 1 / r_val
    k = -10.201256
    a_1 = -1.147640e-04
    a_2 = 1.103643e-10
    a_3 = -2.969608e-14
    a_4 = 5.980271e-19
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)
    if 1 == 0:  # 1-((1+k)*c**2 * r**2) <0:
        amp = 0
    else:
        amp = (c_val * r_arr ** 2) / (
            1 + np.sqrt(1 - (1 + k) * c_val ** 2 * r_arr ** 2)
        )
        amp += a_1 * r_arr ** 2 + a_2 * r_arr ** 4 + a_3 * r_arr ** 6 + a_4 * r_arr ** 8
    return amp


def d_z2b():
    """
    Normal vector to surface lens 2 side B.
    """
    amp_x = 0
    amp_y = 0
    return amp_x, amp_y


def d_z2a(x_arr, y_arr):
    """
    Normal vector to surface lens 2 side A.
    """
    r_val = -871.25
    c_val = 1 / r_val
    k = -10.201256
    a_1 = -1.147640e-04
    a_2 = 1.103643e-10
    a_3 = -2.969608e-14
    a_4 = 5.980271e-19
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)

    coeff_1 = (
        (a_1 * 2)
        + (a_2 * 4 * r_arr ** 2)
        + (a_3 * 6 * r_arr ** 4)
        + (a_4 * 8 * r_arr ** 6)
    )
    coeff_2 = (c_val * 2) / (1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2)))
    coeff_3 = (c_val ** 3 * (k + 1) * r_arr ** 2) / (
        np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))
        * (1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))) ** 2
    )
    amp_x = x_arr * (coeff_1 + coeff_2 + coeff_3)
    amp_y = y_arr * (coeff_1 + coeff_2 + coeff_3)
    return amp_x, amp_y


def m2b_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from lens 2 side B into telescope reference frame.
    """
    x_rot1 = x_arr
    y_rot1 = y_arr * np.cos(th1_l2) - z_arr * np.sin(th1_l2) - LENS2_Y
    z_rot1 = y_arr * np.sin(th1_l2) + z_arr * np.cos(th1_l2)
    return x_rot1, y_rot1, z_rot1


def m2a_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from lens 2 side A into telescope reference frame.
    """
    x_rot2 = x_arr
    y_rot2 = (y_arr * np.cos(th2_l2) - z_arr * np.sin(th2_l2)) - (LENS2_Y - 26.47)
    z_rot2 = y_arr * np.sin(th2_l2) + z_arr * np.cos(th2_l2)
    return x_rot2, y_rot2, z_rot2


def tele_into_m2b(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from telescope into lens 2 side B reference frame.
    """
    y_arr += LENS2_Y
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th1_l2) - z_arr * np.sin(-th1_l2)
    z_temp = y_arr * np.sin(-th1_l2) + z_arr * np.cos(-th1_l2)

    return x_temp, y_temp, z_temp


def tele_into_m2a(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from telescope into lens 2 side A reference frame.
    """
    y_arr += LENS2_Y - 26.47
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th2_l1) - z_arr * np.sin(-th2_l1)
    z_temp = y_arr * np.sin(-th2_l1) + z_arr * np.cos(-th2_l1)

    return x_temp, y_temp, z_temp


def z3a(x_arr, y_arr):
    """
    Surface shape of lens 3 side A.
    """
    r_val = 6608.30
    c_val = 1 / r_val
    k = 414.142695
    a_1 = -2.981594e-06
    a_2 = 3.955202e-10
    a_3 = 3.180727e-16
    a_4 = 0
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)
    if 1 == 0:
        amp = 0
    else:
        amp = (c_val * r_arr ** 2) / (
            1 + np.sqrt(1 - (1 + k) * c_val ** 2 * r_arr ** 2)
        )
        amp += a_1 * r_arr ** 2 + a_2 * r_arr ** 4 + a_3 * r_arr ** 6 + a_4 * r_arr ** 8
    return amp


def z3b(x_arr, y_arr):
    """
    Surface shape of lens 3 side B.
    """
    r_val = 604.44
    c_val = 1 / r_val
    k = -3.177341
    a_1 = 2.366829e-05
    a_2 = -1.000417e-10
    a_3 = -6.565681e-15
    a_4 = 0
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)
    if 1 == 0:
        amp = 0
    else:
        amp = (c_val * r_arr ** 2) / (
            1 + np.sqrt(1 - (1 + k) * c_val ** 2 * r_arr ** 2)
        )
        amp += a_1 * r_arr ** 2 + a_2 * r_arr ** 4 + a_3 * r_arr ** 6 + a_4 * r_arr ** 8
    return amp


def filt():
    """
    Surface shape of filter.
    """
    return 0


def d_z3a(x_arr, y_arr):
    """
    Normal vector on surface of lens 3 side A.
    """
    r_val = 6608.30
    c_val = 1 / r_val
    k = 414.142695
    a_1 = -2.981594e-06
    a_2 = 3.955202e-10
    a_3 = 3.180727e-16
    a_4 = 0
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)

    coeff_1 = (
        (a_1 * 2)
        + (a_2 * 4 * r_arr ** 2)
        + (a_3 * 6 * r_arr ** 4)
        + (a_4 * 8 * r_arr ** 6)
    )
    coeff_2 = (c_val * 2) / (1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2)))
    coeff_3 = (c_val ** 3 * (k + 1) * r_arr ** 2) / (
        np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))
        * (1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))) ** 2
    )
    amp_x = x_arr * (coeff_1 + coeff_2 + coeff_3)
    amp_y = y_arr * (coeff_1 + coeff_2 + coeff_3)

    return amp_x, amp_y


def d_z3b(x_arr, y_arr):
    """
    Normal vector on surface of lens 3 side B.
    """
    r_val = 604.44
    c_val = 1 / r_val
    k = -3.177341
    a_1 = 2.366829e-05
    a_2 = -1.000417e-10
    a_3 = -6.565681e-15
    a_4 = 0
    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)

    coeff_1 = (
        (a_1 * 2)
        + (a_2 * 4 * r_arr ** 2)
        + (a_3 * 6 * r_arr ** 4)
        + (a_4 * 8 * r_arr ** 6)
    )
    coeff_2 = (c_val * 2) / (1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2)))
    coeff_3 = (c_val ** 3 * (k + 1) * r_arr ** 2) / (
        np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))
        * (1 + np.sqrt(1 - ((1 + k) * c_val ** 2 * r_arr ** 2))) ** 2
    )
    amp_x = x_arr * (coeff_1 + coeff_2 + coeff_3)
    amp_y = y_arr * (coeff_1 + coeff_2 + coeff_3)

    return amp_x, amp_y


def m3a_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from lens 3 side A to telescope.
    """
    x_rot1 = x_arr
    y_rot1 = y_arr * np.cos(th1_l3) - z_arr * np.sin(th1_l3) - LENS3_Y
    z_rot1 = y_arr * np.sin(th1_l3) + z_arr * np.cos(th1_l3)
    return x_rot1, y_rot1, z_rot1


def m3b_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from lens 3 side B to telescope.
    """
    x_rot2 = x_arr
    y_rot2 = (y_arr * np.cos(th2_l3) - z_arr * np.sin(th2_l3)) - (LENS3_Y + 35.16)
    z_rot2 = y_arr * np.sin(th2_l3) + z_arr * np.cos(th2_l3)
    return x_rot2, y_rot2, z_rot2


def fp_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from lens 3 side B to telescope.
    """
    x_rot2 = x_arr
    y_rot2 = y_arr * np.cos(th2_l3) - z_arr * np.sin(th2_l3)
    z_rot2 = y_arr * np.sin(th2_l3) + z_arr * np.cos(th2_l3)
    return x_rot2, y_rot2, z_rot2


def tele_into_f3(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from telescope to filter 3.
    """
    y_arr += 18.10815
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th1_l3) - z_arr * np.sin(-th1_l3)
    z_temp = y_arr * np.sin(-th1_l3) + z_arr * np.cos(-th1_l3)

    return x_temp, y_temp, z_temp


def tele_into_f1(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from telescope to filter 1.
    """
    y_arr += 991.84174
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th1_l3) - z_arr * np.sin(-th1_l3)
    z_temp = y_arr * np.sin(-th1_l3) + z_arr * np.cos(-th1_l3)

    return x_temp, y_temp, z_temp


def tele_into_f2(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from telescope to filter 2.
    """
    y_arr += 608.56983
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th1_l3) - z_arr * np.sin(-th1_l3)
    z_temp = y_arr * np.sin(-th1_l3) + z_arr * np.cos(-th1_l3)

    return x_temp, y_temp, z_temp


def tele_into_m3a(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from telescope to lens 3, side A.
    """
    y_arr += LENS3_Y
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th1_l3) - z_arr * np.sin(-th1_l3)
    z_temp = y_arr * np.sin(-th1_l3) + z_arr * np.cos(-th1_l3)

    return x_temp, y_temp, z_temp


def tele_into_m3b(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from telescope to lens 3, side B.
    """
    y_arr += LENS3_Y + 35.16
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th2_l3) - z_arr * np.sin(-th2_l3)
    z_temp = y_arr * np.sin(-th2_l3) + z_arr * np.cos(-th2_l3)

    return x_temp, y_temp, z_temp


def z_lyot():
    """
    Surface shape of Lyot stop.
    """
    return 0


def d_zlyot():
    """
    Normal vector on surface of Lyot.
    """
    return 0, 0


def lyot_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transformation Lyot stop to telescope.
    """
    x_rot1 = x_arr
    y_rot1 = y_arr * np.cos(th1_l3) - z_arr * np.sin(th1_l3) - LYOT_Y
    z_rot1 = y_arr * np.sin(th1_l3) + z_arr * np.cos(th1_l3)
    return x_rot1, y_rot1, z_rot1


def tele_into_lyot(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from telescope to Lyot stop.
    """
    y_arr += LYOT_Y
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th1_l3) - z_arr * np.sin(-th1_l3)
    z_temp = y_arr * np.sin(-th1_l3) + z_arr * np.cos(-th1_l3)

    return x_temp, y_temp, z_temp


def plot_lenses():
    """
    Plots geometry of LATr_val OT lenses.
    """
    x_arr = np.linspace(-(392 / 2), (392 / 2), 100)  # [mm]
    y_arr = np.linspace(-(392 / 2), (392 / 2), 100)  # [mm]
    x_arr, y_arr = np.meshgrid(x_arr, y_arr)
    x_3 = np.linspace(-((374 - 8) / 2), ((374 - 8) / 2), 100)  # [mm]
    y_3 = np.linspace(-((374 - 8) / 2), ((374 - 8) / 2), 100)  # [mm]
    x_3, y_3 = np.meshgrid(x_3, y_3)

    x_2 = np.linspace(-(352 / 2), (352 / 2), 100)  # [mm]
    y_2 = np.linspace(-(352 / 2), (352 / 2), 100)  # [mm]
    x_2, y_2 = np.meshgrid(x_2, y_2)

    x_lyot = np.linspace(-136.54 / 2, 136.54 / 2, 100)  # [mm]
    y_lyot = np.linspace(-136.54 / 2, 136.54 / 2, 100)  # [mm]
    x_lyot, y_lyot = np.meshgrid(x_lyot, y_lyot)

    r_arr = np.sqrt(x_arr ** 2 + y_arr ** 2)
    r_2 = np.sqrt(x_2 ** 2 + y_2 ** 2)
    r_3 = np.sqrt(x_3 ** 2 + y_3 ** 2)

    z1_a = z1a(x_arr, y_arr)
    z1_b = z1b()
    z1_a = np.where(r_arr < 300, z1_a, np.nan)
    z1_b = np.where(r_arr < 300, z1_b, np.nan)

    z2_a = z2a(x_2, y_2)
    z2_b = z2b()
    z2_a = np.where(r_2 < 300, z2_a, np.nan)
    z2_b = np.where(r_2 < 300, z2_b, np.nan)

    z3_a = z3a(x_3, y_3)
    z3_b = z3b(x_3, y_3)
    z3_a = np.where(r_3 < 300, z3_a, np.nan)
    z3_b = np.where(r_3 < 300, z3_b, np.nan)

    xt1b, yt1b, zt1b = m1b_into_tele(x_arr, y_arr, z1_b)
    xt1a, yt1a, zt1a = m1a_into_tele(x_arr, y_arr, z1_a)

    xt2b, yt2b, zt2b = m2b_into_tele(y_2, y_2, z2_b)
    xt2a, yt2a, zt2a = m2a_into_tele(y_2, y_2, z2_a)

    xt3b, yt3b, zt3b = m3b_into_tele(x_3, y_3, z3_b)
    xt3a, yt3a, zt3a = m3a_into_tele(x_3, y_3, z3_a)

    plt.plot(
        yt1b[:, int(len(yt1a) / 2)],
        zt1b[:, int(len(yt1a) / 2)],
        "-",
        color="k",
    )  # label="B1b"
    # )
    plt.plot(
        yt1a[:, int(len(yt1a) / 2)],
        zt1a[:, int(len(yt1a) / 2)],
        "-",
        color="k",
    )  # label="B1a"
    # )
    plt.plot(
        yt2b[:, int(len(yt2a) / 2)],
        zt2b[:, int(len(yt2a) / 2)],
        "-",
        color="k",
    )  # label="B2b"
    # )
    plt.plot(
        yt2a[:, int(len(yt2a) / 2)],
        zt2a[:, int(len(yt2a) / 2)],
        "-",
        color="k",
    )  # label="B2a"
    # )
    plt.plot(
        yt3b[:, int(len(yt3a) / 2)],
        zt3b[:, int(len(yt3a) / 2)],
        "-",
        color="k",
    )  # label="B3b"
    # )
    plt.plot(
        yt3a[:, int(len(yt3a) / 2)],
        zt3a[:, int(len(yt3a) / 2)],
        "-",
        color="k",
    )  # label="B3a"


def filter_geo_new():
    """
    Defines binary array used as metal-mesh filter in ray trace of LATr_val OT.
    """

    image = Image.open("filter/so_filter2.png")
    scale = (np.shape(image)[1]) / (np.shape(image)[0])

    tot_range = 21 * 25.4 / 2
    x_arr = np.linspace(
        -(tot_range / 2) * scale, (tot_range / 2) * scale, (np.shape(image)[1])
    )
    y_arr = np.linspace(-(tot_range / 2), (tot_range / 2), (np.shape(image)[0]))
    x_arr, y_arr = np.meshgrid(x_arr, y_arr)

    shape_2d = (np.shape(image)[0], np.shape(image)[1])

    image = ImageOps.grayscale(image)
    image = np.array(image)

    out = np.zeros(np.shape(image))
    indx_ones = np.where(image < 150)
    out[indx_ones] = 1.0

    out = np.reshape(out, (shape_2d))

    im_high_circ = np.where(
        (abs(x_arr) <= (tot_range * 2)) & (abs(y_arr) <= (tot_range * 2)), out, 0
    )
    x_new = np.ravel(x_arr)
    y_new = np.ravel(y_arr)
    filt_arr = np.ravel(im_high_circ)

    sq_r = 125

    indx_circ = np.where((abs(x_new) < sq_r) & (abs(y_new) < sq_r))

    x_new = x_new[indx_circ]
    y_new = y_new[indx_circ]
    filt_circ = filt_arr[indx_circ]

    x_arr = np.reshape(
        x_new, (int(np.sqrt(np.shape(filt_circ))), int(np.sqrt(np.shape(filt_circ))))
    )
    y_arr = np.reshape(
        y_new, (int(np.sqrt(np.shape(filt_circ))), int(np.sqrt(np.shape(filt_circ))))
    )
    filt_2d = np.reshape(
        filt_circ,
        (int(np.sqrt(np.shape(filt_circ))), int(np.sqrt(np.shape(filt_circ)))),
    )

    y_arr = np.concatenate((y_arr, y_arr, y_arr), axis=1)
    x_arr = np.concatenate((x_arr - (sq_r * 2), x_arr, x_arr + (sq_r * 2)), axis=1)
    filt_mid = np.concatenate((filt_2d, filt_2d, filt_2d), axis=1)

    y_arr = np.concatenate((y_arr - (sq_r * 2), y_arr, y_arr + (sq_r * 2)), axis=0)
    x_arr = np.concatenate((x_arr, x_arr, x_arr), axis=0)
    filt_new = np.concatenate((filt_mid, filt_mid, filt_mid), axis=0)
    out = np.zeros((3, len(filt_new), len(filt_new)))

    filt_final = np.where(x_arr ** 2 + y_arr ** 2 <= (352 / 2) ** 2, filt_new, 0)
    zf_gauss = ndimage.gaussian_filter(filt_final, sigma=(2, 2), order=0)
    zf_gauss = np.where(zf_gauss > 0.9, zf_gauss, 0)
    zf_gauss = np.where(zf_gauss < 0.9, zf_gauss, 1)

    out[0, :, :] = x_arr
    out[1, :, :] = y_arr
    out[2, :, :] = zf_gauss

    return out
