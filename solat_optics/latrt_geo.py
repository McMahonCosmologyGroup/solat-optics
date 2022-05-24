"""
Large aperture telescope mirror definitions.
"""
import numpy as np


class initialize_telescope_geometry:
    """
    Initialize telescope parameters.
    """

    F_2 = 7000
    th_1 = np.arctan(1 / 2)  # Primary mirror tilt angle
    th_2 = np.arctan(1 / 3)  # Secondary mirror tilt angle
    th2 = (-np.pi / 2) - th_2
    th_fwhp = 35 * np.pi / 180  # Full width half power [rad]
    N_scan = 100  # Pixels in 1D of grid
    de_ang = 2 / 60 * np.pi / 180  # Far-field angle increment, arcmin = 1/60 degree
    lambda_ = (30.0 / 150.0) * 0.01  # Source wavelength [m]
    k = 2 * np.pi / lambda_  # Wavenumber [1/m]

    # Receiver feed position [um]
    rx_x = 0
    rx_y = 0
    rx_z = 0

    # Phase reference [m]
    x_phref = 0
    y_phref = -7.2
    z_phref = 0

    # Center of rotation [m]
    x_rotc = 0
    y_rotc = -7.2
    z_rotc = 0

    # Source position (tower) [m]
    x_tow = 0
    y_tow = -7.2
    z_tow = 1e3

    # Azimuth and Elevation center [rad]
    az0 = 0  # np.arctan(-x_tow / z_tow)
    el0 = 0  # np.arctan(-y_tow / z_tow)

    # Aperture plane [m]
    x_ap = 3.0
    y_ap = -7.2
    z_ap = 4.0


# Matrix Coefficients defining mirror surfaces
# Primary Mirror
a1 = np.zeros((7, 7))
a1[0, :] = [0, 0, -57.74022, 1.5373825, 1.154294, -0.441762, 0.0906601]
a1[1, :] = [0, 0, 0, 0, 0, 0, 0]
a1[2, :] = [-72.17349, 1.8691899, 2.8859421, -1.026471, 0.2610568, 0, 0]
a1[3, :] = [0, 0, 0, 0, 0, 0, 0]
a1[4, :] = [1.8083973, -0.603195, 0.2177414, 0, 0, 0, 0]
a1[5, :] = [0, 0, 0, 0, 0, 0, 0]
a1[6, :] = [0.0394559, 0, 0, 0, 0, 0, 0]
# Secondary Mirror
a2 = np.zeros((8, 8))
a2[0, :] = [0, 0, 103.90461, 6.6513025, 2.8405781, -0.7819705, -0.0400483, 0.0896645]
a2[1, :] = [0, 0, 0, 0, 0, 0, 0, 0]
a2[2, :] = [115.44758, 7.3024355, 5.7640389, -1.578144, -0.0354326, 0.2781226, 0, 0]
a2[3, :] = [0, 0, 0, 0, 0, 0, 0, 0]
a2[4, :] = [2.9130983, -0.8104051, -0.0185283, 0.2626023, 0, 0, 0, 0]
a2[5, :] = [0, 0, 0, 0, 0, 0, 0, 0]
a2[6, :] = [-0.0250794, 0.0709672, 0, 0, 0, 0, 0, 0]
a2[7, :] = [0, 0, 0, 0, 0, 0, 0, 0]

R_N = 3000  # [mm]

# These functions define the mirror surfaces,
# and the normal vectors on the surfaces.
def z1(x_arr, y_arr):
    """
    Primary mirror surface.
    """
    amp = 0
    for i_indx in range(7):
        for j_indx in range(7):
            amp += (
                a1[i_indx, j_indx]
                * ((x_arr / R_N) ** i_indx)
                * ((y_arr / R_N) ** j_indx)
            )
    return amp


def z2(x_arr, y_arr):
    """
    Secondary mirror surface.
    """
    amp = 0
    for i_indx in range(8):
        for j_indx in range(8):
            amp += (
                a2[i_indx, j_indx]
                * ((x_arr / R_N) ** i_indx)
                * ((y_arr / R_N) ** j_indx)
            )
    return amp


def d_z1(x_arr, y_arr):
    """
    Primary mirror normal vector to surface.
    """
    amp_x = 0
    amp_y = 0
    for i_indx in range(7):
        for j_indx in range(7):
            amp_x += (
                a1[i_indx, j_indx]
                * (i_indx / R_N)
                * ((x_arr / R_N) ** (i_indx - 1))
                * ((y_arr / R_N) ** j_indx)
            )
            amp_y += (
                a1[i_indx, j_indx]
                * ((x_arr / R_N) ** i_indx)
                * (j_indx / R_N)
                * ((y_arr / R_N) ** (j_indx - 1))
            )
    return amp_x, amp_y


def d_z2(x_arr, y_arr):
    """
    Secondary mirror normal vector to surface.
    """
    amp_x = 0
    amp_y = 0
    for i_indx in range(8):
        for j_indx in range(8):
            amp_x += (
                a2[i_indx, j_indx]
                * (i_indx / R_N)
                * ((x_arr / R_N) ** (i_indx - 1))
                * ((y_arr / R_N) ** j_indx)
            )
            amp_y += (
                a2[i_indx, j_indx]
                * ((x_arr / R_N) ** i_indx)
                * (j_indx / R_N)
                * ((y_arr / R_N) ** (j_indx - 1))
            )
    return amp_x, amp_y


def m1_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from primary into telescope.
    """
    th1 = initialize_telescope_geometry.th_1
    x_temp = x_arr * np.cos(np.pi) + z_arr * np.sin(np.pi)
    y_temp = y_arr
    z_temp = -x_arr * np.sin(np.pi) + z_arr * np.cos(np.pi)

    x_rot1 = x_temp
    y_rot1 = y_temp * np.cos(th1) - z_temp * np.sin(th1) - 7200
    z_rot1 = (y_temp * np.sin(th1) + z_temp * np.cos(th1)) - 3600
    return x_rot1, y_rot1, z_rot1


def m2_into_tele(x_arr, y_arr, z_arr):
    """
    Coordinate transformation from secondary into telescope.
    """
    th2 = initialize_telescope_geometry.th2
    x_temp = x_arr * np.cos(np.pi) - y_arr * np.sin(np.pi)
    y_temp = x_arr * np.sin(np.pi) + y_arr * np.cos(np.pi)
    z_temp = z_arr

    x_rot2 = x_temp
    y_rot2 = (y_temp * np.cos(th2) - z_temp * np.sin(th2)) - 4800 - 7200
    z_rot2 = y_temp * np.sin(th2) + z_temp * np.cos(th2)
    return x_rot2, y_rot2, z_rot2


def tele_into_m1(x_arr, y_arr, z_arr, el_, az_):
    """
    Coordinate transformation from telescope into primary.
    """
    x_arr, y_arr, z_arr = rotate_az_el(x_arr, y_arr, z_arr, el_, az_)

    th1 = initialize_telescope_geometry.th_1
    z_arr += 3600
    y_arr += 7200

    x_temp = x_arr
    y_temp = y_arr * np.cos(-th1) - z_arr * np.sin(-th1)
    z_temp = y_arr * np.sin(-th1) + z_arr * np.cos(-th1)

    x_2 = x_temp * np.cos(np.pi) + z_temp * np.sin(np.pi)
    y_2 = y_temp
    z_2 = -x_temp * np.sin(np.pi) + z_temp * np.cos(np.pi)

    return x_2, y_2, z_2


def tele_into_m2(x_arr, y_arr, z_arr, el_, az_):
    """
    Coordinate transformation from telescope into secondary.
    """
    x_arr, y_arr, z_arr = rotate_az_el(x_arr, y_arr, z_arr, el_, az_)

    th2 = initialize_telescope_geometry.th2
    y_arr += 4800 + 7200
    x_temp = x_arr
    y_temp = y_arr * np.cos(-th2) - z_arr * np.sin(-th2)
    z_temp = y_arr * np.sin(-th2) + z_arr * np.cos(-th2)

    x_2 = x_temp * np.cos(-np.pi) - y_temp * np.sin(-np.pi)
    y_2 = x_temp * np.sin(-np.pi) + y_temp * np.cos(-np.pi)
    z_2 = z_temp

    return x_2, y_2, z_2


def z_ap(x_arr, y_arr):
    """
    Aperture plane surface.
    """
    z_arr = 0 * x_arr * y_arr
    return z_arr


def z_focal(x_arr, y_arr):
    """
    Focal plane surface.
    """
    z_arr = 0 * x_arr * y_arr
    return z_arr


def tele_into_apert(x_arr, y_arr, z_arr, el_, az_):
    """
    Coordinate transformation from telescope into aperture plane.
    """
    x_arr, y_arr, z_arr = rotate_az_el(x_arr, y_arr, z_arr, el_, az_)  # rotate
    z_arr -= 4e3  # go to apert ref frame
    return x_arr, y_arr, z_arr


def foc_into_tele(x_arr, y_arr, z_arr, el_, az_):
    """
    Coordinate transformation from focal plane into telescope.
    """
    x_out = x_arr
    y_out = y_arr * np.cos(np.pi / 2) - z_arr * np.sin(np.pi / 2)
    z_out = y_arr * np.sin(np.pi / 2) + z_arr * np.cos(np.pi / 2)
    x_out, y_out, z_out = rotate_az_el(x_out, y_out, z_out, el_, az_)
    return x_out, y_out, z_out


def tele_into_foc(x_arr, y_arr, z_arr, el_, az_):
    """
    Coordinate transformation from telescope into focal plane.
    """
    x_arr, y_arr, z_arr = rotate_az_el(x_arr, y_arr, z_arr, el_, az_)

    x_out = x_arr
    y_out = y_arr * np.cos(-np.pi / 2) - z_arr * np.sin(-np.pi / 2)
    z_out = y_arr * np.sin(-np.pi / 2) + z_arr * np.cos(-np.pi / 2)

    return x_out, y_out, z_out


def rotate_az_el(x_arr, y_arr, z_arr, el_, az_):
    """
    Rotate coordinates by given elevation and azimuth.
    """
    x_arr -= initialize_telescope_geometry.x_rotc * 1e3
    y_arr -= initialize_telescope_geometry.y_rotc * 1e3
    z_arr -= initialize_telescope_geometry.z_rotc * 1e3

    # rotate el
    x_temp = x_arr
    y_temp = np.cos(el_) * y_arr - np.sin(el_) * z_arr
    z_temp = np.sin(el_) * y_arr + np.cos(el_) * z_arr

    # rotate az
    x_arr = np.cos(az_) * x_temp + np.sin(az_) * z_temp
    y_arr = y_temp
    z_arr = -np.sin(az_) * x_temp + np.cos(az_) * z_temp

    x_arr += initialize_telescope_geometry.x_rotc * 1e3
    y_arr += initialize_telescope_geometry.y_rotc * 1e3
    z_arr += initialize_telescope_geometry.z_rotc * 1e3

    return x_arr, y_arr, z_arr
