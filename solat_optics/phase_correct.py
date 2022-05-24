"""
Corrections and modifications of phase measurements.
"""
import numpy as np
from scipy.optimize import minimize


def twodunwrapx1(array):
    """
    Unwrap 2D phase array.
    """
    phase_offset = np.arange(-1000, 1000) * 2 * np.pi

    end = len(array)

    i = int(end / 2)
    while i < end:
        j = int(end / 2)
        while j < (np.shape(array))[1] - 1:
            current_val = [array[i, j]]
            next_val = array[i, j + 1] + phase_offset
            diff = np.abs(next_val - current_val)
            best = np.where(diff == np.min(diff))
            array[i, j + 1] = next_val[best][0]

            j += 1
        i += 1

    i = int(end / 2)
    while i > 0:
        j = int(end / 2)
        while j > 0:
            current_val = [array[i, j]]
            next_val = array[i, j - 1] + phase_offset
            diff = np.abs(next_val - current_val)
            best = np.where(diff == np.min(diff))
            array[i, j - 1] = next_val[best][0]

            j -= 1
        i -= 1

    i = int(end / 2)
    while i < end - 1:
        j = int(end / 2)
        while j > 0:
            current_val = [array[i, j]]
            next_val = array[i, j - 1] + phase_offset
            diff = np.abs(next_val - current_val)
            best = np.where(diff == np.min(diff))
            array[i, j - 1] = next_val[best][0]

            j -= 1
        i += 1

    i = int(end / 2)
    while i > 0:
        j = int(end / 2)
        while j < end - 1:
            current_val = [array[i, j]]
            next_val = array[i, j + 1] + phase_offset
            diff = np.abs(next_val - current_val)
            best = np.where(diff == np.min(diff))
            array[i, j + 1] = next_val[best][0]

            j += 1
        i -= 1

    return array


def twodunwrap(array):
    """
    Call unwrap function after transposing phase array,
    then transposing again and unwrapping.
    """
    xunwraped = twodunwrapx1(np.transpose(array))
    unwrapped = twodunwrapx1(np.transpose(xunwraped))
    return unwrapped


def do_unwrap(phase_input):
    """
    Call this function to unwrap the 2D phase map.
    Make sure input phase is in degrees and not radians!
    """
    phi = phase_input * (np.pi / 180)  # convert to radians
    unwraped_phi = twodunwrap(phi)
    unwraped_phi = unwraped_phi - unwraped_phi[0, 0]
    unwraped_phi = unwraped_phi * (180 / np.pi)  # convert back to degrees
    return unwraped_phi


def defocus(amp, const, x_0, y_0, x_arr, y_arr):
    """
    Modelling a defocus phase term.
    """
    radius = np.sqrt((x_arr - x_0) ** 2 + (y_arr - y_0) ** 2)
    return amp * (radius ** 2) + const


# def gradient_mod(grad_x,grad_y, x_0, y_0, x, y):
#     '''
#     Modelling a gradient phase term.
#     '''
#     return A * (x - x_0) + B * (y - y_0)


# def rotate_beam(phi, beam_co, beam_cr):
#     '''
#     Optional function. This function takes cross and co-polar
#     beam measurements and tilts by an angle to correct for cross-pol
#     leakage.
#     '''
#     ok = np.where(X ** 2 + Y ** 2 <= 80 ** 2)
#     beamco = np.cos(phi) * (beam_co[ok].real + beam_co[ok].imag) + np.sin(phi) * (
#         beam_cr[ok].real + beam_cr[ok].imag
#     )
#     beamcr = -np.sin(phi) * (beam_co[ok].real + beam_co[ok].imag) + np.cos(phi) * (
#         beam_cr[ok].real + beam_cr[ok].imag
#     )

#     return np.sqrt(beamcr ** 2).sum()


def phase_model_terms(p_guess, x_arr, y_arr, fre):
    """
    Fit this function to measured phase to determine
    8 parameters of model.
    amp = distance from source,
    grad_x & grad_y = gradient terms,
    D = constant offset,
    + 4 centering terms.
    """
    amp = p_guess[0]
    grad_x = p_guess[1]
    grad_y = p_guess[2]
    const = p_guess[3]
    x_0 = p_guess[4]
    y_0 = p_guess[5]
    x_1 = p_guess[6]
    y_1 = p_guess[7]

    lam = (3 * 10 ** 8) / (fre * 1e9)

    radius = np.sqrt((x_arr - x_0) ** 2 + (y_arr - y_0) ** 2)
    amp_phi = amp * (2 * np.pi) / lam
    spherical = amp_phi * (1 - np.cos(radius))
    gradient = grad_x * (x_arr - x_1) + grad_y * (y_arr - y_1)

    return spherical + gradient + const


# def phase_fit(p, X, Y, phase, fre):

#     rad_fit = 4  # deg
#     phase_mod = phase_model_terms(p, X, Y, fre)
#     phase_mod = np.where(X ** 2 + Y ** 2 < (rad_fit * np.pi / 180) ** 2, phase_mod, 0)

#     return np.sum(np.sqrt((phase_mod + phase) ** 2))


def beam_center_new(p_guess, x_arr, y_arr, beam, rad_bndry):
    """
    Function which is fit to data to
    determine centering offset of measurement.
    """
    x_0 = p_guess[0]
    y_0 = p_guess[1]

    radius = np.sqrt((x_arr - x_0) ** 2 + (y_arr - y_0) ** 2)

    power_frac = np.sum(beam[np.where(radius >= rad_bndry)]) / np.sum(beam)

    return power_frac


def beam_centering(theta_x, theta_y, phase_old, beam_old, rad):
    """
    Fits beam amplitude to center (peak finder) and uses
    np.roll to shift array until peak is centered. The
    phase is shifted by the same amount.
    """
    p_guess = [0, 0]
    vals = minimize(
        beam_center_new,
        p_guess,
        args=(theta_x * 180 / np.pi, theta_y * 180 / np.pi, beam_old ** 2, rad),
        options={"eps": 0.1},
    )

    index_x = np.searchsorted(theta_x[0, :], vals.x[0] * np.pi / 180)
    index_y = np.searchsorted(theta_y[:, index_x], vals.x[1] * np.pi / 180)

    center_index = int((len(theta_x) - 1) / 2)

    beam_new = np.roll(beam_old, (center_index - index_x), axis=1)
    beam_new = np.roll(beam_new, (center_index - index_y), axis=0)

    phase_new = np.roll(phase_old, (center_index - index_x), axis=1)
    phase_new = np.roll(phase_new, (center_index - index_y), axis=0)

    return phase_new, beam_new


def center_beam(X, Y, beam, ZZ):
    p0 = [0, 0, 10, 10]

    mid = np.where(X ** 2 + Y ** 2 < 20 ** 2)
    beam = abs(beam) / np.max(abs(beam))
    xx = minimize(gauss_min, p0, args=(X[mid], Y[mid], abs(beam)[mid]))
    index_x = np.searchsorted(X[0, :], xx.x[0])
    index_y = np.searchsorted(Y[:, 0], xx.x[1])

    xshift = int((len(X) / 2) - index_x)
    yshift = int((len(Y) / 2) - index_y)

    ZZ = np.roll(np.roll(ZZ, yshift, axis=0), xshift, axis=1)
    beam = np.roll(np.roll(beam, yshift, axis=0), xshift, axis=1)
    return ZZ, beam


def gauss(X, Y, p):
    x0 = p[0]
    y0 = p[1]
    sig_a = p[2]
    sig_b = p[3]
    a = -((X - x0) ** 2) / sig_a
    b = -((Y - y0) ** 2) / sig_b
    return np.exp(a + b)


def gauss_min(p, X, Y, data):

    model = gauss(X, Y, p)

    return np.sum(np.sqrt((model - data) ** 2))


def center_beam(X, Y, beam, ZZ):
    p0 = [0, 0, 10, 10]

    mid = np.where(X ** 2 + Y ** 2 < 20 ** 2)
    beam = abs(beam) / np.max(abs(beam))
    xx = minimize(gauss_min, p0, args=(X[mid], Y[mid], abs(beam)[mid]))
    index_x = np.searchsorted(X[0, :], xx.x[0])
    index_y = np.searchsorted(Y[:, 0], xx.x[1])

    xshift = int((len(X) / 2) - index_x)
    yshift = int((len(Y) / 2) - index_y)

    ZZ = np.roll(np.roll(ZZ, yshift, axis=0), xshift, axis=1)
    beam = np.roll(np.roll(beam, yshift, axis=0), xshift, axis=1)
    return ZZ, beam

