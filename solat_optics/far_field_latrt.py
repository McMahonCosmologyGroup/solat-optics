"""
Far field simulation.

Grace E. Chesmore
May 2021
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

path_to_package = "/home/chesmore/Desktop/Code/solat-optics"
sys.path.append(path_to_package) 
import solat_optics.solat_apert_field as solat_apert_field

sys.path.append("/home/chesmore/Desktop/Code/holosim_paper/package/holosim-ml")
import pan_mod as pm


def project_to_data(sim, data_x, data_y, tele_geo):
    cc = np.where(sim[15, :] != 0)
    sim_new = np.zeros(np.shape(data_x), dtype=complex)
    x_sim = sim[9, :][cc] / 1e1  # convert to cm
    y_sim = sim[11, :][cc] / 1e1  # convert to cm

    for ii in range(len(x_sim)):
        xx = x_sim[ii]
        yy = y_sim[ii]

        if np.max(data_x[0, :]) < np.max(xx) or np.max(data_y[:, 0]) < np.max(yy):
#             if xx == np.min(xx):
#                 print(np.max(data_x[0, :]), abs(xx))
            # skipping over points in the grid that are outside of the measurement bounds
            continue
        else:
            index_x = np.searchsorted(data_x[0, :], xx)
            index_y = np.searchsorted(data_y[:, index_x], yy)

            sim_new[ index_y,index_x] = np.exp(
                np.complex(0, 1) * tele_geo.k * sim[15, :][cc][ii] / 1e3
            )

    return sim_new