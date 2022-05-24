"""
Parallelized holography simulation of the LATRt using geometric
theory of diffraction. Output is a .txt file with X,Y,amplitude and phase of beam.
Grace E. Chesmore, July 2021

To Run (from terminal): mpiexec -n 5 python3 holog_run.py frequency
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time

import numpy as np
from mpi4py import MPI
from tqdm import tqdm

import solat_optics
from solat_optics import ot_geo, ray_trace


# When calling ray-tracing functions, we do not want to output plots here.
class plotOpts:

    plot = 0
    if plot == 1:
        fig, ax = plt.subplots(figsize=(10, 5.5))
    color = "darkorange"
    alpha = 0.2


freq = float(sys.argv[1])
tele_geo = ot_geo.LatGeo()

# Im setting the y_ap (source's distance above the LATRt
# window. These values match the physical distances
# when I took the measurements in June 2021.)
if freq <= 120:
    tele_geo.y_ap = -1213 - (6.5 * 10)
    tele_geo.th_fwhp = 40.5 * np.pi / 180
else:
    tele_geo.y_ap = -1213 - (12.2 * 10)
    tele_geo.th_fwhp = 25 * np.pi / 180

RES = 51
# tele_geo.y_ap = -1213 - (215.9) # Carlos's thermal beam maps
tele_geo.x_ap = 0
tele_geo.z_ap = 0

# Measurement parameters
SWEEP = 100  # 500  # distance [mm] of sweep in 1D
SHIFT = 0  # optional shift centering of source

tele_geo.lambda_ = 0.299792458 / freq
tele_geo.k = 2 * np.pi / tele_geo.lambda_

# filter_geo = ot_geo.filter_geo_new()
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
status = MPI.Status()
t_i = time.perf_counter()

# Path where simulation is saved
FILE_NAME = (
    "data/source_test_"
    + str(round(-tele_geo.y_ap / 1e3, 2))
    + "m_"
    + str(int(freq))
    + "GHz.txt"
)


def enum(*seq):
    """
    Define list of tags to call during parallelization.
    """
    enums = dict(zip(seq, range(len(seq))))
    return type("Enum", (), enums)


Tags = enum("READY", "CALC", "DONE", "ERROR", "EXIT")

## rank zero is the main thread. It manages the others and passes messages them
if rank == 0:
    # nworkers is the number of sub threads
    pbar = tqdm(total=RES ** 2)
    nworkers = size - 1
    rx = np.linspace(-SWEEP / 2 + SHIFT, SWEEP / 2 + SHIFT, RES)
    result_compile = np.zeros((2, len(rx), len(rx)))
    x_compile = np.zeros((len(rx), len(rx)))
    y_compile = np.zeros((len(rx), len(rx)))
    for i, x in enumerate(rx):
        for j, y in enumerate(rx):
            ## receive messages from the workers
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            ## send back new calculation
            send = {
                "calc_index_x": x,
                "idx_x": i,
                "calc_index_y": y,
                "idx_y": j,
            }
            comm.send(send, dest=source, tag=Tags.CALC)

            if tag == Tags.DONE:
                result_compile[0, msg["idx_x"], msg["idx_y"]] = msg["result"][0]
                result_compile[1, msg["idx_x"], msg["idx_y"]] = msg["result"][1]
                x_compile[msg["idx_x"], msg["idx_y"]] = msg["result"][2]
                y_compile[msg["idx_x"], msg["idx_y"]] = msg["result"][3]

                pbar.update(1)
            elif tag == Tags.ERROR:
                print(msg["error"])

    for _ in range(nworkers):
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        comm.send(None, dest=source, tag=Tags.EXIT)
        if tag == Tags.DONE:
            result_compile[0, msg["idx_x"], msg["idx_y"]] = msg["result"][0]
            result_compile[1, msg["idx_x"], msg["idx_y"]] = msg["result"][1]
            x_compile[msg["idx_x"], msg["idx_y"]] = msg["result"][2]
            y_compile[msg["idx_x"], msg["idx_y"]] = msg["result"][3]

            np.savetxt(
                FILE_NAME,
                np.c_[
                    np.ravel(x_compile),
                    np.ravel(y_compile),
                    np.ravel(result_compile[0, :, :]),
                    np.ravel(result_compile[1, :, :]),
                ],
            )
        elif tag == Tags.ERROR:
            print(msg["error"])

else:

    comm.send(None, dest=0, tag=Tags.READY)

    while True:
        msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == Tags.CALC:
            # print('I have been told to do math on {}'.format(msg['calc_index_x']))
            ## call a function to get result
            rx_x = msg["calc_index_x"]
            rx_z = msg["calc_index_y"]

            tele_geo.N_scan = 80
            aff_rx = ray_trace.rx_to_lyot_model([0, 0, 0], tele_geo, plotOpts)
            out_rx_ly = aff_rx.output()

            tele_geo.N_scan = 150
            aff_so = ray_trace.source_to_lyot_model(
                [rx_x, tele_geo.y_ap, rx_z], tele_geo, plotOpts
            )
            out_so_ly = aff_so.output()

            lyot_r = np.sqrt(out_rx_ly[0, :] ** 2 + out_rx_ly[2, :] ** 2)
            R_LYOT = 136.54 / 2

            rx_new, so_new = ray_trace.regrid(out_rx_ly, out_so_ly)
            lyot_r = np.sqrt(rx_new[0, :] ** 2 + rx_new[2, :] ** 2)

            beam_tot = (
                so_new[4, :]
                * rx_new[4, :]
                * np.exp(complex(0, 1) * rx_new[3, :] * tele_geo.k / 1e3)
                * np.exp(complex(0, 1) * so_new[3, :] * tele_geo.k / 1e3)
            )

            beam_tot = np.sum((beam_tot[np.where((lyot_r <= R_LYOT))]))
            amp_tot = np.abs(beam_tot)
            phi_tot = np.arctan2(np.imag(beam_tot), np.real(beam_tot))

            msg["result"] = [amp_tot, phi_tot, rx_x, rx_z]
            comm.send(msg, dest=0, tag=Tags.DONE)

        elif tag == Tags.EXIT:
            print("bailing")
            break
