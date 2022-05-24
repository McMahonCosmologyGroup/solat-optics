import os

# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
sys.path.append("/data/chesmore/holosim_paper/package/holosim-ml")
import pan_mod as pm

path_to_package = "/home/chesmore/Desktop/Code/solat-optics"
sys.path.append(path_to_package) 

import solat_optics.solat_apert_field as solat_apert_field
import solat_optics.latrt_geo as lgeo
import solat_optics.far_field_latrt as far_field_latrt
import solat_optics.phase_correct as phase_correct

import time

import numpy as np
from mpi4py import MPI
from tqdm import tqdm


freq = int(sys.argv[1])
res = float(sys.argv[2])
num = int(sys.argv[3])
offset = float(sys.argv[4])

######### Propagate through Telescope #############
tele_geo_t = lgeo.initialize_telescope_geometry()

rx_t = np.array([0, 0, 0])

tele_geo_t.rx_x = rx_t[0] 
tele_geo_t.rx_z = rx_t[2]

if freq <= 120:
    tele_geo_t.rx_y = rx_t[1] - offset
    sim_d = 1.25
else:
    tele_geo_t.rx_y = rx_t[1] - offset
    sim_d = 1.32

tele_geo_t.lambda_ = (30.0 / freq) * 0.01
tele_geo_t.k = 2 * np.pi / tele_geo_t.lambda_

path = "/data/chesmore/latrt_ot_sim/MF_simset/source_test_1.33m_150GHz.txt"
file_name = "../output_files/source_test_1.33m_150GHz.txt"

foc_fields = np.loadtxt(path)
shapenew = int(np.sqrt(len(foc_fields)))

# CHANGE THIS REVERSE WHEN DOING SIMS
x = np.reshape(foc_fields[:, 0], (shapenew, shapenew))/1e1 # + (rxz/1e1)
y = np.reshape(foc_fields[:, 1], (shapenew, shapenew))/1e1 # + (rxx/1e1)

amp = np.reshape(foc_fields[:, 2], (shapenew, shapenew))
phi = np.reshape(foc_fields[:, 3], (shapenew, shapenew))

adj1 = np.random.randn(1092) * 0
adj2 = np.random.randn(1092) * 0

tele_geo_t.N_scan = 50
th = np.linspace(np.pi - 0.00005, np.pi + 0.00005, tele_geo_t.N_scan)
ph = np.linspace(0, np.pi, tele_geo_t.N_scan)

save = 0
pan_mod2 = pm.panel_model_from_adjuster_offsets(
    2, adj2, 1, save
)  # Panel Model on M2
pan_mod1 = pm.panel_model_from_adjuster_offsets(
    1, adj1, 1, save
)  # Panel Model on M1

p_source = np.array([0, -7.2, 100e3])  # [m] total of 100km away from aperture plane

tele_geo_t.x_tow = p_source[0]
tele_geo_t.y_tow = p_source[1]
tele_geo_t.z_tow = p_source[2]
tele_geo_t.el0 = np.arctan(
        -tele_geo_t.y_tow / tele_geo_t.z_tow
    )  # elevation of telescope based on position of source tower [rad]

XX = x
YY = y
znew = amp
pnew = phase_correct.do_unwrap(phi * 180 / np.pi) * np.pi / 180
data = np.zeros((3, int((len(XX))), int((len(YY)))), dtype=complex)
data[0, :, :] = XX
data[1, :, :] = YY
data[2, :, :] = abs(znew) * np.exp(complex(0, 1) * np.mod(pnew, 2 * np.pi))

msmt_geo = tele_geo_t

# Break out many quantities from msmt_geo
N_scan = msmt_geo.N_scan
de_ang = res / 60 * np.pi / 180  # msmt_geo.de_ang

lambda_ = msmt_geo.lambda_

x_tow = msmt_geo.x_tow
y_tow = msmt_geo.y_tow
z_tow = msmt_geo.z_tow

x_phref = msmt_geo.x_phref
y_phref = msmt_geo.y_phref
z_phref = msmt_geo.z_phref

x_rotc = msmt_geo.x_rotc
y_rotc = msmt_geo.y_rotc
z_rotc = msmt_geo.z_rotc

k = 2.0 * np.pi / lambda_  # Wavenumber [1/m]

# Complex fields
ima = complex(0, 1)

############
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
status = MPI.Status()
t_i = time.perf_counter()

el0 = msmt_geo.el0
az0 = msmt_geo.az0

def enum(*seq):
    enums = dict(zip(seq, range(len(seq))))
    return type("Enum", (), enums)

N_scan = int(num)

len1d = 2*N_scan+1
azs = np.linspace(-N_scan * de_ang + az0, N_scan * de_ang + az0, len1d)
els = np.linspace(-N_scan * de_ang + el0, N_scan * de_ang + el0, len1d)

out = np.zeros((len1d), dtype=complex)

tags = enum("READY", "CALC", "DONE", "ERROR", "EXIT")

## rank zero is the main thread. It manages the others and passes messages them
if rank == 0:
    # nworkers is the number of sub threads
    pbar = tqdm(total=(len1d)**2)
    nworkers = size - 1
    result_compile = np.zeros((2, len1d,len1d))
    x_compile = np.zeros((len1d,len1d))
    y_compile = np.zeros((len1d,len1d))
    out = np.zeros((2, len1d, len1d), dtype=complex)

    for i_ang, az_cur in enumerate(azs):
        for j_ang, el_cur in enumerate(els):

            ## receive messages from the workers
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            ## send back new calculation
            send = {
                "calc_index_x": az_cur,
                "idx_x": i_ang,
                "calc_index_y": el_cur,
                "idx_y": j_ang,

            }
            comm.send(send, dest=source, tag=tags.CALC)

            if tag == tags.DONE:
                result_compile[0, msg["idx_x"],msg["idx_y"]] = msg["result"][0]
                result_compile[1, msg["idx_x"],msg["idx_y"]] = msg["result"][1]
                x_compile[msg["idx_x"],msg["idx_y"]] = msg["result"][2]
                y_compile[msg["idx_x"],msg["idx_y"]] = msg["result"][3]

                pbar.update(1)
            elif tag == tags.ERROR:
                print(msg["error"])

    for _ in range(nworkers):
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        comm.send(None, dest=source, tag=tags.EXIT)
        if tag == tags.DONE:
            result_compile[0, msg["idx_x"],msg["idx_y"]] = msg["result"][0]
            result_compile[1, msg["idx_x"],msg["idx_y"]] = msg["result"][1]
            x_compile[msg["idx_x"],msg["idx_y"]] = msg["result"][2]
            y_compile[msg["idx_x"],msg["idx_y"]] = msg["result"][3]

            np.savetxt(
                file_name,
                np.c_[
                    np.ravel(x_compile),
                    np.ravel(y_compile),
                    np.ravel(result_compile[0, :,:]),
                    np.ravel(result_compile[1, :,:]),
                ],
            )

        elif tag == tags.ERROR:
            print(msg["error"])

else:

    comm.send(None, dest=0, tag=tags.READY)

    while True:
        msg = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == tags.CALC:
            # print('I have been told to do math on {}'.format(msg['calc_index_x']))
            ## call a function to get result
            el = msg["calc_index_y"]
            az = msg["calc_index_x"]
     
            ## do math here
            apert_rx = solat_apert_field.ray_mirror_pts(msmt_geo, th, ph, el, az)
            F_out = solat_apert_field.aperature_fields_from_panel_model(
                pan_mod1, pan_mod2, msmt_geo, th, ph, apert_rx, el, az
            )

            X_spat = data[0, :, :]
            Y_spat = data[1, :, :]
            out_new = far_field_latrt.project_to_data(F_out, X_spat, Y_spat, msmt_geo)
            Npts = len(out_new)
            out_beam = np.sum(out_new * data[2, :, :]) / Npts**2

            amp = abs(out_beam)
            phi = np.arctan2(np.imag(out_beam),np.real(out_beam))
            msg["result"] = [amp,phi, az, el]

            comm.send(msg, dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
#             print("bailing")
            break
