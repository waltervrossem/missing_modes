#!/usr/bin/env python

import os
import shutil
import shlex
import sys
import time

import dill
import platform
import tarfile
import tqdm
import tempfile

from matplotlib import pyplot as plt

import numpy as np
import multiprocessing as mp
import importlib
from scipy import interpolate as ip
from wsssss import load_data as ld
from wsssss.plotting import plotting as pl
from wsssss.plotting import utils as pu
from wsssss.constants import pre15140 as const
from wsssss._bin.gyre_driver import gyre_driver

global model
global config

import AIMS_configure as config


curdir = os.path.abspath('.')

if 'AIMS_DIR' in os.environ:
    AIMS_dir = os.environ['AIMS_DIR']
    print(f"{AIMS_dir=}")
else:
    raise EnvironmentError('Environment variable AIMS_DIR is not set')

for name in ['constants', 'utilities', 'functions', 'aims_fortran', 'model']:
    if name == 'aims_fortran':
        files = os.listdir(f'{AIMS_dir}/src/')
        files = [_ for _ in files if _.startswith(f'aims_fortran.cpython-{sys.version_info.major}{sys.version_info.minor}')]
        if len(files) == 1:
            file_path = f'{AIMS_dir}/src/{files[0]}'
        else:
            raise ImportError(f'Cannot find correct compiled aims_fortran.f90 in {AIMS_dir}/src/.\n'
                              f'Expecting aims_fortran.cpython-{sys.version_info.major}{sys.version_info.minor}-*.so\n'
                              f'Found: {",".join(files)}')
    else:
        file_path = f'{AIMS_dir}/src/{name}.py'
    module_name = name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    exec(f'{name} = module')

model.config = config

msol = constants.solar_mass
rsol = constants.solar_radius
lsol = constants.solar_luminosity
tsol = constants.solar_temperature
numax_sol = constants.solar_numax

def load_grid(grid_path):
    with open(grid_path, 'rb') as handle:
        grid = dill.load(handle)
    config.user_params = grid.user_params
    config.grid_params = grid.grid_params
    model.config.user_params = grid.user_params
    model.nglb = 9 + len(config.user_params)
    model.nlin = 6 + len(config.user_params)
    model.ifreq_ref = 6 + len(config.user_params)
    model.iradius = 7 + len(config.user_params)
    model.iluminosity = 8 + len(config.user_params)
    model.init_user_param_dict()

    base_model_indeces = {
        'Age_adim': model.iage_adim,
        'Age': model.iage,
        'Mass': model.imass,
        'Temperature': model.itemperature,
        'Teff': model.itemperature,
        'Z0': model.iz0,
        'X0': model.ix0,
        'Freq_ref': model.ifreq_ref,
        'Radius': model.iradius,
        'Luminosity': model.iluminosity
    }
    model.user_params_index.update(base_model_indeces)
    return grid


def get_track_index(tarfile_path, target_file):
    tf = tarfile.open(tarfile_path, mode='r')
    for member in tf.getmembers():
        if member.path.endswith('.track'):
            hist_member = member
        elif member.path.endswith('.index'):
            index_member = member

    for member in [hist_member, index_member]:
        tf.extract(member, path=f'{curdir}')
    tf.close()
    return ld.MesaData(hist_member.path, index_name=index_member.path.split('/')[-1])


def extract_files(tarfile_path, target_files, extract_dir=None):
    if extract_dir is None:
        extract_dir = '.'
    if isinstance(target_files, str):
        target_files = [target_files]
    # Can have _ separating profile number
    target_files = [_.replace('_n', '.n') for _ in target_files]

    if not os.path.exists(tarfile_path):
        raise FileNotFoundError(tarfile_path)

    with tarfile.open(tarfile_path, mode='r') as tf:
        out_paths = []
        for member in tf.getmembers():
            for target_file in target_files:
                if os.path.exists(member.path) and member.isfile():
                    out_paths.append(member.path)
                    continue
                if target_file in member.path.replace('_n', '.n'):
                    tf.extract(member, path=extract_dir)
                    out_paths.append(member.path)
    return out_paths


def split_name(fname):
    mass = float(fname[1:6])
    ovh = float(fname[10:14])
    aFe = float(fname[27:29].replace('p', '0.').replace('m', '-0.'))
    FeH = float(fname[33:38])
    DYDZ = float(fname[43:46])
    Y0 = float(fname[48:53])
    if len(fname) == 53:
        return mass, ovh, aFe, FeH, DYDZ, Y0
    else:
        pnum = int(fname[-3:].replace('n', ''))
        return mass, ovh, aFe, FeH, DYDZ, Y0, pnum


def get_gyre_profile(model_name, base_path, temporary=False):
    fname = model_name.split('/')[-1]
    mass, ovh, aFe, FeH, DYDZ, Y0, pnum = split_name(fname)
    Y_aFe_dir = 'Y' + fname[50:52] + fname[27:29]
    tarfile_name = f'{base_path}/{Y_aFe_dir}/' + fname[::-1].split('.', maxsplit=1)[-1][::-1] + '.tar.gz'
    out_paths = extract_files(tarfile_name, fname, '.')
    gps = [ld.GyreProfile(path) for path in out_paths]
    if temporary:
        for path in out_paths:
            os.remove(path)
    return gps


def get_grid_index_by_name(grid, name):
    name_no_prof_num = name[::-1].split('n', maxsplit=1)[1][::-1][:-1]
    for i_grid, track in enumerate(grid.tracks):
        if track.names[0].startswith(name_no_prof_num):
            for i_track, tname in enumerate(track.names):
                if tname.endswith(name):
                    return i_grid, i_track
        else:
            continue
    raise IndexError(f'{name} not found in grid ')


def get_gyre_summary(model_name, base_path):
    fname = model_name.split('/')[-1]
    mass, ovh, aFe, FeH, DYDZ, Y0, pnum = split_name(fname)
    Y_aFe_dir = 'Y' + fname[50:52] + fname[27:29]
    tarfile_name = f'{base_path}/{Y_aFe_dir}/' + fname[::-1].split('.', maxsplit=1)[-1][::-1] + '.tar.gz'
    out_paths = extract_files(tarfile_name, fname, '.')
    return [ld.GyreSummary(path) for path in out_paths]


def get_model(grid, i_grid, i_track, aFe=None):
    track = grid.tracks[i_grid]
    return model.Model(track.glb[i_track],
                       _modes=track.modes[track.mode_indices[i_track]:track.mode_indices[i_track + 1]],
                       _name=track.names[i_track], aFe=aFe)


def worker_queue(i):
    # print(f'pid, i = {os.getpid()}, {i}, {q.qsize()}')
    # grid = load_grid(f'/home/walter/work/aims/marco_grid2025/{grid_kind}/grid_{grid_kind}_MAZYAp.aimsgrid')
    status = []
    t_prev = time.time()
    n_prev = q.qsize()
    while not q.empty():
        i_grid, i_track = q.get()
        try:
            amodel = get_model(grid, i_grid, i_track)
            if amodel.glb[model.user_params_index['AoFe.mod']] < 0:
                status.append([i_grid, i_track, 1])
                continue

            fname = amodel.name.split('/')[-1]
            mass, ovh, aFe, FeH, DYDZ, Y0, pnum = split_name(fname)
            Y_aFe_dir = 'Y' + fname[50:52] + fname[27:29]
            tarfile_name = f'{base_path_prof}/{Y_aFe_dir}/' + fname[::-1].split('.', maxsplit=1)[-1][::-1] + '.tar.gz'

            with tempfile.TemporaryDirectory() as tmpdir:
                outfiles = extract_files(tarfile_name, fname + '.profile.GYRE', extract_dir=tmpdir)
                f_nfreq = 2
                os.system(f'gyre-driver 012 MESA {tmpdir}/{outfiles[0]} --gyre G9 --n-sig-lo 3 --n-sig-hi 4.5 '
                          f'--out-dir {curdir}/gyre_out/ --in-dir {tmpdir}  --min-numax 5 --no-output --f-nfreq {f_nfreq}')
                gs = ld.GyreSummary(f'{curdir}/gyre_out/{outfiles[0].split("/")[-1]}.sgyre_l')
                while check_gs(gs):
                    f_nfreq *= 5
                    os.system(f'gyre-driver 012 MESA {tmpdir}/{outfiles[0]} --gyre G9 --n-sig-lo 3 --n-sig-hi 4.5 '
                              f'--out-dir {curdir}/gyre_out/ --in-dir {tmpdir}  --min-numax 5 --no-output --f-nfreq {f_nfreq}')
                    gs = ld.GyreSummary(f'{curdir}/gyre_out/{outfiles[0].split("/")[-1]}.sgyre_l')
                    if f_nfreq >= 250:
                        status.append([i_grid, i_track, 2])
                        break
                status.append([i_grid, i_track, 0])
        except Exception as exc:
            print(f'Failed: {i_grid} {i_track} {exc}')
            status.append([i_grid, i_track, 2])
        n_now = q.qsize()
        t_now = time.time()
        eta = (t_now - t_prev) / (n_prev - n_now) * n_now
        stat_str = ''
        if status[-1][-1] != 0:
            stat_str = f'{i_grid: >4} {i_track: >3} {status[-1][-1]}'
        print(f'worker-{i: >2} {q.qsize(): >6} {int(eta // 3600): >2}h{int(eta // 60) % 60:0>2}m{eta % 60 :0>5.2f}s {stat_str}')
    return status


def check_gs(gs):
    have_missing = False
    for l in [1, 2]:
        mask = gs.data.l == l
        n_p = gs.data.n_p[mask]
        n_g = gs.data.n_g[mask]
        n_pg = gs.data.n_pg[mask]
        if l == 1:
            mask_ng = n_p < n_g
            have_missing = not (np.all(np.diff(n_pg[mask_ng]) == 1) and np.all(np.diff(n_pg[~mask_ng]) == 1))
        else:
            have_missing = not np.all(np.diff(n_pg) == 1)
        if have_missing:
            return have_missing
    return have_missing

# base_path_prof = '/media/walter/My Book/grid_mesa_marco/grid_nofreqs'
# base_path_freq = '/media/walter/My Book/grid_mesa_marco/models_mesa_wfreq'
base_path_prof = '/data/walter/grid_nofreqs'
base_path_freq = '/data/walter/models_mesa_wfreq'

os.environ['OMP_NUM_THREADS'] = '1'
nworker = 24
grid_kind = 'RGB'
curdir = os.path.abspath('.')
if __name__ == "__main__":
    if not os.path.exists(f'{curdir}/gyre_out'):
        os.mkdir(f'{curdir}/gyre_out')
    # fname = f'grid_RGB_MAZYAp.aimsgrid_bad_counts.npz'
    # fname = 'bad_modes_grid_RGB_MAZYAp_v2.1.aimsgrid.npy'
    fname = 'bad_modes_grid_RGB_MAZYAp_ori.aimsgrid.npy'
    # fname = 'bad_names.txt'
    grid = load_grid(f'/home/walter/work/fix_grid_clean/grids/grid_{grid_kind}_MAZYAp_ori.aimsgrid')
    if fname.endswith('.npz'):
        bad_counts = np.load(fname)['arr_0']
    elif fname.endswith('.txt'):
        with open(fname, 'r') as handle:
            bad_names = handle.readlines()
        bad_counts = []
        for name in bad_names:
            name = name.strip()
            bad_counts.append(get_grid_index_by_name(grid, name.replace('.profile.GYRE.sgyre_l', '').replace('_n', '.n')))
        bad_counts = np.array(bad_counts, dtype=int)
    else:
        bad_counts = np.load(fname)

    q = mp.Queue()
    for _ in bad_counts[:, :2]:
        amodel = get_model(grid, _[0], _[1])
        if amodel.glb[model.user_params_index['AoFe.mod']] < 0:  # [a/Fe] = -0.2 Gyre profiles not available
            continue
        mname = amodel.name.split('/')[-1]
        # if os.path.exists(f'out/{mname}.profile.GYRE.sgyre_l'):
        #     continue
        # if os.path.exists(f'out/{mname.replace(".n", "_n")}.profile.GYRE.sgyre_l'):
        #     continue

        q.put(_)

        # if q.qsize() >= 240:
        #     break

    ts = time.time()
    out_status = []
    with mp.Pool(nworker) as p:
        out_status = p.map(worker_queue, np.arange(nworker, dtype=int))
    tt = time.time() - ts

    print(f'{int(tt // 3600): >2}h{int(tt // 60) % 60:0>2}m{tt % 60 :0>5.2f}s')
    out_status = np.concatenate([_ for _ in out_status if len(_) > 0])
    np.save('status.npy', out_status)
