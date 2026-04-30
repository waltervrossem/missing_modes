#!/usr/bin/env python

import os
import shutil
import tempfile
import tqdm

import numpy as np
import multiprocessing as mp

from wsssss import load_data as ld
from wsssss.constants import pre15140 as c

os.environ['OMP_NUM_THREADS'] = '2'
nworker = int(mp.cpu_count() / int(os.environ['OMP_NUM_THREADS']))


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


def worker(track):
    path = f'/data/walter/new_grid_part/tracks_p0/{track}/LOGS/'
    contents = os.listdir(path)
    base_name = contents[0]
    path = f'{path}/{base_name}'
    contents = os.listdir(path)
    hist = ld.History(f'{path}/{base_name}.track', index_name=f'{base_name}.index')
    hist = hist[np.isin(hist.get('model_number'), hist.index[:,0])]  # Only keep models with profiles
    avg_rho = (hist.get('star_mass')*c.msun)/((4/3)*np.pi*(hist.get('photosphere_r')*c.rsun)**3)
    concentration = np.log10(hist.get('center_Rho')/avg_rho)
    res = []
    for i, pnum in enumerate(hist.index[:,2]):
        if concentration[i] > 8:
            continue
        gp_path = f'{path}/{base_name}_n{pnum}.profile.GYRE'
        if not os.path.exists(gp_path):
            res.append([-pnum, base_name])
            continue
        gs = ld.GyreSummary(f'{gp_path}.freql012')
        if os.path.exists(f'{gp_path}.sgyre_l'):
            gs = ld.GyreSummary(f'{gp_path}.sgyre_l')
        with tempfile.TemporaryDirectory() as tmpdir:
            f_nfreq = 2
            while check_gs(gs):
                f_nfreq *= 5
                os.system(f'gyre-driver 012 MESA {gp_path} --gyre G9 --n-sig-lo 3 --n-sig-hi 4.5 '
                          f'--out-dir {path} --in-dir {tmpdir} --no-output --f-nfreq {f_nfreq}')
                gs = ld.GyreSummary(f'{gp_path}.sgyre_l')
                if f_nfreq >= 250:
                    res.append([pnum, base_name])
                    break
    return res


def worker2(args):
    i, gp_path = args
    if not os.path.exists(gp_path):
        return i, 1

    gp = ld.GyreProfile(gp_path)
    M_star = gp.header['star_mass']
    R_star = gp.header['star_radius']
    L_star = gp.header['star_luminosity']
    Teff_star = (L_star / (4 * np.pi * R_star ** 2 * c.boltz_sigma)) ** 0.25

    avg_rho = M_star/((4/3)*np.pi*R_star**3)
    concentration = np.log10(gp.data.density[0]/avg_rho)
    nu_max = 3100 * R_star/c.msun / ((R_star/c.rsun) ** 2 * np.sqrt(Teff_star / 5777))

    if concentration > 8 or nu_max < 2.5:
        return i, 2

    gs_path0 = f'{gp_path}.freql012'  # From Marco's gyre driver
    gs_path1 = f'{gp_path}.sgyre_l'

    try:
        if os.path.exists(gs_path1):  # Can exist if restarting check_gyre
            gs = ld.GyreSummary(gs_path1)
        else:
            gs = ld.GyreSummary(gs_path0)
    except FileNotFoundError:
        return i, 3

    with tempfile.TemporaryDirectory() as tmpdir:
        f_nfreq = 2
        while check_gs(gs):
            f_nfreq *= 5
            ierr = os.system(f'gyre-driver 012 MESA {gp_path} --gyre G9 --n-sig-lo 3 --n-sig-hi 4.5 '
                      f'--out-dir {path} --in-dir {tmpdir} --no-output --f-nfreq {f_nfreq}')
            if ierr != 0:
                i, 4
            if os.path.exists(gs_path1):
                gs = ld.GyreSummary(gs_path1)
            else:
                return i, 5
            if f_nfreq >= 250:
                return i, 6
    return i, 0


if __name__ == "__main__":
    base_dir = '/home/walter/desktop/data/new_grid_part/tracks_p0'
    base_dir = '/data/walter/new_grid_part/tracks_p0'
    track_dirs = [_ for _ in os.listdir(base_dir) if _.startswith('track_')]
    gp_paths = []
    for track in track_dirs:
        path = f'{base_dir}/{track}/LOGS/'
        contents = os.listdir(path)
        base_name = contents[0]
        path = f'{path}/{base_name}'
        contents = np.array([_ for _ in os.listdir(path) if _.endswith('.profile.GYRE')])

        # Sort by profile number
        pnum = np.array([int(gp_name.split('_n')[1].split('.')[0]) for gp_name in contents])
        contents = contents[np.argsort(pnum)]
        for gp_name in contents:
            gp_paths.append(f'{path}/{gp_name}')
    args = list(zip(np.arange(len(gp_paths)), gp_paths))

    out_status = []
    with mp.Pool(nworker) as p, tqdm.tqdm(total=len(args)) as pbar:
        for res in p.imap_unordered(worker2, args):
            out_status.append(res)
            pbar.update(1)
    out_status = np.array(out_status)
    out_status = out_status[np.argsort(out_status[:,0])]
    print(out_status)

    with open('out.stats', 'w') as handle:
        s = '\n'.join([str(_) for _ in out_status])
        for line in out_status:
            for item in line:
                s += str(item) + ' '
            s = s[:-1] + '\n'
        handle.write(s)
