#!/usr/bin/env python

import os
import shutil
import sys
import tempfile
import tqdm
import tarfile

import numpy as np
import multiprocessing as mp

from wsssss import load_data as ld
from wsssss.constants import pre15140 as c


def check_gs(gs):
    if gs is None:
        return True
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


def check_skip(gp):
    M_star = gp.header['star_mass']
    R_star = gp.header['star_radius']
    L_star = gp.header['star_luminosity']
    Teff_star = (L_star / (4 * np.pi * R_star ** 2 * c.boltz_sigma)) ** 0.25

    avg_rho = M_star / ((4 / 3) * np.pi * R_star ** 3)
    concentration = np.log10(gp.data.density[0] / avg_rho)
    nu_max = 3100 * M_star / c.msun / ((R_star / c.rsun) ** 2 * np.sqrt(Teff_star / 5777))

    return concentration > 8 or nu_max < 2.5

def extract_files(tarfile_path, target_files, extract_dir):
    if isinstance(target_files, str):
        target_files = [target_files]
    # Can have _ separating profile number
    target_files = [_.replace('_', '.') for _ in target_files]

    if not os.path.exists(tarfile_path):
        raise FileNotFoundError(tarfile_path)
    tf = tarfile.open(tarfile_path, mode='r')

    out_paths = []
    for member in tf.getmembers():
        for target_file in target_files:
            if target_file in member.path.replace('_', '.'):
                tf.extract(member, path=f'{extract_dir}')
                out_paths.append(f'{extract_dir}/{member.path}')
    tf.close()
    return out_paths


def get_mask(hist, kind):
    center_h1 = hist.get('center_h1')
    center_he4 = hist.get('center_he4')
    log_L_trialpha = hist.get('log_LHe')
    if kind == 'MS':
        mask = (center_h1 >= 1.0e-6)
    elif kind == 'RGB':
        mask = (center_h1 < 1.0e-12) & (center_he4 > 0.90) &  (log_L_trialpha < 0.2) & (hist.get('nu_max') > 2.5)
    elif kind == 'RC':
        mask = (center_h1 < 1.0e-12) & (center_he4 > 1e-9) & (log_L_trialpha >= 0.2) & (hist.get('mass_conv_core') > 0) & (hist.get('nu_max') > 2.5)
    else:
        raise NotImplemented
    return mask


def worker(args):
    i, gs_name = args

    track_name = gs_name.split('/')[-1].split('_n')[0]
    model_name = gs_name.split('/')[-1].replace('.freql012', '')
    Y_aFe_dir = 'y' + track_name[50:52] + track_name[27:29]

    tarfile_path = f'{base_gp_dir}/{Y_aFe_dir}/{track_name}.tar.gz'
    if not os.path.exists(tarfile_path):
        return i, 7

    with tempfile.TemporaryDirectory() as tmpdir:
        gp_path = extract_files(tarfile_path, model_name, tmpdir)[0]
        if not os.path.exists(gp_path):
            return i, 1

        gp = ld.GyreProfile(gp_path)
        if check_skip(gp):
            return i, 2

        try:
            gs = ld.GyreSummary(f'{base_gs_dir}/{Y_aFe_dir}/{track_name}/{gs_name}')
        except FileNotFoundError:
            return i, 3

        f_nfreq = 2
        while check_gs(gs):
            f_nfreq *= 5
            ierr = os.system(f'gyre-driver 012 MESA {gp_path} --gyre G9 --n-sig-lo 3 --n-sig-hi 4.5 '
                             f'--out-dir {base_gs_dir}/{Y_aFe_dir}/{track_name} --in-dir {tmpdir} --no-output '
                             f'--f-nfreq {f_nfreq} --summary-suffix .freql012')
            if ierr != 0:
                i, 4
            if os.path.exists(gs_path0):
                gs = ld.GyreSummary(gs_path0)
            else:
                return i, 5
            if f_nfreq >= 250:
                return i, 6
    return i, 0


def worker_track(args):
    i, gs_names = args
    gs_name = gs_names[0]  # Just to get track name
    track_name = gs_name.split('/')[-1].split('_n')[0]
    Y_aFe_dir = 'y' + track_name[50:52] + track_name[27:29]

    tarfile_path = f'{base_gp_dir}/{Y_aFe_dir}/{track_name}.tar.gz'
    out = []
    if not os.path.exists(tarfile_path):
        for j in range(len(gs_names)):
            out.append([i, j, 7])
        return out
    gp_names = [_.replace('.freql012', '') for _ in gs_names]

    with tempfile.TemporaryDirectory() as tmpdir:
        gp_paths = extract_files(tarfile_path, gp_names, tmpdir)
        for j, gp_path in enumerate(gp_paths):
            if not os.path.exists(gp_path):
                out.append([i, j, 1])
                continue

            gp = ld.GyreProfile(gp_path)
            if check_skip(gp):
                out.append([i, j, 2])
                continue

            gs_path = f'{base_gs_dir}/{Y_aFe_dir}/{track_name}/{gs_name}'
            try:
                gs = ld.GyreSummary(gs_path)
            except FileNotFoundError:
                out.append([i, j, 3])
                continue

            f_nfreq = 2
            for f_nfreq in [2, 10, 25]:
                ierr = os.system(f'gyre-driver 012 MESA {gp_path} --gyre G9 --n-sig-lo 3 --n-sig-hi 4.5 '
                                 f'--out-dir {base_gs_dir}/{Y_aFe_dir}/{track_name} --in-dir {tmpdir} --no-output '
                                 f'--f-nfreq {f_nfreq} --summary-suffix .freql012')
                gs = ld.GyreSummary(gs_path)
                have_missing = check_gs(gs)
                if not have_missing:
                    break
            if have_missing:
                out.append([i, j, 6])
            else:
                out.append([i, j, 0])
            #
            # while check_gs(gs):
            #     f_nfreq *= 5
            #     ierr = os.system(f'gyre-driver 012 MESA {gp_path} --gyre G9 --n-sig-lo 3 --n-sig-hi 4.5 '
            #                      f'--out-dir {base_gs_dir}/{Y_aFe_dir}/{track_name} --in-dir {tmpdir} --no-output '
            #                      f'--f-nfreq {f_nfreq} --summary-suffix .freql012')
            #     if ierr != 0:
            #         i, 4
            #     gs = ld.GyreSummary(gs_path)
            #     if f_nfreq >= 50:
            #         out.append([i, j, 6])
            #         continue
            # out.append([i, j, 0])
    return out


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['GYRE_DIR'] = '/home/walter/Software/gyre/gyre-9.0'
nworker = int(mp.cpu_count() / int(os.environ['OMP_NUM_THREADS']))

base_gs_dir = '/data/walter/repackage_models_mesa_wfreq/storage_tracks'
base_gp_dir = '/data/walter/grid_nofreqs'
if __name__ == "__main__":
    YaFe_list = os.listdir(base_gs_dir)

    if not os.path.exists('input.txt'):
        args = []
        i = 0
        for YaFe in tqdm.tqdm(YaFe_list):
            if 'm2' in YaFe:
                continue
            for track_name in tqdm.tqdm(os.listdir(f'{base_gs_dir}/{YaFe}')):
                hist = ld.History(f'{base_gs_dir}/{YaFe}/{track_name}/{track_name}.track', save_dill=False,
                                  index_name=f'{track_name}.index')
                hist = hist[np.isin(hist.get("model_number"), hist.index[:,0])]
                avg_rho = c.msun * hist.get("star_mass")[0] / (
                            (4 / 3) * np.pi * c.rsun ** 3 * hist.get('photosphere_r') ** 3)
                concentration = np.log10(hist.get("center_Rho") / avg_rho)
                mask = (hist.get("nu_max") >= 2.5) & (concentration < 8)

                # Only check for models used in grid
                kind_mask = np.zeros_like(mask, dtype=bool)
                for kind in ['MS', 'RGB', 'RC']:
                    kind_mask = kind_mask | get_mask(hist, kind)
                mask = mask & kind_mask

                for pnum in hist.index[mask, 2]:
                    gs_name = f'{track_name}_n{pnum}.profile.GYRE.freql012'
                    gs_path = f'{base_gs_dir}/{YaFe}/{track_name}/{gs_name}'
                    if os.path.exists(gs_path):
                        gs = ld.GyreSummary(gs_path)
                    else:
                        gs = None
                    if check_gs(gs):
                        args.append([i, gs_name])
                        i += 1

        np.savetxt('input.txt', args, fmt=['%8s', '%s'])
    else:
        with open('input.txt', 'r') as handle:
            args = handle.readlines()
        for i, line in enumerate(args):
            line_i, path = line.split()
            args[i] = [int(line_i), path]

    args = np.array(args)
    args = args[np.argsort(args[:,1])]
    _args = []
    i = 0
    for _ in args[:,1]:
        _args.append([i, _.split('/')[-1]])
        i += 1
    args = _args
    del _args

    mapping = {}
    for line in args:
        track = line[1].split('_n')[0]
        if 'aFem2' in track:
            continue  # Skip m2 as don't have profiles
        if not track in mapping.keys():
            mapping[track] = [line[1]]
        else:
            mapping[track].append(line[1])
    args_track = mapping.values()
    args_track = list(zip(range(len(args_track)), args_track))
    out_status = []
    with mp.Pool(nworker) as p, tqdm.tqdm(total=len(args)) as pbar:
        # for res in p.imap_unordered(worker, args):
        #     out_status.append(res)
        #     pbar.update(1)
        for res in p.imap_unordered(worker_track, args_track):
            out_status.extend(res)
            pbar.update(len(res))
    # for arg in tqdm.tqdm(args_track[:1]):
    #     out_status.extend(worker_track(arg))
    out_status = np.array(out_status)
    out_status = out_status[np.argsort(out_status[:,0])]
    print(out_status)
    print(out_status.shape)
    np.savetxt('out.stats', out_status, fmt=['%8i', '%3i', '%3i'])

    # Remove intermediate files from cancelled runs
    os.system(f'rm {base_gs_dir}/*/*/*.sgyre_l0')
    os.system(f'rm {base_gs_dir}/*/*/*.sgyre_l1')
    os.system(f'rm {base_gs_dir}/*/*/*.sgyre_l2')
