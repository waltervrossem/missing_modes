#!/usr/bin/env python

import os
import tqdm
import tempfile
import tarfile
import shutil
import multiprocessing as mp
import numpy as np
from wsssss import load_data as ld


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


def get_gyre_summary(model_name, base_path):
    fname = model_name.split('/')[-1]
    mass, ovh, aFe, FeH, DYDZ, Y0, pnum = split_name(fname)
    Y_aFe_dir = 'Y' + fname[50:52] + fname[27:29]
    tarfile_name = f'{base_path}/{Y_aFe_dir}/' + fname[::-1].split('.', maxsplit=1)[-1][::-1] + '.tar.gz'
    out_paths = extract_files(tarfile_name, fname, '.')
    return [ld.GyreSummary(path) for path in out_paths]


def get_grid_track_index_by_name(grid, name):
    name_no_prof_num = name[::-1].split('n', maxsplit=1)[1][::-1][:-1]
    for i_grid, track in enumerate(grid.tracks):
        if track.names[0].startswith(name_no_prof_num):
            for i_track, tname in enumerate(track.names):
                if tname.endswith(name):
                    return i_grid, i_track
        else:
            continue
    raise IndexError(f'{name} not found in grid ')


def modes_to_simple(path, modes):
    freq_min = modes['freq'][modes['l'] == 0].min()
    freq_max = modes['freq'][modes['l'] == 0].max()
    modes = modes[(modes['freq'] >= freq_min) & (modes['freq'] <= freq_max)]
    s = '# l     n_pg    Freq(uHz)       n_g     E_norm\n'
    for mode in modes:
        n_p, n_g, l, freq, E_norm = mode
        s += f'{l:> 6} {n_p:> 5} {freq:> 19e} {n_g: >19e} {E_norm:> 19e}\n'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(s)


def gs_to_simple(path, gs):
    modes = gs.data[['n_p', 'n_g', 'l', 'Re(freq)', 'E_norm']].astype(
        [('n', int), ('n_g', int), ('l', int), ('freq', float), ('inertia', float)])
    modes_to_simple(path, modes)


def do_track(name):
    mass, ovh, aFe, FeH, DYDZ, Y0 = split_name(name)
    tarfile_path = f"{base_path_freq}/Y{int(100*Y0)}p{int(10*aFe)}/{name}.tar.gz".replace('p-', 'm')
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tarfile_path, mode='r') as tf:
            tf.extractall(path=tmpdir)
        dirpath, dirnames, fnames = os.walk(tmpdir, topdown=False).__next__()
        shutil.move(dirpath, f'{tmpdir}')
        req_pnums = [_.split('.n')[1].split('.')[0] for _ in os.listdir(f'{target_base}/{name}')]

        for pnum in req_pnums:
            gs = ld.GyreSummary(f'{tmpdir}/{name}/{name}_n{pnum}.profile.GYRE.freql012')
            gs_to_simple(f'{target_base}/{name}/{name}.n{pnum}.freq', gs)

        shutil.rmtree(f'{tmpdir}')
    return name


# base_dir = '/home/walter/work/AIMS_count_modes/rerun/storage_tracks'
# base_path_freq = '/data/walter/models_mesa_wfreq'

base_dir = '/home/walter/work/fix_grid/gyre_out'
target_base = '/home/walter/work/fix_grid/storage_files_for_grids/files_grid_RGB_MAZYAp'

if __name__ == "__main__":
    out = []
    names = os.listdir(target_base)
    nproc = mp.cpu_count()
    nproc = 12
    # with mp.Pool(nproc) as p, tqdm.tqdm(total=len(names)) as pbar:
    #     for res in p.imap(do_track, names):
    #         out.append(res)
    #         pbar.update(1)

    bad = []
    for dirpath, dirnames, filenames in os.walk(base_dir, topdown=False):
        for fname in tqdm.tqdm(filenames):
            try:
                if 'gyre_out' not in dirpath:
                    continue
                if not fname.endswith('.sgyre_l'):
                    continue
                gs = ld.GyreSummary(os.path.join(dirpath, fname))
                freq = gs.data['Re(freq)']

                # Check if something went wrong when running gyre_driver
                if gs.data['l'][-1] == 0:
                    bad.append(fname)
                    continue
                if gs.data['l'][0] != 0:
                    bad.append(fname)
                    continue

                mass, ovh, aFe, FeH, DYDZ, Y0, pnum = split_name(fname.replace('.profile.GYRE.sgyre_l', ''))
                basename = fname.replace('_', '.').split('.n')[0]

                path = f'{target_base}/{basename}/{basename}.n{pnum}.freq'
                gs_to_simple(path, gs)
            except:
                print(os.path.join(dirpath, fname))
                print(fname)
                raise
    print(bad)
    with open('bad_names.txt', 'w') as handle:
        for line in bad:
            handle.write(f'{line}\n')
