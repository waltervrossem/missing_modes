#!/usr/bin/env python

import os
import glob
import tarfile
import tempfile
import tqdm
import multiprocessing as mp
import numpy as np

from wsssss import load_data as ld
from wsssss.constants import pre15140 as c

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


def get_hist_and_index(path, kind):
    if path.endswith('.track'):
        index_path = path.split('/')[-1].replace('.track', '.index')
        hist = ld.History(path, index_name=index_path)
    else:
        with tarfile.open(path, mode='r') as tf:
            for member in tf.getmembers():
                if member.path.endswith('.track'):
                    hist_member = member
                elif member.path.endswith('.index'):
                    index_member = member

            with tempfile.TemporaryDirectory() as tmpdir:
                for member in [hist_member, index_member]:
                    tf.extract(member, path=f'{tmpdir}')
                hist = ld.History(hist_member.path, index_name=index_member.path.split('/')[-1])
                hist.data  # Load data before closing tmpdir
    hist = hist[np.isin(hist.data.model_number, hist.index[:,0])]
    hist = hist[get_mask(hist, kind)]
    hist.index = hist.index[np.isin(hist.index[:,0], hist.data.model_number)]
    return hist


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


def get_age_adim(hist, kind):
    if kind == 'MS':
        age_adim = hist.get('center_h1')
    elif kind == 'RGB':
        age_adim = np.log10(hist.get('center_Rho') / (hist.get('star_mass')*c.msun) / (4/3 * np.pi * c.rsun**3 * hist.get('photosphere_r')**3))
    elif kind == 'RC':
        age_adim = (hist.get('star_age') - hist.get('star_age')[0]) / (hist.get('star_age')[-1] - hist.get('star_age')[0])
    else:
        raise NotImplemented
    return age_adim


def make_input_dat(hist, kind):
    base_name = hist.fname.replace('.track', '')
    mass, ovh, aFe, FeH, DYDZ, Y0 = split_name(base_name)
    ones = np.ones_like(hist.get('star_mass'))

    Xs = hist.get('surface_h1') + hist.get('surface_h2')
    Ys = hist.get('surface_he3') + hist.get('surface_he4')
    Zs = 1 - Xs - Ys

    Xc = hist.get('center_h1') + hist.get('center_h2')
    Yc = hist.get('center_he3') + hist.get('center_he4')
    Zc = 1 - Xc - Yc

    # Zsol from initial Z of input composition file input_initial_xa0.600_pp_extras.net0.55+0.0-0.22.data
    Z_ini = 10**FeH * (0.04714908164218401 / 10**0.55)
    X_ini = 1 - Y0 - Z_ini

    dat = [hist.get('star_mass') * c.msun, 'Mass',
           hist.get('photosphere_r') * c.rsun, 'Radius',
           hist.get('photosphere_L') * c.lsun, 'Luminosity',
           ones*Z_ini, 'Zi',
           ones*X_ini, 'Xi',
           hist.get('star_age') / 1e6, 'Age',
           hist.get('effective_T'), 'Teff',
           get_age_adim(hist, kind), 'Age_adim',
           Zs, 'Zs',
           Xs, 'Xs',
           ones*Y0, 'Yi',
           Ys, 'Ys',
           Zc, 'Zc',
           Xc, 'Xc',
           Yc, 'Yc',
           hist.get('nu_max'), 'numax.mod',
           hist.get('delta_nu'), 'Dnu.mod',
#           ones, 'DnuG.mod', #DnuG not neccesary as calculated in AIMS as Dnu
           hist.get('delta_Pg'), 'DPg.mod',
           ones*(Z_ini / X_ini), 'ZHsurf.i',
           Zs/Xs, 'ZHsurf',
           ones*FeH, 'FeH.i',
           ones*aFe, 'aFe.i',
           hist.get('he_core_mass'), 'mHec',
           hist.get('he_core_radius'), 'rHec',
           hist.get('c_core_mass'), 'mCc',
           hist.get('c_core_radius'), 'rCc',
           hist.get('mass_conv_core'), 'mconv.core',
           np.log10(hist.get('center_T')), 'logTc',
           np.log10(hist.get('center_P')), 'logPc',
           np.log10(hist.get('center_Rho')), 'logRhoc',
           hist.get('pp'), 'logpp',
           hist.get('cno'), 'logcno',
           hist.get('tri_alfa'), 'log3a',
           hist.get('center_c12'), 'C12c',
           hist.get('surface_c12'), 'C12s',
           hist.get('center_n14'), 'N14c',
           hist.get('surface_n14'), 'N14s',
           hist.get('center_o16'), 'O16c',
           hist.get('surface_o16'), 'O16s',
           hist.get('center_mg24'), 'Mg24c',
           hist.get('surface_mg24'), 'Mg24s',
           hist.get('surface_li7'), 'Li7s'
           ]
    dat, cols = dat[::2], dat[1::2]
    dat = np.array(dat).T

    paths = [f'{base_name}/{base_name}.n{pnum}' for pnum in hist.index[:,2]]

    return paths, dat, cols


def worker(track_path):
    hist = get_hist_and_index(track_path, kind)
    return make_input_dat(hist, kind)

kind = 'MS'  # MS, RGB, or RC
output_path = f'/data/walter/grid_data/{kind}'
grid_name = f'grid_{kind}_MAZYAp.dat'
os.makedirs(output_path, exist_ok=True)

base_path = '/data/walter/repackage_models_mesa_wfreq/storage_tracks'
freq_suffix = '.freql012'
nworker = mp.cpu_count()
if __name__ == "__main__":
    args = []
    YaFe_list = os.listdir(base_path)
    for YaFe in YaFe_list:
        if 'm2' in YaFe:  # Don't have profiles for these so have missing modes.
            continue
        for track_name in os.listdir(f'{base_path}/{YaFe}'):
            if track_name.endswith('.tar.gz'):
                args.append(f'{base_path}/{YaFe}/{track_name}')
            else:
                args.append(f'{base_path}/{YaFe}/{track_name}/{track_name}.track')
    out = []
    with mp.Pool(nworker) as p, tqdm.tqdm(total=len(args)) as pbar:
        for res in p.imap_unordered(worker, args):
            out.append(res)
            pbar.update(1)
    columns = out[0][2]
    data = np.concatenate([row[1] for row in out])
    paths = np.concatenate([row[0] for row in out])

    sort_cols = ['Age', 'FeH.i', 'Mass', 'Yi', 'aFe.i']
    sort_i = [columns.index(col) for col in sort_cols]
    data = data[np.lexsort(data[:, sort_i].T)]

    with open(f'{output_path}/{grid_name}', 'w') as handle:
        handle.write(f'{base_path}    {freq_suffix}\n')
        for i, path in enumerate(paths):
            handle.write(f'{path:<112}' + str(data[i])[1:-1].replace('\n', '') + '\n')
    with open(f'{output_path}/{grid_name}_columns', 'w') as handle:
        handle.write('\n'.join(columns))
