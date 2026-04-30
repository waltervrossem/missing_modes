#!/usr/bin/env python

import os
import shutil
import sys
import dill
import platform
import tarfile
import tqdm
import time
import argparse
import traceback
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

marker = {0: 's',
          1: 'o',
          2: 'd'}

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


def get_model(grid, i_grid, i_track, aFe=None):
    track = grid.tracks[i_grid]
    return model.Model(track.glb[i_track],
                       _modes=track.modes[track.mode_indices[i_track]:track.mode_indices[i_track + 1]],
                       _name=track.names[i_track], aFe=aFe)


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


def make_plot(grid, i_grid, i_track, gs=None, norm_inertia=True, norm_DP=True):
    """
    Create a diagnostic plot for missing modes.

    :param grid: AIMS grid
    :param i_grid: Track index in grid
    :param i_track: Model index in track
    :param gs: Gyre summary (or AIMS model) with full mode spectrum.
    :param norm_inertia: Normalize mode inertia.
    :param norm_DP: Normalize period spacing diagram with asymptotic period spacing.
    :return:
    """
    tgt_model = get_model(grid, i_grid, i_track)

    f, axes = plt.subplot_mosaic('AB\nDC', figsize=[6.4, 6.4])
    ax = axes['A']
    ax.plot(grid.tracks[i_grid].glb[:, model.itemperature],
            grid.tracks[i_grid].glb[:, model.iluminosity] / constants.solar_luminosity, '.-')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.plot(tgt_model.glb[model.itemperature], tgt_model.glb[model.iluminosity] / constants.solar_luminosity, 'o')
    ax.set_xlabel('Teff (K)')
    ax.set_ylabel('L')

    ax1 = axes['B']
    ax2 = axes['C']
    ax3 = axes['D']
    ax1.sharex(ax2)
    ax3.sharex(ax1)
    a_surf = np.array([0, 0])
    Dnu_c = tgt_model.find_surface_corrected_large_separation(a_surf) * tgt_model.glb[model.ifreq_ref]
    i = -1

    nu_offset = 0.95 * Dnu_c - max(
        tgt_model.modes['freq'][tgt_model.modes['l'] == 0] * tgt_model.glb[model.ifreq_ref]) % Dnu_c
    if gs is not None:
        if isinstance(gs, ld.GyreSummary):
            mask_l0 = gs.data['l'] == 0
            freqs = gs.data['Re(freq)']
            inertias = gs.data['E_norm']
            ls = gs.data['l']
        elif isinstance(gs, model.Model):
            mask_l0 = gs.modes['l'] == 0
            freqs = gs.modes['freq'] * tgt_model.glb[model.ifreq_ref]
            inertias = gs.modes['inertia']
            ls = gs.modes['l']
        if norm_inertia:
            inertia_l0 = inertias[mask_l0]
            ip_l0 = ip.make_interp_spline(freqs[mask_l0], np.log10(inertias[mask_l0]), k=1)
        for l, _marker in marker.items():
            mask = ls == l
            freq = freqs[mask]
            inertia = inertias[mask]
            if norm_inertia:
                inertia = inertia / 10 ** ip_l0(freq)
            axes['B'].scatter(freq, (freq + nu_offset) % Dnu_c, c=f'C{l}', marker=_marker, zorder=9)
            axes['C'].scatter(freq, inertia, c=f'C{l}', marker=_marker, zorder=9, s=20)

            if l != 0:
                fDP_norm = 1
                if norm_DP:
                    fDP_norm = np.sqrt(l * (l + 1)) / (tgt_model.string_to_param('DPg.mod') * np.sqrt(2))
                axes['D'].scatter(freq[:-1], -np.diff(1e6 / freq) * fDP_norm, c=f'C{l}', marker=_marker, zorder=9, s=10)
    else:
        mask_l0 = tgt_model.modes['l'] == 0
        freqs = tgt_model.modes['freq'] * tgt_model.glb[model.ifreq_ref]
        inertias = tgt_model.modes['inertia']
        if norm_inertia:
            inertia_l0 = inertias[mask_l0]
            ip_l0 = ip.make_interp_spline(freqs[mask_l0], np.log10(inertias[mask_l0]), k=1)
    for l in [0, 1, 2]:
        mask = tgt_model.modes['l'] == l
        freq = tgt_model.get_freq()[mask] * tgt_model.glb[model.ifreq_ref]

        ax1.scatter(freq, (freq + nu_offset) % Dnu_c, c=f'C{l}',
                    marker=marker[l], label=fr'$\ell = {l}$', zorder=10, edgecolors='k')
        ax1.scatter(freq, (freq + nu_offset) % Dnu_c + Dnu_c,
                    c=f'C{l}', marker=marker[l], zorder=10, edgecolors='k')
        if norm_inertia:
            ax2.scatter(freq, tgt_model.modes['inertia'][mask] / 10 ** ip_l0(freq), c=f'C{l}', marker=marker[l],
                        zorder=10, edgecolors='k')
        else:
            ax2.scatter(freq, tgt_model.modes['inertia'][mask], c=f'C{l}',
                        marker=marker[l],
                        zorder=10, edgecolors='k')
        if l == 0:
            for i in range(len(freq)):
                ax1.axvline(freq[i], 0, 1, color='grey', ls=':')
                ax2.axvline(freq[i], 0, 1, color='grey', ls=':')
                ax3.axvline(freq[i], 0, 1, color='grey', ls=':')
    numax = tgt_model.numax
    sigma = 0.66 * numax ** 0.88 / (2 * np.sqrt(2 * np.log(2)))
    for i in [1, 2]:
        ax1.axvspan(numax - i * sigma, numax + i * sigma, alpha=0.2, zorder=-1)
        ax2.axvspan(numax - i * sigma, numax + i * sigma, alpha=0.2, zorder=-1)
        ax3.axvspan(numax - i * sigma, numax + i * sigma, alpha=0.2, zorder=-1)
    ax1.axvline(numax, 0, 1, ls='--', c='k', label=r'$\nu_\mathrm{max}$')
    ax2.axvline(numax, 0, 1, ls='--', c='k')
    ax3.axvline(numax, 0, 1, ls='--', c='k')
    ax1.axvline(tgt_model.cutoff, 0, 1, ls='--', c='r', label=r'$\nu_\mathrm{cut}$')
    ax2.axvline(tgt_model.cutoff, 0, 1, ls='--', c='r')
    ax2.set_yscale('log')
    Dnu_c = tgt_model.find_surface_corrected_large_separation(a_surf) * tgt_model.glb[model.ifreq_ref]
    ax1.set_ylim(0, Dnu_c * 1.2)
    ax1.axhline(Dnu_c, color='k', ls=':')
    ax1.set_ylabel(fr'$\nu\; \mathrm{{mod}}\; {Dnu_c:.2f} \; (\mu$Hz)')
    if norm_inertia:
        ax2.set_ylabel(r'Inertia/Inertia$_{\ell0}$')
    else:
        ax2.set_ylabel('Inertia')
    if norm_DP:
        ax3.set_ylabel('Normalized DP')
    else:
        ax3.set_ylabel('DP')
    ax2.set_xlabel(r'$\nu \; (\mu$Hz)')
    labels, handles = list(
        zip(*{label: handle for handle, label in list(zip(*ax1.get_legend_handles_labels()))}.items()))
    pu.top_legend(ax1)
    f.suptitle(tgt_model.name.split('/')[-1])
    ax1.axes.xaxis.set_visible(False)
    f.set_constrained_layout(True)

    return f, axes


def check_model(inputs):
    if len(inputs) == 2:
        i_grid, i_track = inputs
    elif len(inputs) == 3:
        i_grid, i_track, tmpdir = inputs
    amodel = get_model(grid, i_grid, i_track)
    Dnu0 = amodel.find_large_separation()
    max_Dnu = args.Dnu_thresh * Dnu0

    Ong_regime = amodel.glb[model.user_params_index['lgRhoc']] - np.log10(
                        amodel.glb[model.imass] / (4 * np.pi / 3 * amodel.glb[model.iradius] ** 3)) >= 8
    if Ong_regime:
        return i_grid, i_track, 0

    mask_l0 = amodel.modes['l'] == 0
    freq_min = amodel.modes['freq'][mask_l0].min()
    freq_max = amodel.modes['freq'][mask_l0].max()
    amodel.modes = amodel.modes[(amodel.modes['freq'] >= freq_min) & (amodel.modes['freq'] <= freq_max)]
    mask_l0 = amodel.modes['l'] == 0
    masks = {}
    for l in args.ell:
        mask = amodel.modes['l'] == l
        masks[l] = mask

    # Check for large variation in Delta nu
    # for l in args.ell:
    #     mask = masks[l]
    #     failed = np.any(np.diff(amodel.modes['freq'][mask]) > max_Dnu)
    #     if failed:
    #         return i_grid, i_track, 1

    # Check for high mode inertia
    # ip_l0 = ip.make_interp_spline(amodel.modes['freq'][mask_l0], amodel.modes['inertia'][mask_l0], k=1)
    # for l in args.ell:
    #     mask = masks[l] & (amodel.modes['freq'] >= min(amodel.modes['freq'][mask_l0])) & (amodel.modes['freq'] <= max(amodel.modes['freq'][mask_l0]))
    #     if sum(mask) == 0:
    #         return i_grid, i_track, 4
    #     norm_inertia = amodel.modes['inertia'][mask] / ip_l0(amodel.modes['freq'][mask])
    #     failed = np.any(norm_inertia/np.mean(norm_inertia) > 4)
    #     if failed:
    #         return i_grid, i_track, 2

    if not args.no_gyre and amodel.string_to_param('DPg.mod') > 0:
        # Check for too-large difference in period in gyre output for non-Ong models

        if os.path.exists(f'gyre_out/{amodel.name.split("/")[1]}.profile.GYRE.sgyre_l'.replace('.n', '_n')):
            gs = ld.GyreSummary(f'gyre_out/{amodel.name.split("/")[1]}.profile.GYRE.sgyre_l'.replace('.n', '_n'))
        else:
            gs = ld.GyreSummary(f'{tmpdir}/{amodel.name.replace(".n", "_n")}.profile.GYRE.freql012')
        fmin, fmax = (gs.data['Re(freq)'][gs.data['l'] == 0])[[0, -1]]
        gs.data = gs.data[(gs.data['Re(freq)'] >= fmin) & (gs.data['Re(freq)'] <= fmax)]

        amodel.read_file_CLES(f'{args.freq_files_base}/{amodel.name}.freq')
        amodel.modes['freq'] /= amodel.glb[model.ifreq_ref]
        amodel.modes = amodel.modes[(amodel.modes['freq'] >= freq_min) & (amodel.modes['freq'] <= freq_max)]

        for l in args.ell:
            mask_gs = gs.data.l == l
            # freqs = gs.data['Re(freq)'][mask_gs]
            mask = amodel.modes['l'] == l
            freqs = amodel.glb[model.ifreq_ref]*amodel.modes['freq'][mask]
            DPnorm = -np.diff(1e6 / freqs) * np.sqrt(l * (l + 1)) / (
                        amodel.string_to_param('DPg.mod') * np.sqrt(2))
            failed = np.any(DPnorm > 1.75)
            # if failed:
            #     return i_grid, i_track, 3

            n_p = gs.data.n_p[mask_gs]
            n_g = gs.data.n_g[mask_gs]
            n_pg = gs.data.n_pg[mask_gs]
            if l == 1:
                mask_ng = n_p < n_g
                have_missing = not (np.all(np.diff(n_pg[mask_ng]) == 1) and np.all(np.diff(n_pg[~mask_ng]) == 1))
            else:
                have_missing = not np.all(np.diff(n_pg) == 1)
            if have_missing:
                return i_grid, i_track, 5

    return i_grid, i_track, 0


def check_track(i_grid):
    n_track = len(grid.tracks[i_grid].names)

    out = []
    if args.no_gyre:
        for i_track in range(n_track):
            out.append(check_model((i_grid, i_track, None)))
        return out

    base_name = grid.tracks[i_grid].names[0].split('/')[0]
    mass, ovh, aFe, FeH, DYDZ, Y0 = split_name(base_name)
    Y_aFe_dir = 'Y' + base_name[50:52] + base_name[27:29]
    tarfile_path = f'{args.freq_files_base_tgz}/{Y_aFe_dir}/{base_name}.tar.gz'
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tarfile_path, mode='r') as tf:
            tf.extractall(path=tmpdir)
        dirpath, dirnames, fnames = os.walk(tmpdir, topdown=False).__next__()
        shutil.move(dirpath, f'{tmpdir}')
        shutil.rmtree(f'{tmpdir}/storage_tracks')
        for i_track in range(n_track):
            out.append(check_model((i_grid, i_track, tmpdir)))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('grid_path', type=str,
                        help="Path to grid.")
    parser.add_argument('freq_files_base', type=str,
                        help='Base path to frequency files.')
    parser.add_argument('freq_files_base_tgz', type=str, default='',
                        help='Base path to frequency file tar balls.')
    parser.add_argument('--ell', type=str, default='012',
                        help='Which degree modes to check')
    parser.add_argument('--Dnu-thresh', type=float, default=1.7,
                        help='Maximum difference in frequency in units of Delta nu above which a mode is determined as'
                             'missing.')
    parser.add_argument('-n', type=int, default=0,
                        help='Number of worker processes. Defaults to nproc.')
    parser.add_argument('--no-gyre', action='store_const', const=True, default=False,
                        help="Skip frequency file check.")
    args = parser.parse_args()
    args.ell = sorted([int(_) for _ in args.ell if _ != '0'])
    if args.n == 0:
        args.n = mp.cpu_count()

    grid = load_grid(args.grid_path)

    num_models = np.sum([_.glb.shape[0] for _ in grid.tracks])
    grid_indeces = np.zeros((num_models, 2), dtype=int)
    k = 0
    for i, track in enumerate(grid.tracks):
        for j in range(len(track.glb)):
            grid_indeces[k] = (i, j)
            k += 1
    grid_indeces = grid_indeces[-1000:]
    num_models = len(grid_indeces)

    out = []

    with mp.Pool(args.n) as p, tqdm.tqdm(total=len(grid.tracks)) as pbar:
        for res in p.imap(check_track, np.arange(len(grid.tracks))):
            out.append(res)
            pbar.update(1)
    out = np.concatenate(out)

    out = out[np.lexsort([out[:, 1], out[:, 0]])]

    np.save(f'bad_modes_{os.path.basename(args.grid_path)}.npy', out[out[:,2] != 0])
    print(f'Need to rerun {sum(out[:,2] != 0)} of {len(out)} models.')
    print(np.bincount(out[:,2]))
