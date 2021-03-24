#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Example script for factorization, archetype classification, evaluation
variable export and visualization.

Several input datasets are required: hyperspectral data has to be in .bsq
format (band sequential) with header file (.hdr) (can be adapted to other
requirements). A minimum of two hyperspectral datasets should be supplied.
Vector data for clipping HSI to the ROI must be supplied as geopackage (.gpkg)
'''

import pandas as pd
import numpy as np

from fmch.factorator import Factorator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path
plt.style.use('seaborn-paper')

#%% Data preparation (adapt as needed)
wdir0 = Path('path')
fdir0 = Path('resultspath')

# Create metadata table - 3 columns: filename, site name, timestamp
meta0 = pd.DataFrame(data = [['03_vnir_gsb_2018', 'gsb', '2018'], ['03_vnir_gsb_2019', 'gsb', '2019']], columns = ['fnames', 'site', 'time'])
''' Optional: create file list for many input files and adjust DF creation to your needs
fn0 = [x.stem for x in list((wdir0/'data').glob('*vnir*.bsq')) if '2017' not in x.stem]
meta1 = pd.DataFrame(data = [[fn, fn[8:11], fn[12:16]] for fn in fn0],
                     columns = ['fnames', 'site', 'time'])
'''
saving = False # results are saved in a wdir0 subfolder

#%% New factorization
gsb_factor = Factorator(wdir0, meta0, 'tereno_boundaries.gpkg', do_mkdir=saving) # init with metadata
gsb_factor.load_new_roi(row_ix=2, wls_only=False, save=saving) # set row_ix due to test GPKG with mult. geometries
gsb_factor.preprocess_hsi_1(skip_bad_px=False, save=saving) # remove bad bands, resampling
gsb_factor.preprocess_hsi_2(save=saving) # remove bad pixels, transform
gsb_factor.factorize_hsi(n_arch=30, na_val=np.nan, save=saving) # factorization

#%% Load results
'''
gsb_factor = Factorator(wdir0, meta0, 'tereno_boundaries.gpkg', do_mkdir=False)
gsb_factor.load_full_meta(fdir0, row_ix=2)
gsb_factor.load_pp1(fdir0)
gsb_factor.load_pp2(fdir0)
gsb_factor.load_coefs(fdir0)
'''

# After visual inspection: classify archetypes
idx_b = [0,2,4,6,7,8,10,13,14,15,21,22,24,26,27,29] # background
idx_s = [9,11,16,17,18,20,25] # stressed
idx_h = [1,3,5,12,19,23,28] # healthy
ixl = [idx_h, idx_s, idx_b]

gsb_factor.classify_arch(idx_list=ixl)

#%% Sample evaluation variables and export for inference

# List of names of used vegetation indices
vis = ['chl_opt', 'car_opt', 'rendvi', 'pri512', 'ctr2', 'mcari2', 'msavi2', 'wbi', 'deriv950']

# Generate and export variables for statistical inference
gsb_vars = gsb_factor.export_fit_vars(vi_list=vis, lag=25, save=saving) # lag can be determined by variogram analysis

#%% Double Layer visualization

gsb_factor.rgb(clipq = [0.0001, 0.001]) # calculate RGB representations

gsb_factor.dlplot('2018', save=False, rgb=True)
gsb_factor.dlplot('2019', save=False, rgb=True)

gsb_factor.dlplot('2018', save=False)
gsb_factor.dlplot('2019', save=False)

#%% Plot classified archetypes
ix_list = idx_h + idx_s
ix_list2 = [idx_h + idx_s, idx_b]

fig, axes = plt.subplots(1,2, figsize=(20,7), sharey=True)
pd.DataFrame(gsb_factor.sivm_base[:, idx_h]).set_index(pd.Index(gsb_factor.wls[0])).plot(
    ylim=(-0.02,0.75), ax=axes[0], color='#00b4ff', legend=None)
pd.DataFrame(gsb_factor.sivm_base[:, idx_s]).set_index(pd.Index(gsb_factor.wls[0])).plot(
    ylim=(-0.02,0.75), ax=axes[0], color='#ff006a', legend=None)
pd.DataFrame(gsb_factor.sivm_base[:, idx_b]).set_index(pd.Index(gsb_factor.wls[0])).plot(
    ylim=(-0.02,0.75), ax=axes[1], color='dimgrey', legend=None) # background archetypes in 2nd panel

custom_lines = [Line2D([0], [0], color='#00b4ff', lw=2), Line2D([0], [0], color='#ff006a', lw=2)]
axes[0].legend(custom_lines, ['Healthy archetypes', 'Stressed archetypes'], fontsize=20, loc='lower right')

[ax.axvline(x=685) for ax in axes]
[ax.set_xlabel('Wavelength [nm]') for ax in axes]
[ax.set_ylabel('Reflectance') for ax in axes]
[ax.set_title(['(a)', '(b)'][i], x=0.05, y=0.85, fontsize=24) for i,ax in enumerate(axes)]
[ax.grid() for ax in axes]
for ax in axes:
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    for item in ([ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(20)
fig.tight_layout()

fig.savefig(Path(gsb_factor._odir, gsb_factor.site + '_arch_.png'), overwrite = True, dpi=300)