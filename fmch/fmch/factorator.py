#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This class contains methods for preprocessing and factorizing hyperspectral
imagery (HSI) with simplex volume maximisation (SiVM) [1], saving, loading and
visualizing results. For the factorization itself, the pymf module is used
(https://github.com/cthurau/pymf) with minor modifications (adapted to Python
3).

[1] Thurau, C., Kersting, K., & Bauckhage, C. (2010). Yes we can: simplex
volume maximization for descriptive web-scale matrix factorization. In Proc.
Int. Conf. on Information and Knowledge Management, 1785-1788.

Methods:
    load_new_roi(row_ix, wls_only, save): Read and crop HSI.
    load_full_meta(fdir): Load metadata from disc.
    preprocess_hsi_1(self, thresh0, thresh1, thresh2, adj_bands, skip_bad_px,
                     max_res_ix, save): Detect and remove bad, persistent and
        all-zero bands. Identify bad pixels. Resample to common resolution.
    load_pp1(fdir, no_imgs): Load intermediary results from disc.
    preprocess_hsi_2(save): Remove bad pixels, transform to 2D and concatenate
        for factorization.
    load_pp2(fdir, no_imgs): Load intermediary results from disc.
    factorize_hsi(n_arch, dist_meas, na_val, save): factorize preprocessed and
        transformed HSI with SiVM to generate archetype and coefficient matrix.
    load_coefs(fdir): Load factorization results from disc.
    classify_arch(idx_list): Classify archetypes and aggregate coefficient
        matrix accordingly.
    hs_vi(hsi, wls, vi_name): Calculate vegetation indices for evaluation of
        factorization results.
    export_fit_vars(self, vi_list, lag, save): Generate and export a DataFrame
        of factorization results and evaluation variables.
    rgb(clipq): Generate RGB representations of the preprocessed HSI.
    dlplot(ts, col, rgb, save): Visualize factorization results with double
        layer maps.
'''

__version__ = '0.1'
__author__ = 'Floris Hermanns'
__license__ = 'BSD 3-Clause'
__status__ = 'Prototype'

from geopandas import read_file
import numpy as np
import pandas as pd

import spectral.io.envi as envi
from pathlib import Path
from math import ceil, floor, pi
from scipy import signal
from skimage.transform import resize
import pickle
#import os
import json
from datetime import date
from pymf.sivm import SIVM

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.transforms as mtransforms
#import matplotlib.offsetbox as mosb
from colorsys import hls_to_rgb as htor 
from fmch.HSI2RGB import HSI2RGB

class Factorator():
    
    def __init__(self, wdir, meta, maskfile, do_mkdir = False):
        '''
        Args:
            wdir (str): The working directory where BSQ files are located.
            meta (pandas.DataFrame): Metadata of the hyperspectral image files.
                Must contain three columns: 'fnames' contains filenames without
                extension.
                Column 'site' could be used to differentiate between different
                research sites. However, processing of multiple sites is
                currently not implemented. Only used for output file names.
                Column 'time' should contain a string identifier for different
                time steps (can be year, date, etc.).
            do_mkdir (bool): If true, the output directory is created
                automatically (including parents).
        '''
        self._wdir = wdir
        # index is reset to prevent errors
        self.meta = meta.sort_values('time').reset_index(drop=True)
        if len(self.meta['site'].unique()) > 1:
            raise ValueError('Currently only 1 site can be processed at once')
        self.site = self.meta['site'].unique()[0]
        self._odir = wdir / ('out_' + self.site + '_' + str(date.today()))
        # pathlib.Path: Output directory is created inside wdir.
        if do_mkdir == True:
            self._odir.mkdir(parents=True, exist_ok=True)
        self._maskname = maskfile
        
        
    def _get_mask(self, wdir, maskname, _row_ix):
        '''
        From a prepared vector file (.gpkg) that contains bounding boxes of the
        ROI, the geometry corresponding to the chosen site is loaded as a
        gpd.GeoDataFrame.
        '''
        bounds = read_file(Path(wdir, maskname))
        self.meta['bounds'] = bounds.loc[_row_ix, 'geometry']
               
    def _crop_hsi(self, wdir, fname, mask):
        '''
        Crops hyperspectral images based on a vector geometry (gpd.GeoDataFrame).
        It returns a numpy 3D array (HSI) and the list of central wavelengths.
        '''
        bsq = envi.open(Path(wdir, fname + '.hdr'), Path(wdir, fname + '.bsq'))
        hdr = envi.read_envi_header(Path(wdir, fname + '.hdr'))
        wls = [float(i) for i in hdr['wavelength']] # read wavelengths (e.g. for plot labels)

        ext_s = tuple(np.unique(mask.exterior.coords.xy))
        self._crs = hdr['coordinate system string']
        
        # Calculate array indices (that correspond to subset coordinates)
        xmin0 = float(hdr['map info'][3])
        ymax0 = float(hdr['map info'][4])
        xsize = round(float(hdr['map info'][5]), 3) # 3 decimals for pixel size accuracy
        ysize = round(float(hdr['map info'][6]), 3)
        
        # vector extent is rounded for clipping rows & cols of image
        xmin = floor((ext_s[0] - xmin0) / xsize) # distance from xmin to xmin of subset
        xmax = xmin + ceil((ext_s[1] - ext_s[0]) / xsize)
        # y dimension is "upside down"! -> ymin is calculated as the distance from image bottom (ymax0)...
        # ...to subset bottom (ymax of ext_s object)
        ymin = floor((ymax0 - ext_s[3]) / ysize)
        # to calculate "ymax", the y extent of the subset is added to the previously calculated "ymin"
        ymax = ymin + ceil((ext_s[3] - ext_s[2]) / ysize)
        
        # Calculate geographic extent of subset
        xminA = xmin0 + xmin * xsize
        xmaxA = xminA + (xmax-xmin) * xsize
        ymaxA = ymax0 - ymin * ysize
        yminA = ymaxA - (ymax-ymin) * ysize
        extent_arr = (xminA, xmaxA, yminA, ymaxA)
        
        hsi = bsq[ymin:ymax, xmin:xmax, :]
        return hsi, wls, xsize, extent_arr # assuming that x- & y-res are equal
    
    def load_new_roi(self, row_ix = 0, wls_only = False, save = False):
        '''
        Reads hyperspectral image files (.bsq) from disk and crops the image
        cubes to the region of interest (ROI).
        
        Args:
            row_ix (int, optional): Index of the ROI geometry within the GPKG. It is
                assumed that the GPKG only contains one geometry: default is 0.
            wls_only (bool, optional): If True, will only load the wavelengths.
            save (bool, optional): If true, metadata and CRS is saved to disc.
        '''
        self.hsi_raw = []
        self.wls_raw = []
        ext = []
        sr = []
        self._get_mask(self._wdir, self._maskname, _row_ix = row_ix)
        for i,fname in enumerate(self.meta['fnames']):
            hsi, wls, sr_, ext_ = self._crop_hsi(self._wdir, fname, self.meta['bounds'][i])
            if wls_only == False:
                self.hsi_raw.append(hsi)
                '''list of np.ndarrays : HSI cropped to the ROI'''
            self.wls_raw.append(wls)
            ext.append(tuple([round(x,5) for x in ext_]))
            sr.append(sr_)
        self.meta['ext_subset'] = ext
        self.meta['sres'] = sr
        if save == True:
            self.meta.to_pickle(Path(self._odir, self.site + '_meta.pkl'))
            with open(Path(self._odir, self.site + '_crs.pkl'), 'wb') as f:
                pickle.dump(self._crs, f)
        return
    
    
    def load_full_meta(self, fdir, row_ix = 0):
        '''
        Load metadata, crs and vector mask from the ROI initialization from
        disc. Requires file nomenclature as specified in the Factorator class.
        
        Args:
            fdir (str): Directory to read files from.
            row_ix (int, optional): Index of the ROI geometry within the GPKG.
                Default assumption: the GPKG only contains one geometry.
        '''
        self.meta = pd.read_pickle(Path(fdir, self.site + '_meta.pkl'))
        with open(Path(fdir, self.site + '_crs.pkl'), 'rb') as f:
            self._crs = pickle.load(f)
        self._get_mask(self._wdir, self._maskname, _row_ix = row_ix)
        return
            
    
    def _detect_bad_band(self, hsi, I, thresh0 = 0.8):
        '''
        A hyperspectral data preprocessing function to detect noisy bands.
        Two adjacent bands of a HSI tend to have very high correlation. The
        bands that are not similar to their neighbors may contain a high level
        of noise (Cai et al. 2007). Returns an index to remove noisy bands from
        the HSI arrays.
        
        Note:
            if multiple HSI are used in SiVM, the maximum number of bands will
            be removed from all datasets!
        '''
        pair_cor = []
        for i in range(hsi.shape[2]-1): # -1: to account for band pairs being analyzed
            bands = np.column_stack((hsi[:,:, i].flatten(), hsi[:,:, i+1].flatten()))
            pair_cor.append(np.corrcoef(bands, rowvar=False)[1,0]) # use corrcoef to compare 2 bands as 2 variables
        pair_cor = np.array(pair_cor)
        idx = np.where(pair_cor < thresh0)[0].tolist()
        if any(pair_cor < thresh0) == False:
            print('Dataset {}: No bad bands found'.format(self.meta['fnames'][I]))
        else:
            print('Dataset {}: The band pairs {} are noisy'.format(self.meta['fnames'][I], idx))
        return idx
    
    def _roll_ptp(self, hsi, window, step):
        '''
        A helper function that performs a rollapply of numpy's ptp function along
        a 1D numpy array. ptp (peak to peak) calculates the range
        of values within the moving windows and returns a 1D array that is shorter
        than the input array (edge effect). Pad values at the beginning if needed.
        '''
        nrows = (hsi.size - window)//step + 1
        n = hsi.strides[0]
        s = np.lib.stride_tricks.as_strided(hsi, shape=(nrows, window), strides=(step*n, n))
        return s.ptp(1)
    
    def _detect_persistent_and_zero_px(self, hsi, I, window=20, step=1):
        '''
        A function to detect pixels with persistent and all-zero values along
        the band dimension. Returns a 2D boolean array to mask problematic pixels
        before resampling.
        '''
        # 1. Persistence test
        pmask = np.full((hsi.shape[:2]), False) # create all-false array
        for ij in np.ndindex(hsi.shape[:2]): # loop over pixel spectra
            # pad extracted spectra with nan values before performing rollapply ptp
            hsi1d = np.pad(hsi[ij], (window-1,0), 'constant', constant_values=np.nan)
            ''' threshold of absolute value range below which values are deemed
            persistent must be very low as adjacent bands can be extremely similar'''
            pmask1d = self._roll_ptp(hsi1d, window, step) < 0.000001
            pmask[ij] = pmask1d.any()
        # if any window of a certain length (e.g. 20) is deemed persistent, the pixel is excluded
        # 2. All-zero pixels
        hsi_zeros = hsi == 0     
        zmask = np.all(hsi_zeros, axis=2)
        full_mask = np.logical_or(pmask, zmask)
        if np.count_nonzero(full_mask) > 0:
            print('Dataset {}: {} persistent & all-zero pixels found before resampling'.format(self.meta['fnames'][I], np.count_nonzero(full_mask)))
        else:
            print('Dataset {}: No persistent & all-zero pixels found before resampling'.format(self.meta['fnames'][I]))
        return full_mask
    
    def _detect_bad_px(self, hsi, I, thresh1 = 0.8, thresh2 = 1.9, adj_bands = 6):
        '''
        A hyperspectral data preprocessing function to detect bad pixels.
        Using two threshold values, bad pixel candidates are selected and
        validated using values from the same pixel in adjacent bands. Thus,
        the accidental removal of normal pixels with large reflectance can be
        avoided (Cai et al. 2007). Returns a 2D mask that allows to remove
        problematic pixels before factorization (all values along the band
        dimension are removed).
        '''
        w = hsi.shape[2]
        bad_px_coords = []
        mask = np.full((hsi.shape[:2]), False)
        
        for j in range(w): # band loop
            band = hsi[:,:, j]/np.amax(hsi[:,:, j])
            band_pxs = np.where(band >= thresh1) # locations of bad pixel candidates
            
            # the zero-filling circumvents border issues (only valid indices)
            adj0 = j-adj_bands
            adj0 = 0 if adj0 <= 0 else adj0 # "ternary conditional operation" = 1-line if else
            adj1 = j+adj_bands
            adj1 = w if adj1 >= w else adj1 # ensures correct indexing for adjacent bands
            
            for i,(x,y) in enumerate(zip(band_pxs[0], band_pxs[1])): # pixel loop
                prob = hsi[x, y, j]
                # get max values of same pixel in +-10 adjacent bands
                bench = np.amax(hsi[x, y, list(range(adj0, j)) + list(range(j+1, adj1))])
                if prob/bench >= thresh2:
                    bad_px_coords.append((x,y))
        if bad_px_coords:
            coords = [tuple(x) for x in np.unique(bad_px_coords, axis=0)] # remove redundant coords
            for coord in coords:
                mask[coord[0], coord[1]] = True
            print('Dataset {}: {} bad pixel locations were found'.format(self.meta['fnames'][I], len(coords)))
        else:
            print('Dataset {}: No bad pixels were found'.format(self.meta['fnames'][I]))
        return mask
    
    def _resample(self, hsi, xy_res):
        '''
        Uses resize() from skimage.transform to interpolate HSI using cubic
        splines to align the spatial resolution before factorization.
        '''
        hsi_R = resize(hsi, (xy_res[0], xy_res[1], len(self.wls[0])), order=3) # order=3 -> bicubic
        hsi_R[hsi_R < 0] = 0
        return hsi_R

    def preprocess_hsi_1(self, thresh0 = 0.8, thresh1 = 0.8, thresh2 = 1.9,
                         adj_bands = 6, skip_bad_px = True, max_res_ix = None,
                         save = False):
        '''
        First HSI preprocessing method. Applies detection of bad bands, and bad,
        persistent & all-zero pixels. Bad bands and problematic pixels are re-
        moved before interpolation. Also, problematic pixel masks are saved for
        later use. Imagery is resampled to the highest spatial resolution of
        input HSI using bi-cubic interpolation.
        
        Args:
            thresh0 (float): Value below which the Pearson correlation
                coefficient of a band pair is deemed poor.
            thresh1 (float): If pixel P value > thresh1 * maximum pixel value
                of a given HSI band, then P is a bad pixel candidate.
            thresh2 (float): If candidate pixel P exceeds the maximum value of
                the identical pixels in adjacent bands by thresh2 * P value,
                then P is deemed a bad pixel.
            adj_bands (int): Number of adjacent bands for bad pixel validation.
            skip_bad_px (bool, optional): If true, bad pixel detection is
                skipped (saves computation time if analysis is repeated and no
                bad px were found before).
            max_res_ix (int, optional): By default, all images are resampled
                to the resolution of the max. resolution image. If you want to
                change this behaviour, you can use this arg to use a specific
                metadata row index as target resolution object.
            save (bool, optional): If true, preprocessed HSI, pixel masks and
                the updated wavelength lists are saved to disc.
        '''
        nf = len(self.meta)
        self.hsi_pp0 = [0]*nf # dummy list
        '''list of np.ndarrays: HSI after removal of bad bands.'''
        self.hsi_pp1_R = [0]*nf
        '''list of np.ndarrays: HSI after resampling.'''
        self.wls = [0]*nf
        '''list of lists: HSI central wavelengths after bad band removal.'''
        bad_band_idx = []
        pz_mask = [0]*nf
        bad_px_mask = [0]*nf
        self.exclude_mask = [0]*nf
        '''list of np.ndarrays: 2D masks for removal of problematic pixels in
            the next preprocessing step.'''
        
        # REMOVE NOISY BANDS
        # Generate preproc indices for each dataset
        [bad_band_idx.append(self._detect_bad_band(cube, i, thresh0)) for i,cube in enumerate(self.hsi_raw)]
        del_band_idx = set([x for sl in bad_band_idx for x in sl])
        
        # assumption: all wls_raw are equal / come from one specific sensor setup.
        keep_band_idx = list(set(range(0, len(self.wls_raw[0]))) - del_band_idx)
        if not del_band_idx:
            print('No bad bands were found')
        else:
            print('Bands {} are noisy in at least one dataset and will be removed from ALL datasets'.format(del_band_idx))
        for i,cube in enumerate(self.hsi_raw): 
            self.hsi_pp0[i] = cube[:,:, keep_band_idx]
            self.wls[i] = [x for i,x in enumerate(self.wls_raw[i]) if i in keep_band_idx]
        
        ''' PERSISTENCE TEST
        Persistent (repeated almost zero values) and all-zero pixels must be
        converted to nan pixels BEFORE resampling to be excluded from the inter-
        polation
        '''
        for i,cube in enumerate(self.hsi_pp0):
            pz_mask[i] = self._detect_persistent_and_zero_px(cube, i, window=20)
            cube[pz_mask[i], :] = np.nan # mask values
        
        ''' RESAMPLING
        The following code is used for storing one target resolution (or
        better: extent in # of np raster cells) with the metadata. If the user
        supplies one specific image as target, the calculation is skipped.
        '''
        if max_res_ix:
            tres0 = self.hsi_pp0[max_res_ix].shape[:2]
        else:
            target_res = []
            for cube in self.hsi_pp0:
                target_res.append(cube.shape[:2])
            tres0 = max(target_res)

        for i,cube in enumerate(self.hsi_pp0):
            if tres0 > cube.shape[:2]:
                print('Dataset {} is resampled to target resolution {}'.format(self.meta['fnames'][i], tres0))
                self.hsi_pp1_R[i] = self._resample(cube, tres0)
            elif tres0 == cube.shape[:2]:
                print('Dataset {} is already at target resolution'.format(self.meta['fnames'][i]))
                self.hsi_pp1_R[i] = cube
            else:
            	raise ValueError('Resampling target resolution is smaller than the original resolution. Something went wrong!')
            # "Resampling the masks" - overwriting
            pz_mask[i] = np.all(np.isnan(self.hsi_pp1_R[i]), axis=2)
        
        # now overwrite tres from shape to resolution values
        if max_res_ix:
            self.meta['tres'] = self.meta['sres'][max_res_ix]
        else:
            self.meta['tres'] = self.meta.sres.min()
        
        '''EXCLUDE BAD PX & ZEROS MASK'''
        for i,cube in enumerate(self.hsi_pp1_R):
            if skip_bad_px == True:
                bad_px_mask[i] = np.full((cube.shape[0], cube.shape[1]), False)
            else:
                bad_px_mask[i] = self._detect_bad_px(cube, i, thresh1, thresh2, adj_bands)
            
            self.exclude_mask[i] = np.logical_or(bad_px_mask[i], pz_mask[i])
       
        if save == True:
            self.meta.to_pickle(Path(self._odir, self.site + '_meta.pkl'))
            [np.save(Path(self._odir, self.meta['fnames'][i] + '_pp1'), x) for i,x in enumerate(self.hsi_pp1_R)]
            [np.save(Path(self._odir, self.meta['fnames'][i] + '_exclude_mask'), x) for i,x in enumerate(self.exclude_mask)]
            with open(Path(self._odir, self.site + '_wls_list.pkl'), 'wb') as f:
                pickle.dump(self.wls, f, pickle.HIGHEST_PROTOCOL)
        return
    
    
    def load_pp1(self, fdir, no_imgs = False):
        '''
        Load results of the first preprocessing routine from disc. Requires
        file nomenclature as specified in the Factorator class.
        
        Args:
            fdir (str): Directory to read files from.
            wls_only (bool, optional): If true, only central wavelengths lists
                are loaded.
            no_imgs (bool, optional): If true, resampled HSI (intermediary
                results from PP1) are not loaded (save RAM).
        '''
        with open(Path(fdir, self.site + '_wls_list.pkl'), 'rb') as f:
            self.wls = pickle.load(f)
        self.exclude_mask = [np.load(Path(fdir, x + '_exclude_mask.npy')) for x in self.meta['fnames']]
        if no_imgs == True:
            return
        else:            
            self.hsi_pp1_R = [np.load(Path(fdir, x + '_pp1.npy')) for x in self.meta['fnames']]
            return
        
        
    def _filter_transp(self, hsi, mask):
        '''
        Use problematic pixel masks to remove those pixels before factorization.
        Masks are applied per time step and a lookup table is created for
        backtransformation of (2D) coefficient values from factorization. This
        allows to fill problematic pixels with a NA value after factorization.
        '''
        dims = mask.shape
        if dims != hsi.shape[0:2]:
            raise ValueError('The HSI array should have the same xy dimensions as the mask array!')
        px_val = []
        valid_px_coords = {}
        for i in range(dims[0]):
            for j in range(dims[1]):
                if mask[i,j] == False:
                    px_idx = len(px_val)
                    px_val.append(hsi[i,j,:].tolist())
                    valid_px_coords[px_idx] = (i,j)
        hsi_f_t = np.asarray(px_val).T
        return hsi_f_t, valid_px_coords
        
    def preprocess_hsi_2(self, save = False):
        '''
        Second HSI preprocessing method. Applies problematic pixel masks from
        precprocess_hsi_1() to remove pixels before factorization. Since HSI
        must be transformed to 2D (x*y, bands) for factorization, a lookup
        table of valid pixel values is created to enable backtransformation
        while masking problematic pixels as NA.

        Args:
            save (bool, optional): If true, preprocessed HSI, lookup table and
                the dimensions list are saved to disc.
        '''
        nf = len(self.hsi_pp1_R)
        self.hsi_pp2_tf = [0]*nf
        '''list of np.ndarrays: HSI after 2D transformation and removal of
            problematic pixels.'''
        self.backtransform_lookup = [0]*nf
        '''list of dicts: Lookup table for backtransformation of factorization
            results.'''
        self.wls_c = [0]*nf
        self.dims = [0]*nf
        '''list of lists: Concatenated central wavelength lists'''

        for i,cube in enumerate(self.hsi_pp1_R):
            self.hsi_pp2_tf[i], self.backtransform_lookup[i] = self._filter_transp(cube, self.exclude_mask[i])
            dims = list(np.shape(cube))[0:2] + list(np.shape(self.hsi_pp2_tf[i]))
            self.dims[i] = dims
            print('File {} filtered and transposed, resulting dims: {}'.format(self.meta['fnames'][i], dims))
        '''list of lists: Metadata about extents of each HSI in 2D/3D.
            Necessary for backtransformation.'''
        if save == True:
            [np.save(Path(self._odir, self.meta['fnames'][i] + '_pp2'), x) for i,x in enumerate(self.hsi_pp2_tf)]
            f = open(Path(self._odir, self.site + '_backtransform.json'),'w')
            json.dump(self.backtransform_lookup, f, indent=4)
            f.close()
            with open(Path(self._odir, self.site + '_dims_list.pkl'), 'wb') as f:
                pickle.dump(self.dims, f, pickle.HIGHEST_PROTOCOL)
        return

    def _jsonkeys2int(self, x):
        '''
        Helper function to convert JSON string keys into Python integer keys
        when importing dicts from JSON.
        '''
        if isinstance(x, dict):
                return {int(k):v for k,v in x.items()}
        return x

    def load_pp2(self, fdir, no_imgs = False):
        '''
        Load results of the second preprocessing routine from disc. Requires
        file nomenclature as specified in the Factorator class.
        
        Args:
            fdir (str): Directory to read files from.
            dims_only (bool, optional): If true, only HSI dimensions lists are
                loaded.
            no_imgs (bool, optional): If true, resampled HSI (intermediary
                results from PP2) are not loaded (save RAM).
        '''
        with open(Path(fdir, self.site + '_dims_list.pkl'), 'rb') as f:
            self.dims = pickle.load(f)
        with open(Path(fdir, self.site + '_backtransform.json'), 'r') as f:
            self.backtransform_lookup = json.load(f, object_hook=self._jsonkeys2int)
        if no_imgs == True:
            return
        else:            
            self.hsi_pp2_tf = [np.count_nonzero(Path(fdir, x + '_pp2.npy')) for x in self.meta['fnames']]
            return

    
    def factorize_hsi(self, n_arch = 30, dist_meas = 'l2', na_val = np.nan, save = False):
        '''
        HSI factorization routine. Applies the SiVM algorithm of Thurau et
        al. (2010) to the proprocessed, concatenated input images. Produces a
        base and coefficient matrix as results.
        
        Args:
            n_arch (int): factorization hyperparameter - number of archetypes
                to be calculated.
            dist_meas ('l2' | 'cosine' | 'l1' | 'kl'): 'l2' maximizes the
                volume of the simplex. More information in [1].
            na_val (int): value for filling invalid pixel values after back-
                transformation.
            save (bool, optional): If true, the resulting base and coef matrices
                are saved to disc.
        '''
        self.sivm_coef = [0]*len(self.meta)
        self.sivm_coef_kn = [0]*len(self.meta)
        '''list of np.ndarrays: Coefficient matrices after backtransformation
        and splitting into one matrix per input time step.'''
        hsi_c = np.concatenate(self.hsi_pp2_tf, axis = 1)
        lvm = SIVM(hsi_c, num_bases=n_arch, dist_measure=dist_meas)
        lvm.factorize()
        #os.system('spd-say "matrix factorization has finished"')
        coef = lvm.H
        self.sivm_base = lvm.W
        '''np.ndarray: The base or archetype matrix. Contains typical extreme
            reflectance spectra (archetypes).'''
        # Backtransform
        add = 0
        for i,dims in enumerate(self.dims):
            coef_3d = np.empty((dims[0], dims[1], n_arch))
            if i > 0: # not needed in first iteration
                add += self.dims[i-1][3] # always use dims[3] from iteration before!
            # Fill in valid pixel values
            coords = self.backtransform_lookup[i]
            for j in coords:
                x,y = coords[j][0], coords[j][1]
                coef_3d[x, y, :] = coef[:, j+add] # note the index adaptation!
            # Assign an NA value to all invalid pixels (no loop needed!)
            coef_3d[self.exclude_mask[i], :] = na_val
            self.sivm_coef_kn[i] = coef[:, add:self.dims[i][3]+add]
            self.sivm_coef[i] = coef_3d
        
        if save == True:
            [np.save(Path(self._odir, self.meta['fnames'][i] + '_coef'), x) for i,x in enumerate(self.sivm_coef)]
            [np.save(Path(self._odir, self.meta['fnames'][i] + '_coef_kn'), x) for i,x in enumerate(self.sivm_coef_kn)]
            np.save(Path(self._odir, self.site + '_base'), self.sivm_base)
        return


    def load_coefs(self, fdir):
        '''
        Load results of the factorization routine from disc. Requires
        file nomenclature as specified in the Factorator class.
        
        Args:
            fdir (str): Directory to read files from.
        '''
        self.sivm_coef = [np.load(Path(fdir, x + '_coef.npy')) for x in self.meta['fnames']]
        self.sivm_coef_kn = [np.load(Path(fdir, x + '_coef_kn.npy')) for x in self.meta['fnames']]
        self.sivm_base = np.load(Path(fdir, self.site + '_base.npy'))
        return
    
    
    def classify_arch(self, idx_list):
        '''
        Classify archetypes into background, healthy and stressed signatures.
        Classified archetypes are summed up to produce aggregated probabilities
        (aggregation property of Dirichlet-distributed variables). Adapt code
        to include other classes. The resulting metrics are abbreviated as HWSI
        (hyperspectral water stress index).

        Args:
            idx_list (list of lists): Contains classification indices. The
            order of the index sublists must be 1) background, 2) healthy and
            3) stressed.
        '''
        n = len(self.sivm_coef)
        self.hwsi_b = [0]*n
        self.hwsi_h = [0]*n
        self.hwsi_s = [0]*n
        if isinstance(idx_list, list):
            idx_h = idx_list[0]
            idx_s = idx_list[1]
            idx_b = idx_list[2]
        else:
            raise ValueError('idx_list must be a list of lists')
            
        for i,coef in enumerate(self.sivm_coef): # loop over coef 3d arrays (they are sorted by year)
            self.hwsi_b[i] = np.sum(coef[:,:, idx_b], axis=2)
            self.hwsi_h[i] = np.sum(coef[:,:, idx_h], axis=2)
            self.hwsi_s[i] = np.sum(coef[:,:, idx_s], axis=2)
        return
            
    
    def _fnv(self, values, target):
        '''
        A convenience function that finds the index of a value in a list closest
        to a target value. Can be used to select wavelengths of remote sensing
        sensor specifications that best meet the criteria of vegetation indices.
        '''
        if type(values) == list:
            idx = min(range(len(values)), key=lambda i:abs(values[i]-target))
            if target > max(values) or target < min(values): # filter validity to prevent returning indices whose values are not really "near" the target.
                raise ValueError('target value outside sensor range')
        elif type(values) == np.ndarray:
            idx = np.abs(values-target).argmin()
            if target > max(values) or target < min(values): # filter validity to prevent returning indices whose values are not really "near" the target.
                raise ValueError('target value outside sensor range')
        else:
            raise ValueError('wavelength values should be provided as list or np.ndarray')
        return idx
    
    def hs_vi(self, hsi, wls, vi_name):
        '''
        Calculation of various narrow- and broadband vegetation indices based
        on hyperspectral optical input data.
        
        Args:
            hsi (np.ndarray): A hyperspectral image cube. Dimensions must be
                defined as (x,y,bands). For comparison with factorization
                results, the resampled and quality-checked cubes from PP1
                should be used.
            wls (list of lists): list of the central wavelengths of the hyper-
                spectral sensor, adapted to the corrections from PP1
            vi_name (string): identifier of the vegetation index to be calcu-
                lated (see below).
        '''
        v = self._fnv
        r,c,_ = np.shape(hsi)
        x0 = np.linspace(-3,3,len(self.wls))
        dx = x0[1] - x0[0]
        if vi_name == 'ari1': # (1 / R550 - 1 / R700), reflectance values must be in [%]
            vi = np.squeeze( ( 1 / (100 * hsi[:,:,v(wls, 550)]) ) - ( 1 / (100 * hsi[:,:,v(wls, 700)]) ) )
        elif vi_name == 'cri1': # (1 / R510 - 1 / R550), reflectance values must be in [%]
            vi = np.squeeze( ( 1 / (100 * hsi[:,:,v(wls, 510)]) ) - ( 1 / (100 * hsi[:,:,v(wls, 550)]) ) )
        elif vi_name == 'pri': # (R531 - R570) / (R531 + R570)
            wl1 = hsi[:, :, v(wls, 531)]
            wl2 = hsi[:, :, v(wls, 570)]
            vi = np.squeeze( (wl1 - wl2) / (wl1 + wl2) )
        elif vi_name == 'msr': # (R750 - R445) / (R705 - R445)
            vi = np.squeeze( (hsi[:,:,v(wls, 750)] - hsi[:,:,v(wls, 445)]) / (hsi[:,:,v(wls, 705)] - hsi[:,:,v(wls, 445)]) )
        elif vi_name == 'rendvi': # (R750 - R705) / (R750 + R705)
            vi = np.squeeze( (hsi[:,:,v(wls, 750)] - hsi[:,:,v(wls, 705)]) / (hsi[:,:,v(wls, 750)] + hsi[:,:,v(wls, 705)]) )
        elif vi_name == 'chl_re':
            re1 = hsi[:,:,v(wls, 680):v(wls, 730)]
            nir = hsi[:,:,v(wls, 780):v(wls, 800)]
            re2 = hsi[:,:,v(wls, 755):v(wls, 780)]
            vi = np.squeeze( ((1 / np.sum(re1, axis=2)/re1.shape[2]) - (1 / np.sum(nir, axis=2)/nir.shape[2])) * np.sum(re2, axis=2)/re2.shape[2] )
        elif vi_name == 'mcari2': 
            wl1 = hsi[:, :, v(wls, 800)]
            wl2 = hsi[:, :, v(wls, 670)]
            wl3 = hsi[:, :, v(wls, 550)]
            vi = np.squeeze((1.5*(2.5*(wl1 - wl2) - 1.3*(wl1 - wl3))) / np.sqrt((2*wl1 + 1)**2 - (6*wl1 - (5*np.sqrt(wl2))) - 0.5))    
        elif vi_name == 'chl_opt': 
            re1 = hsi[:, :, v(wls, 680) : v(wls, 730)]
            nir = hsi[:, :, v(wls, 780) : v(wls, 800)]
            re2 = hsi[:, :, v(wls, 755) : v(wls, 780)]
            vi = np.squeeze(((1/ np.mean(re1, axis=2)) - (1/ np.mean(nir, axis=2))) * np.mean(re2, axis=2))
        elif vi_name == 'vog1':
            vi = np.squeeze(hsi[:, :, v(wls, 740)] / hsi[:, :, v(wls, 720)])
        elif vi_name == 'pri512':
            wl1 = hsi[:, :, v(wls, 531)]
            wl2 = hsi[:, :, v(wls, 512)]
            vi = np.squeeze((wl1 - wl2) / (wl1 + wl2))
        elif vi_name == 'pri_n': 
            wl1 = hsi[:, :, v(wls, 531)]
            wl2 = hsi[:, :, v(wls, 570)]
            wl3 = hsi[:, :, v(wls, 800)]
            wl4 = hsi[:, :, v(wls, 670)]
            wl5 = hsi[:, :, v(wls, 700)]
            vi = np.squeeze(((wl1 - wl2) / (wl1 + wl2)) / ((((wl3 - wl4) / (np.sqrt(wl3 + wl4)))) * (wl5 / wl4)))
        elif vi_name == 'ctr2':
            vi = np.squeeze(hsi[:, :, v(wls, 695)] / hsi[:, :, v(wls, 760)]) 
        elif vi_name == 'wi':
            vi = np.squeeze(hsi[:, :, v(wls, 970)] / hsi[:, :, v(wls, 900)])
        elif vi_name == 'wbi':
            vi = np.squeeze(hsi[:, :, v(wls, 900)] / hsi[:, :, v(wls, 970)])
        elif vi_name == 'nwi1':
            wl1 = hsi[:, :, v(wls, 970)]
            wl2 = hsi[:, :, v(wls, 900)]
            vi = np.squeeze((wl1 - wl2) / (wl1 + wl2))
        elif vi_name == 'nwi2':
            wl1 = hsi[:, :, v(wls, 970)]
            wl2 = hsi[:, :, v(wls, 850)]
            vi = np.squeeze((wl1 - wl2) / (wl1 + wl2))
        elif vi_name == 'car_opt':
            wl1 = hsi[:, :, v(wls, 510) : v(wls, 530)]
            wl2 = hsi[:, :, v(wls, 680) : v(wls, 730)]
            wl3 = hsi[:, :, v(wls, 760) : v(wls, 780)]
            vi = np.squeeze(((1/ np.mean(wl1, axis=2)) - (1/np.mean(wl2, axis=2))) * (np.mean(wl3, axis=2)))
        elif vi_name == 'msavi2':
            nir = np.mean(hsi[:, :, v(wls, 779.8) : v(wls, 885.8)], axis=2)
            red = np.mean(hsi[:, :, v(wls, 649.1) : v(wls, 680.1)], axis=2)
            vi = np.squeeze(( ((2*nir + 1) - np.sqrt(( (2*nir + 1)**2 - 8*(nir - red) ))) / 2))
        elif vi_name == 'deriv942':
            vi = np.empty((r, c))
            for (row,col),i in np.ndenumerate(vi):
                Zf = signal.savgol_filter(hsi[row,col,:], window_length=21, polyorder=2, deriv=1, delta=dx)
                vi[row,col] = Zf[v(wls, 942.5)]
        elif vi_name == 'deriv950':
            vi = np.empty((r, c))
            for (row,col),i in np.ndenumerate(vi):
                Zf = signal.savgol_filter(hsi[row,col,:], window_length=21, polyorder=2, deriv=1, delta=dx)
                vi[row,col] = Zf[v(wls, 950.5)]
            
        test1 = np.sum(np.isnan(vi))
        test2 = np.sum(np.isinf(vi))
        print('"{}" - NaN values: {}\nInf values: {}'.format(vi_name, test1, test2))
        return vi
    
    def export_fit_vars(self, vi_list, lag, save=False):
        '''
        Calculation of vegetation index (VI) values for evaluation of the
        stress metrics resulting from factorization with SiVM. After calcula-
        tion of VIs, the evaluation dataset and stress metrics are subsampled
        to avoid spatial autocorrelation. The samples are stored in a pandas
        DataFrame sorted by timestamp of input images. Can be used for cor-
        relation analysis and statistical inference (e.g. in R).
        
        Args:
            vi_list (list of strings): list of vegetation indices that should
                be calculated. identifiers must correspond to indices available
                in the 'hs_vi' function.
            lag (int): lag applied in x and y direction (in number of pixels)
                when subsampling to avoid spatial autocorrelation of variables.
                A suitable value can be determined beforehand via variogram
                analysis.
            save (bool, optional): If true, the pd.DataFrame containing sub-
                sampled vegetation index and HWSI response values is saved to
                disc.
        '''
        n = len(self.sivm_coef)
        imgs_all = [0]*n
        vars_all = [0]*n
        
        for i,hsi in enumerate(self.hsi_pp1_R): # loop fills imgs sublists per year/point in time
            imgs_all[i] = [self.hs_vi(hsi, self.wls[i], vi) for vi in vi_list] # LC loops over VIs
            imgs_all[i].insert(0, self.hwsi_s[i]) # fill in stress response
            imgs_all[i].insert(0, self.hwsi_h[i]) # fill in healthy response, new index 0
        
        for j,img_list in enumerate(imgs_all): # outer loop: list of input image lists (years/points in time)
            # 1 img_list contains both responses and VIs/derivatives for 1 point in time
            na_list = tuple([np.isnan(x) for x in img_list])
            mask01 = np.logical_or.reduce(na_list)
            mask02 = np.logical_or(self.hwsi_b[j]>0.4, mask01) # background filter: over 40% bg archetype prob.

            #rows, cols = self.dims[0][:2]
            rows, cols = img_list[0].shape # should be identical for all VIs!
            var_list = []
            for img in img_list:
                img_vals = []
                for row in np.arange(0, rows, step=lag): # two inner loops: loop over coordinates
                    for col in np.arange(0, cols, step=lag):
                        if mask02[row, col] == False:
                            img_vals.append(img[row, col])
                var_list.append(img_vals)
            var_list.insert(0, [int(self.meta['time'][j])]*len(var_list[-1]))
            vars_ = np.array(var_list).T
            vars_all[j] = pd.DataFrame.from_records(vars_)   
        vars_df = pd.concat(vars_all).round(8)
        vars_df.columns = ['year', 'hwsi_h', 'hwsi_s'] + vi_list
        vars_df.iloc[:, 0] = vars_df.iloc[:, 0].apply(np.int64)
        
        if save == True:
            vars_df.to_csv(Path(self._odir, self.site + '_fitvars.csv'), index=False, header=True)
        return vars_df

    def rgb(self, clipq = [0.001, 0.001]):
        '''
        Calculation of rgb representations of the hyperspectral imagery. Uses
        the "HSI2RGB" function from Magnusson et al. (2020) [2]
        
        Args:
            clipq (list of floats): thresholding quantiles for contrast
                increase.
        
        [2] Magnusson, M., Sigurdsson, J., Armannsson, S., Ulfarsson, M.,
        Deborah, H., & Sveinsson, J. (2020, July). ‘Creating RGB images from
        hyperspectral images using a color matching function. In 2020 IEEE
        International Geoscience and Remote Sensing Symposium (Vol. 7).
        '''
        self.rgbs = [0]*len(self.meta)
        for i,cube in enumerate(self.hsi_pp1_R):
            x,y,b = np.shape(cube)
            hsi = cube.copy()
            hsi[np.isnan(hsi)] = 0
            hsi = np.reshape(hsi, [-1,b], order='F')
  
            self.rgbs[i] = HSI2RGB(self.wls[i], hsi, y, x, 65, clipq[i])
            self.rgbs[i] = np.fliplr(np.rot90(self.rgbs[i], k=3, axes=(0,1))) 
        return


    def _draw_pie(self, ax, color, ratios=[0.4,0.3,0.3], X=0, Y=0, size=100):
        '''
        Helper function for dual layer visualization (currently not used).
        '''
        xy = []
        start = 0.
        for ratio in ratios:
            x = [0] + np.cos(np.linspace(2*pi*start,2*pi*(start+ratio), 30)).tolist()
            y = [0] + np.sin(np.linspace(2*pi*start,2*pi*(start+ratio), 30)).tolist()
            xy.append(zip(x,y))
            start += ratio
    
        for i, xyi in enumerate(xy):
            ax.scatter([X],[Y] , marker=list(xyi), s=size, facecolor=color[i])

    def dlplot(self, ts, col=['#00b4ff', '#ff006a','#f5ff00'], rgb=False,
               save=False):
        ''' 
        The "Double layer visualization" displays mixed-pixel information ef-
        fectively by overlaying Layer II (detail layer) on Layer I (background
        layer). Layer I displays the general spatial distribution of categories
        while Layer II displays the detailed composition of the pixels using
        pie charts. More information in [3]. In the current implementation,
        HWSI and RGB arrays are always cropped to a multiple of 11 so that some
        edge rows and columns might be removed.
        
        Note:
            returns a new extent tuple adjusted to the necessary cropping.
            Layout is optimized for saving the maps to disc on Ubuntu. Note
                that display will look quite different in an IDE plotting
                device, e.g. when using Spyder. Saving to disc is therefore
                recommended!
        
        Args:
            ts (str): the timestamp (year, date, etc.) of the factorized HSI to
                be mapped. Must correspond to a value from self.meta['time'].
            col (list): List of HTML color strings. Enter as many colors as the
                number of endmembers. The 3 default colors are optimized for
                color blindness and good distinction.
            rgb (list of np.ndarrays): RGB representation of the HSI. Can be
                calulated with the "rgb" function. If present, double layer
                plots will be drawn alongside 
            save (bool, optional): If true, the resulting visualization is
                saved to disk.
        
        [3] Cai, S., Du, Q., & Moorhead, R. J. (2007). Hyperspectral imagery
        visualization using double layers. IEEE Transactions on Geoscience and
        Remote Sensing, 45(10), 3028-3036.
        '''
        ix = self.meta.index[self.meta['time'] == ts][0]
        hwsi = np.concatenate((np.atleast_3d(self.hwsi_h[ix]),
                               np.atleast_3d(self.hwsi_s[ix]),
                               np.atleast_3d(self.hwsi_b[ix])), axis=2)
        # adjust dimensions for pie charts
        y,x,b = np.shape(hwsi)
        n_pies_y = floor(y/11) 
        rest_y = y%11  
        n_pies_x = floor(x/11) 
        rest_x = x%11   
        
        ext = list(self.meta.loc[ix, 'ext_subset'])
        res = self.meta.loc[ix, 'tres']
        # crop both layers
        hwsi2 = hwsi[ (0+rest_y):y, 0:(x-rest_x), : ]
        if rgb == True:
            rgbarr = self.rgbs[ix][ (0+rest_y):y, 0:(x-rest_x), :]
        ext[1] = ext[1]-rest_x*res
        ext[3] = ext[3]-rest_y*res
        ext0 = tuple(ext)
        '''Layer 1 - background'''
        hues = [197.508, 335.016, 60] 
        c1 = np.array([[x*255 for x in htor(hues[0]/360, 0.5, 1)],
                       [x*255 for x in htor(hues[1]/360, 0.5, 1)],
                       [x*255 for x in htor(hues[2]/360, 0.5, 1)]], dtype=np.uint8)   
        colos = tuple(map(tuple, c1/255))
        out = hwsi2.copy()
        for ij in np.ndindex(hwsi2.shape[:2]):
            out[ij] = np.round(hwsi2[ij] @ c1, 0)
        out = out.astype(np.int)
        '''Layer 2 - detail'''      
        pie = np.zeros((n_pies_y, n_pies_x, b))
        yx = np.zeros((n_pies_y, n_pies_x), dtype=object)
        for band in range(0, b):
            for i in range(0, n_pies_y):
                for j in range(0, n_pies_x):
                    pie[i,j,band] = np.nanmean(hwsi2[11*i:11*i+11, 11*j:11*j+11, band])
                    yx[i,j] = (11*i+5, 11*j+5) # centroid coords of pies
                    
        # Plotting Prep
        patch1 = mpatches.Patch(fc=colos[0], ec='darkgrey', lw=2, label='"Healthy" archetypes')
        patch2 = mpatches.Patch(fc=colos[1], ec='darkgrey', lw=2, label='"Stressed" archetypes')
        patch3 = mpatches.Patch(fc=colos[2], ec='darkgrey', lw=2, label='"Background" archetypes')
        ''' Background for north arrow can be activated if needed
        da = mosb.DrawingArea(18, 90, 0, 0)
        p = mpatches.Rectangle((0,0),18,90, edgecolor='w', facecolor='w') # dummy geometry
        da.add_artist(p)
        ab = mosb.AnnotationBbox(da, (x,y), xybox=(0, -42.5), # - offset for overlap with later annotation
                                 xycoords='axes fraction', boxcoords='offset points', arrowprops=dict(alpha=0)) # arrow not needed
        '''
        
        '''Plotting'''
        if rgb == True:
            x, y, arrow_length = .97, .98, .1 # north arrow placement, map units
            fig, axes = plt.subplots(2,1, figsize=(12,24), sharex=True)
            # plot RGB
            axes[0].imshow(rgbarr, extent=ext0)
            scalebar = AnchoredSizeBar(axes[0].transData + mtransforms.Affine2D().scale(1),
                                       50, '50 m', 'lower right', pad=0.2, borderpad=0.4, sep=6, color='w',
                                       frameon=False, label_top=True, size_vertical=2, fill_bar=True,
                                       fontproperties=fm.FontProperties(size=24, weight='regular'))
            axes[0].add_artist(scalebar)
            #axes[0].add_artist(ab) # add north arrow background
            axes[0].annotate('N', xy=(x, y), xytext=(x, y-arrow_length), c='w',
                             arrowprops=dict(ec='w', fc='w', width=8, headwidth=24, headlength=22, shrink=.05),
                             ha='center', va='center', fontsize=34, fontweight='regular',
                             xycoords='axes fraction', zorder=10) # draw in front

            plt.subplots_adjust(left=.05, right=.95, top=.99, bottom=.015, hspace=.025) # make room for labels
            
            # plot HWSI
            axes[1].imshow(out, extent=ext0)
            ax2 = axes[1].twinx() # dummy axes for legend with zorder = 1
            ax2.axis('off')

            for ax in axes:
                #ax.ticklabel_format(useOffset=False)
                ax.xaxis.set_major_locator(mticker.MultipleLocator(50)) # tick distance [m]
                ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
                for item in (ax.get_xticklabels() + ax.get_yticklabels() + [ax.yaxis.get_offset_text()]):
                    item.set_fontsize(18)
                for item in ([ax.xaxis.label, ax.yaxis.label]):
                    item.set_fontsize(18)
                ax.axis('tight') # removes unnecessary y-axis whitespace

            gs1 = fig.add_gridspec(nrows=n_pies_y, ncols=n_pies_x, left=.053, right=.947,
                                   top=.496, bottom=.016, wspace=.1, hspace=.1)
            for i in range(0, n_pies_y):
                for j in range(0, n_pies_x):
                    fig.add_subplot(gs1[i, j], aspect=1)
                    plt.pie([pie[i,j,0], pie[i,j,1], pie[i,j,2]], colors = col)
                    plt.axis('equal')
                    plt.margins(0, 0)

            # Additional map elements
            ax2.set_zorder(1) # draw in front
            ax2.legend(handles=[patch1, patch2, patch3], loc='lower right', fontsize=24,
                       facecolor='w', edgecolor='black', framealpha=1, borderpad=0.4)
            
            if save == True:  
                fig.savefig(Path(self._odir, self.site + '_dl+rgb_plot_' + ts + '.png'),
                                 overwrite = True) 
                    
        else:
            x, y, arrow_length = .04, .98, .1 # north arrow placement, map units
            fig = plt.figure(figsize=(12,12))
            ax1 = fig.add_axes([.05, .03, .9, .94])
            ax1.imshow(out, extent=ext0)
            ax2 = ax1.twinx() # dummy axes for legend with zorder = 1
            
            ax1.xaxis.set_major_locator(mticker.MultipleLocator(50)) # tick distance [m]
            ax1.yaxis.set_major_locator(mticker.MultipleLocator(50))
            for item in (ax1.get_xticklabels() + ax1.get_yticklabels() + [ax1.yaxis.get_offset_text()]):
                item.set_fontsize(16)
            for item in ([ax1.xaxis.label, ax1.yaxis.label]):
                item.set_fontsize(18)
            ax1.axis('tight')
            
            ''' alternative to gridspec approach (not recommended)
            for i in range(0, n_pies_y):
                for j in range(0, n_pies_x):
                    Y, X = yx[i,j]
                    self._draw_pie(ax1, col, pie[i,j,:].tolist(), X, Y, size=size) # size, e.g. 100
            '''
            gs1 = fig.add_gridspec(nrows=n_pies_y, ncols=n_pies_x, left=0.052, right=0.948,
                                   top=0.968, bottom=0.031, wspace=0.1, hspace=0.1)
            for i in range(0, n_pies_y):
                for j in range(0, n_pies_x):
                    fig.add_subplot(gs1[i, j], aspect=1)
                    plt.pie([pie[i,j,0], pie[i,j,1], pie[i,j,2]], colors = col)
                    plt.axis('equal')
                    plt.margins(0, 0)
                    
            ax2.set_zorder(1) # draw in front
            ax2.legend(handles=[patch1, patch2, patch3], loc='upper right', fontsize=22,
                       facecolor='w', edgecolor='black', framealpha=1, borderpad=0.4)

            ax2.annotate('N', xy=(x, y), xytext=(x, y-arrow_length), c='w',
                         arrowprops=dict(ec='w', fc='w', width=5, headwidth=15, headlength=13.5, shrink=0.05),
                         ha='center', va='center', fontsize=32, fontweight='bold', fontfamily='serif',
                         xycoords='axes fraction', zorder=10)
            sb = AnchoredSizeBar(ax1.transData + mtransforms.Affine2D(),
                                 50, '50 m', 'lower right', pad=0.4, borderpad=0.6, sep=6, color='w',
                                 frameon=False, label_top=True, size_vertical=2, fill_bar=True,
                                 fontproperties=fm.FontProperties(size=24, weight='semibold'))
            ax2.add_artist(sb)
            ax2.axis('off')
            
            if save == True:    
                fig.savefig(Path(self._odir, self.site + '_dl_plot_' + ts + '.png'),
                                 overwrite = True)
        return ext0
