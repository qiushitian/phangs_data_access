"""
Gathers all functions to estimate photometry
"""

import os
from pathlib import Path, PosixPath
import warnings

from astropy.nddata import Cutout2D
from astropy.units.quantity_helper.erfa import helper_pvu
from astropy.wcs import WCS
from astropy.io import ascii, fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip
from astropy.visualization.wcsaxes import SphericalCircle
from astropy import constants as const
from astroquery.simbad import Simbad

from pandas import read_csv

from phangs_data_access.phot_access import PhotAccess

speed_of_light_kmps = const.c.to('km/s').value
from scipy.constants import c as speed_of_light_mps
from scipy.spatial import ConvexHull

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.colorbar import ColorbarBase

from reproject import reproject_interp
import sep

import photutils
from photutils import datasets
from photutils import DAOStarFinder
from photutils import aperture_photometry
from photutils import CircularAperture, CircularAnnulus
from photutils import detect_sources, detect_threshold
from photutils.datasets import make_100gaussians_image
from photutils.centroids import centroid_quadratic
from photutils.profiles import RadialProfile


from photutils.aperture import SkyCircularAperture, SkyCircularAnnulus, CircularAnnulus, CircularAperture
from photutils.aperture import ApertureStats
from photutils.aperture import aperture_photometry
from photutils.utils import calc_total_error
from astropy.stats import SigmaClip
from photutils.background import SExtractorBackground, Background2D, MedianBackground


from astropy.stats import sigma_clipped_stats
import pandas as pd


import numpy as np

from phangs_data_access import phys_params, phangs_info, sample_access, helper_func

from phangs_data_access.dust_tools import DustTools

class PhotTools:
    """
    all functions related to photometry
    """

    @staticmethod
    def get_ap_corr(obs, band, target=None):
        if obs=='hst':
            return phys_params.hst_broad_band_aperture_4px_corr[target][band]
        elif obs == 'hst_ha':
            return phys_params.hst_ha_aperture_4px_corr[target][band]
        elif obs == 'nircam':
            return phys_params.nircam_aperture_corr[band]['ap_corr']
        elif obs == 'miri':
            return -2.5*np.log10(2)

    @staticmethod
    def get_ap_rad(obs, band, wcs):
        if (obs == 'hst') | ( obs == 'hst_ha'):
            return wcs.proj_plane_pixel_scales()[0].value * 3600 * 4
        if obs == 'nircam':
            return wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.nircam_aperture_corr[band]['n_pix']
        if obs == 'miri':
            return phys_params.miri_aperture_rad[band]

    @staticmethod
    def get_annulus_rad(obs, band=None, wcs=None):
        if (obs == 'hst') | ( obs == 'hst_ha'):
            return (wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.hst_bkg_annulus_radii_pix['in'],
                    wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.hst_bkg_annulus_radii_pix['out'])
        if obs == 'nircam':
            return (wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.nircam_bkg_annulus_radii_pix['in'],
                    wcs.proj_plane_pixel_scales()[0].value * 3600 * phys_params.nircam_bkg_annulus_radii_pix['out'])
        if obs == 'miri':
            return (phys_params.miri_bkg_annulus_radii_arcsec[band]['in'],
                    phys_params.miri_bkg_annulus_radii_arcsec[band]['out'])

    @staticmethod
    def extract_flux_from_circ_aperture_jimena(ra, dec, data, err, wcs, aperture_rad, annulus_rad_in, annulus_rad_out):
        mask = ((np.isinf(data)) | (np.isnan(data)) | (np.isinf(err)) | (np.isnan(err)))

        pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        coords_pix = wcs.world_to_pixel(pos)

        positions_sk_xp1 = wcs.pixel_to_world(coords_pix[0] + 1, coords_pix[1])
        positions_sk_xl1 = wcs.pixel_to_world(coords_pix[0] - 1, coords_pix[1])
        positions_sk_yp1 = wcs.pixel_to_world(coords_pix[0], coords_pix[1] + 1)
        positions_sk_yl1 = wcs.pixel_to_world(coords_pix[0], coords_pix[1] - 1)

        apertures = SkyCircularAperture(pos, aperture_rad * u.arcsec)
        apertures_xp1 = SkyCircularAperture(positions_sk_xp1, aperture_rad * u.arcsec)
        apertures_xl1 = SkyCircularAperture(positions_sk_xl1, aperture_rad * u.arcsec)
        apertures_yp1 = SkyCircularAperture(positions_sk_yp1, aperture_rad * u.arcsec)
        apertures_yl1 = SkyCircularAperture(positions_sk_yl1, aperture_rad * u.arcsec)

        annulus_aperture = SkyCircularAnnulus(pos, r_in=annulus_rad_in * u.arcsec, r_out=annulus_rad_out * u.arcsec)
        annulus_aperture_xp1 = SkyCircularAnnulus(positions_sk_xp1, r_in=annulus_rad_in * u.arcsec,
                                                  r_out=annulus_rad_out * u.arcsec)
        annulus_aperture_xl1 = SkyCircularAnnulus(positions_sk_xl1, r_in=annulus_rad_in * u.arcsec,
                                                  r_out=annulus_rad_out * u.arcsec)
        annulus_aperture_yp1 = SkyCircularAnnulus(positions_sk_yp1, r_in=annulus_rad_in * u.arcsec,
                                                  r_out=annulus_rad_out * u.arcsec)
        annulus_aperture_yl1 = SkyCircularAnnulus(positions_sk_yl1, r_in=annulus_rad_in * u.arcsec,
                                                  r_out=annulus_rad_out * u.arcsec)

        pixel_scale = wcs.proj_plane_pixel_scales()[0].value * 3600

        annulus_aperture_xy = CircularAnnulus(coords_pix, annulus_rad_in / pixel_scale, annulus_rad_out / pixel_scale)
        annulus_masks = annulus_aperture_xy.to_mask(method='exact')
        sigclip = SigmaClip(sigma=3.0, maxiters=5)

        phot = aperture_photometry(data, apertures, wcs=wcs, error=err, mask=mask)
        aper_stats = ApertureStats(data, apertures, wcs=wcs, error=err, sigma_clip=None, mask=mask)
        bkg_stats = ApertureStats(data, annulus_aperture, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                  sum_method='exact')
        # bkg_stats_2 = ApertureStats(data, annulus_aperture, error=err, wcs=w, sigma_clip=None, mask=mask)

        phot_xp1 = aperture_photometry(data, apertures_xp1, wcs=wcs, error=err, mask=mask)
        phot_xl1 = aperture_photometry(data, apertures_xl1, wcs=wcs, error=err, mask=mask)
        phot_yp1 = aperture_photometry(data, apertures_yp1, wcs=wcs, error=err, mask=mask)
        phot_yl1 = aperture_photometry(data, apertures_yl1, wcs=wcs, error=err, mask=mask)

        bkg_stats_xp1 = ApertureStats(data, annulus_aperture_xp1, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                      sum_method='exact')
        bkg_stats_xl1 = ApertureStats(data, annulus_aperture_xl1, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                      sum_method='exact')
        bkg_stats_yp1 = ApertureStats(data, annulus_aperture_yp1, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                      sum_method='exact')
        bkg_stats_yl1 = ApertureStats(data, annulus_aperture_yl1, error=err, wcs=wcs, sigma_clip=sigclip, mask=mask,
                                      sum_method='exact')

        bkg_median = bkg_stats.median
#         area_aper = aper_stats.sum_aper_area.value
#         # area_aper[np.isnan(area_aper)] = 0
#         area_sky = bkg_stats.sum_aper_area.value
# #         area_sky[np.isnan(area_sky)] = 0
#         total_bkg = bkg_median * area_aper
#
#         flux_dens = (phot['aperture_sum'] - total_bkg)
#
#         return flux_dens


        # bkg_median[np.isnan(bkg_median)] = 0
        #
        bkg_median_xp1 = bkg_stats_xp1.median
        # bkg_median_xp1[np.isnan(bkg_median_xp1)] = 0
        bkg_median_xl1 = bkg_stats_xl1.median
        # bkg_median_xl1[np.isnan(bkg_median_xl1)] = 0
        bkg_median_yp1 = bkg_stats_yp1.median
        # bkg_median_yp1[np.isnan(bkg_median_yp1)] = 0
        bkg_median_yl1 = bkg_stats_yl1.median
        # bkg_median_yl1[np.isnan(bkg_median_yl1)] = 0

        # bkg_10 = []
        # bkg_90 = []
        # bkg_10_clip = []
        # bkg_90_clip = []
        # N_pixels_annulus = []
        # N_pixel_annulus_clipped = []

        # N_pixels_aperture=[]

        # we want the range of bg values, to estimate the range of possible background levels and the uncertainty in the background
        annulus_data = annulus_masks.multiply(data)
        # print(annulus_data)
        # print(annulus_masks.data)
        if annulus_data is not None:
            annulus_data_1d = annulus_masks.multiply(data)[
                (annulus_masks.multiply(data) != 0) & (np.isfinite(annulus_masks.multiply(data))) & (
                    ~np.isnan(annulus_masks.multiply(data)))]
            if len(annulus_data_1d) > 0:
                # annulus_data=annulus_data[~np.isnan(annulus_data) & ~np.isinf(annulus_data)]
                annulus_data_filtered = sigclip(annulus_data_1d, masked=False)
                bkg_low, bkg_hi = np.quantile(annulus_data_1d,
                                              [0.1, 0.9])  # the 10% and 90% values among the bg pixel values
                bkg_low_clip, bkg_hi_clip = np.quantile(annulus_data_filtered, [0.1, 0.9])
                bkg_10 = bkg_low
                bkg_90 = bkg_hi
                bkg_10_clip = bkg_low_clip
                bkg_90_clip = bkg_hi_clip
                # annulus_data_1d = annulus_data[mask_an.data > 0]
                N_pixels_annulus = len(annulus_data_1d)
                # N_pixel_annulus_clipped = len(annulus_data_1d)-len(annulus_data_filtered)
            else:
                bkg_low = 0.
                bkg_hi = 0.
                bkg_10 = bkg_low
                bkg_90 = bkg_hi
                bkg_10_clip = 0.
                bkg_90_clip = 0.
                # annulus_data_1d = annulus_data[mask_an.data > 0]
                N_pixels_annulus = 0
        else:
            bkg_low = 0.
            bkg_hi = 0.  # the 10% and 90% values among the bg pixel values
            # bkg_low_clip, bkg_hi_clip = np.quantile(annulus_data_filtered, [0.1,0.9])
            bkg_10 = bkg_low
            bkg_90 = bkg_hi
            bkg_10_clip = 0
            bkg_90_clip = 0
            # annulus_data_1d = annulus_data[mask_an.data > 0]
            N_pixels_annulus = 0

        # bkg_10=0.1*bkg_stats_2.sum
        # bkg_90=0.9*bkg_stats_2.sum
        area_aper = aper_stats.sum_aper_area.value
        # area_aper[np.isnan(area_aper)] = 0
        area_sky = bkg_stats.sum_aper_area.value
        # area_sky[np.isnan(area_sky)] = 0
        total_bkg = bkg_median * area_aper
        total_bkg_xp1 = bkg_median_xp1 * area_aper
        total_bkg_xl1 = bkg_median_xl1 * area_aper
        total_bkg_yp1 = bkg_median_yp1 * area_aper
        total_bkg_yl1 = bkg_median_yl1 * area_aper

        total_bkg_10 = bkg_10 * area_aper
        total_bkg_90 = bkg_90 * area_aper
        total_bkg_10_clip = bkg_10_clip * area_aper
        total_bkg_90_clip = bkg_90_clip * area_aper

        bkg_std = bkg_stats.std
        # bkg_std[np.isnan(bkg_std)] = 0

        flux_dens = (phot['aperture_sum'] - total_bkg)

        flux_dens_xp1 = (phot_xp1['aperture_sum'] - total_bkg_xp1)
        flux_dens_xl1 = (phot_xl1['aperture_sum'] - total_bkg_xl1)
        flux_dens_yp1 = (phot_yp1['aperture_sum'] - total_bkg_yp1)
        flux_dens_yl1 = (phot_yl1['aperture_sum'] - total_bkg_yl1)

        flux_err_delta_apertures = np.sqrt(((flux_dens - flux_dens_xp1) ** 2 + (flux_dens - flux_dens_xl1) ** 2 + (
                    flux_dens - flux_dens_yp1) ** 2 + (flux_dens - flux_dens_yl1) ** 2) / 4.)

        flux_dens_bkg_10 = (phot['aperture_sum'] - total_bkg_10)
        flux_dens_bkg_90 = (phot['aperture_sum'] - total_bkg_90)
        flux_dens_bkg_10_clip = (phot['aperture_sum'] - total_bkg_10_clip)
        flux_dens_bkg_90_clip = (phot['aperture_sum'] - total_bkg_90_clip)

        flux_dens_err = np.sqrt(pow(phot['aperture_sum_err'], 2.) + (
                    pow(bkg_std * area_aper, 2) / bkg_stats.sum_aper_area.value) * np.pi / 2)
        # flux_dens_err_ir=np.sqrt(pow(phot['aperture_sum_err'],2.)+(pow(bkg_std*area_aper,2)/bkg_stats.sum_aper_area.value)*np.pi/2)/counts
        # sigma_bg times sqrt(pi/2) times aperture_area
        # phot_ap_error=np.sqrt(pow(phot['aperture_sum_err'],2.))/counts
        # err_bkg=np.sqrt((pow(bkg_std*area_aper,2)/bkg_stats.sum_aper_area.value)*np.pi/2)/counts
        # delta_90_10=(flux_dens_bkg_10 - flux_dens_bkg_90)
        # delta_90_10_clip=(flux_dens_bkg_10_clip - flux_dens_bkg_90_clip)
        flux_dens_err_9010 = np.sqrt(flux_dens_err ** 2 + (flux_dens_bkg_10 - flux_dens_bkg_90) ** 2)
        flux_dens_err_9010_clip = np.sqrt(flux_dens_err ** 2 + (flux_dens_bkg_10_clip - flux_dens_bkg_90_clip) ** 2)

        return flux_dens, flux_dens_err, flux_dens_err_9010, flux_dens_err_9010_clip, flux_err_delta_apertures

    @staticmethod
    def compute_phot_jimena(ra, dec, data, err, wcs, obs, band, aperture_rad=None, annulus_rad_in=None,
                            annulus_rad_out=None, target=None, gal_ext_corr=True):
        if aperture_rad is None:
            aperture_rad = PhotTools.get_ap_rad(obs=obs, band=band, wcs=wcs)

        if (annulus_rad_in is None) | (annulus_rad_out is None):
            annulus_rad_in, annulus_rad_out = PhotTools.get_annulus_rad(obs=obs, wcs=wcs, band=band)

        flux, flux_err, flux_err_9010, flux_err_9010_clip, flux_err_delta_apertures =\
            PhotTools.extract_flux_from_circ_aperture_jimena(
                ra=ra, dec=dec, data=data, err=err, wcs=wcs, aperture_rad=aperture_rad, annulus_rad_in=annulus_rad_in,
                annulus_rad_out=annulus_rad_out)
        if gal_ext_corr:
            fore_ground_ext = DustTools.get_target_gal_ext_band(target=target, obs=obs, band=band)
            flux *= 10 ** (fore_ground_ext / -2.5)

        return {'flux': flux, 'flux_err': flux_err, 'flux_err_9010': flux_err_9010,
                'flux_err_9010_clip': flux_err_9010_clip, 'flux_err_delta_apertures': flux_err_delta_apertures,}


    @staticmethod
    def compute_ap_corr_phot_jimena(target, ra, dec, data, err, wcs, obs, band):
        flux_dict = PhotTools.compute_phot_jimena(ra=ra, dec=dec, data=data, err=err, wcs=wcs, obs=obs, band=band,
                                                  target=target)
        aperture_corr = PhotTools.get_ap_corr(obs=obs, band=band, target=target)

        flux_dict['flux'] *= 10 ** (aperture_corr / -2.5)

        return flux_dict



    @staticmethod
    def extract_flux_from_circ_aperture_sinan(X, Y, image, annulus_r_in, annulus_r_out, aperture_radii):

        """
        This function was adapted to meet some standard ike variable naming. The functionality is untouched

        Calculate the aperture photometry of given (X,Y) coordinates

        Parameters
        ----------

        annulus_r_in:
            the inner radius of the annulus at which to calculate the background
        annulus_r_out: the outer radius of the annulus at which to calculate the background
        aperture_radii: the list of aperture radii at which the photometry will be calculated.
                        in units of pixels.
        """

        # SETTING UP FOR APERTURE PHOTOMETRY AT THE GIVEN X-Y COORDINATES
        print('Initializing entire set of photometric apertures...')
        'Initializing entire set of photometric apertures...'
        # begin aperture photometry for DAO detections
        # first set positions

        # print(X, Y)
        # positions = (X, Y)
        # positions = [(X, Y)]
        """The below line transforms the x & y coordinate list or single entries into the form photutils expects the input to be in."""
        positions = np.column_stack((X, Y))

        # then circular apertures
        apertures = [CircularAperture(positions, r=r) for r in aperture_radii]
        """Possibly no need, but may need to uncomment the below two lines in case 
        two different annuli need to be defined"""
        # then a single annulus aperture (for background) - Brad used 7-9 pixels
        # annulus_apertures_phangs = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_in)
        # another annulus aperture for the aperture correction
        annulus_apertures_ac = CircularAnnulus(positions, r_in=annulus_r_in, r_out=annulus_r_out)
        # need to subtract the smaller annulus_apertures_phangs from annulus_apertures_ac
        # finally make a mask for the annulus aperture
        annulus_masks = annulus_apertures_ac.to_mask(method='center')

        """To plot the mask, uncomment below"""
        # plt.imshow(annulus_masks[0])
        # plt.colorbar()
        # plt.show()

        # FOR REFERENCE DETECTION IMAGE... determine robust, sig-clipped  median in the background annulus aperture at each detection location
        bkg_median = []
        bkg_std = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(image)  # 25Feb2020 --  check whether the entire image is fed here
            annulus_data_1d = annulus_data[mask.data > 0]
            mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
            bkg_std.append(std_sigclip)
        bkg_median = np.array(bkg_median)
        bkg_std = np.array(bkg_std)

        # FOR REFERENCE DETECTION IMAGE... conduct the actual aperture photometry, making measurements for the entire set of aperture radii specified above
        # PRODUCING THE SECOND TABLE PRODUCT ASSOCIATED WITH DAOFIND (CALLED 'APPHOT' TABLE)
        #print('Conducting aperture photometry in progressive aperture sizes on reference image...')
        'Conducting aperture photometry in progressive aperture sizes on reference image...'
        apphot = aperture_photometry(image, apertures)
        # FOR REFERENCE DETECTION IMAGE... add in the ra, dec, n_zero and bkg_median info to the apphot result
        # apphot['ra'] = ra
        # apphot['dec'] = dec
        apphot['annulus_median'] = bkg_median
        apphot['annulus_std'] = bkg_std
        #apphot['aper_bkg'] = apphot['annulus_median'] * aperture.area

        for l in range(len(aperture_radii)):
            # FOR REFERENCE DETECTION IMAGE... background subtract the initial photometry
            apphot['aperture_sum_'+str(l)+'_bkgsub'] = apphot['aperture_sum_'+str(l)] - (apphot['annulus_median'] * apertures[l].area)

        #obj_list.append(np.array(apphot))

        """convert to pandas dataframe here - note that apphot.colnames & radii are local parameters """

        structure_data = np.array(apphot)
        print('Number of structures: ', structure_data.shape[0])

        structure_data_arr = np.zeros(shape=(structure_data.shape[0], len(apphot.colnames)))

        """Note that the majority of the operations around here are to convert the mildly awful astropy
        table format to a pandas dataframe"""

        for arr_x in range(structure_data.shape[0]):
            for arr_y in range(len(apphot.colnames)):
                structure_data_arr[arr_x][arr_y] = structure_data[apphot.colnames[arr_y]][arr_x]

        structure_df = pd.DataFrame(structure_data_arr, columns=apphot.colnames, dtype=np.float32)

        return structure_df


    @staticmethod
    def extract_flux_from_circ_aperture(data, wcs, pos, aperture_rad, data_err=None):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS``
        pos : ``astropy.coordinates.SkyCoord``
        aperture_rad : float
        data_err : ``numpy.ndarray``

        Returns
        -------
        flux : float
        flux_err : float
        """
        # estimate background
        bkg = sep.Background(np.array(data, dtype=float))
        # get radius in pixel scale
        pix_radius = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=aperture_rad, wcs=wcs, dim=1)
        # pix_radius_old = (wcs.world_to_pixel(pos)[0] -
        #               wcs.world_to_pixel(SkyCoord(ra=pos.ra + aperture_rad * u.arcsec, dec=pos.dec))[0])
        # print(pix_radius)
        # print(pix_radius_old)
        # exit()
        # get the coordinates in pixel scale
        pixel_coords = wcs.world_to_pixel(pos)

        data = np.array(data.byteswap().newbyteorder(), dtype=float)
        if data_err is None:
            bkg_rms = bkg.rms()
            data_err = np.array(bkg_rms.byteswap().newbyteorder(), dtype=float)
        else:
            data_err = np.array(data_err.byteswap().newbyteorder(), dtype=float)

        flux, flux_err, flag = sep.sum_circle(data=data - bkg.globalback, x=np.array([float(pixel_coords[0])]),
                                              y=np.array([float(pixel_coords[1])]), r=np.array([float(pix_radius)]),
                                              err=data_err)

        return float(flux), float(flux_err)

    @staticmethod
    def get_rad_profile_from_img(img, wcs, ra, dec, max_rad_arcsec, img_err=None, norm_profile=True):
        # get central pixels
        central_pos = wcs.world_to_pixel(SkyCoord(ra=ra*u.deg, dec=dec*u.deg))
        max_rad_pix = helper_func.CoordTools.transform_world2pix_scale(length_in_arcsec=max_rad_arcsec, wcs=wcs, dim=0)

        rad, profile, err =  PhotTools.get_rad_profile(data=img, x_pos=central_pos[0], y_pos=central_pos[1], max_rad=max_rad_pix, err=img_err)
        # change rad into arc seconds
        rad = helper_func.CoordTools.transform_pix2world_scale(length_in_pix=rad, wcs=wcs, dim=0)
        if norm_profile:
            err /=  np.nanmax(profile)
            profile /=  np.nanmax(profile)
        return rad, profile, err

    @staticmethod
    def get_rad_profile(data, x_pos, y_pos, max_rad, err=None, method='exact'):
        edge_radii = np.linspace(0, max_rad, 50)
        rp = RadialProfile(data, (x_pos, y_pos), edge_radii[1:], error=err, mask=None, method=method)
        return rp.radius, rp.profile, rp.profile_error

    # @staticmethod
    # def compute_photo_ew(wave_min_left_band, wave_max_left_band, wave_min_right_band, wave_max_right_band,
    #                      wave_min_narrow_band, wave_max_narrow_band, flux_left_band, flux_right_band, flux_narrowband):

    @staticmethod
    def compute_hst_photo_ew(target, left_band, right_band, narrow_band, flux_left_band, flux_right_band, flux_narrow_band, flux_unit='mJy'):
        # get the wavelength of both bands
        pivot_wave_left_band = helper_func.ObsTools.get_hst_band_wave(
            band=left_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=left_band),
            wave_estimator='pivot_wave', unit='angstrom')
        w_eff_left_band = helper_func.ObsTools.get_hst_band_wave(
            band=left_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=left_band),
            wave_estimator='w_eff', unit='angstrom')

        pivot_wave_right_band = helper_func.ObsTools.get_hst_band_wave(
            band=right_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=right_band),
            wave_estimator='pivot_wave', unit='angstrom')
        w_eff_right_band = helper_func.ObsTools.get_hst_band_wave(
            band=right_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=right_band),
            wave_estimator='w_eff', unit='angstrom')

        pivot_wave_narrow_band = helper_func.ObsTools.get_hst_band_wave(
            band=narrow_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=narrow_band),
            wave_estimator='pivot_wave', unit='angstrom')
        w_eff_narrow_band = helper_func.ObsTools.get_hst_band_wave(
            band=narrow_band, instrument=helper_func.ObsTools.get_hst_instrument(target=target, band=narrow_band),
            wave_estimator='w_eff', unit='angstrom')

        # now change from fluxes to flux densities
        flux_dens_left_band = helper_func.UnitTools.get_flux_unit_conv_fact(old_unit='mJy', new_unit='erg A-1 cm-2 s-1',
                                                                            pixel_size=None,
                                                                            band_wave=pivot_wave_left_band)
        flux_dens_right_band = helper_func.UnitTools.get_flux_unit_conv_fact(old_unit='mJy',
                                                                             new_unit='erg A-1 cm-2 s-1',
                                                                            pixel_size=None,
                                                                            band_wave=pivot_wave_right_band)
        flux_dens_narrow_band = helper_func.UnitTools.get_flux_unit_conv_fact(old_unit='mJy',
                                                                              new_unit='erg A-1 cm-2 s-1',
                                                                            pixel_size=None,
                                                                            band_wave=pivot_wave_narrow_band)

        # calculate the weighted continuum flux
        weight_left_band = (pivot_wave_narrow_band - pivot_wave_left_band) / (pivot_wave_right_band - pivot_wave_left_band)
        weight_right_band = (pivot_wave_right_band - pivot_wave_narrow_band) / (pivot_wave_right_band - pivot_wave_left_band)
        weighted_continuum_flux_dens = weight_left_band * flux_dens_left_band + weight_right_band * flux_dens_right_band


        print('flux_dens_left_band ', flux_dens_left_band)
        print('flux_dens_right_band ', flux_dens_right_band)
        print('flux_dens_narrow_band ', flux_dens_narrow_band)
        print('weighted_continuum_flux_dens ', weighted_continuum_flux_dens)


        # ew =

        exit()




























