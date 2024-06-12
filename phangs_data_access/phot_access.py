"""
Construct a data access structure for HST and JWST imaging data
"""
import os.path
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.constants import c as speed_of_light

from phangs_data_access import phangs_access_config, helper_func, phangs_info, phys_params


class PhotAccess:
    """
    Access class to organize data structure of HST, NIRCAM and MIRI imaging data
    """

    def __init__(self, target_name=None, target_ha_name=None):
        """
        In order to access photometry data one need to specify data path, versions and most important target names.
        For example NGC 628 has in HST a specification "c" for center or "e" for east.
        The HST broad band filter are also provided in mosaic versions however, this is not the case for H-alpha nor
        for NIRCAM or MIRI

        Parameters
        ----------
        target_name : str
            Default None. Target name
        target_name : str
            Default None. Target name used for Hs observation
        """

        # get target specifications
        # check if the target names are compatible
        if (target_name not in phangs_info.phangs_galaxy_list) & (target_name is not None):
            raise AttributeError('The target %s is not in the PHANGS photometric sample or has not been added to '
                                 'the current package version' % target_name)

        self.target_name = target_name

        # choose the best H-alpah target name of not provided
        if (target_ha_name is None) & (target_name is not None):
            # use the most common target names like central observations for ngc 628
            if target_name == 'ngc0628':
                target_ha_name = 'ngc0628c'

        self.target_ha_name = target_ha_name

        # loaded data dictionaries
        self.hst_bands_data = {}
        self.hst_ha_bands_data = {}
        self.nircam_bands_data = {}
        self.miri_bands_data = {}
        self.astrosat_bands_data = {}

        super().__init__()

    def get_hst_img_file_name(self, band, file_type='sci'):
        """

        Parameters
        ----------
        band : str
        file_type : str
            can be sci, err or wht
        Returns
        -------
        data_file_path : ``Path``
        """

        if band in phangs_info.hst_obs_band_dict[self.target_name]['acs_wfc1_observed_bands']:
            instrument = 'acs'
        elif band in phangs_info.hst_obs_band_dict[self.target_name]['wfc3_uvis_observed_bands']:
            instrument = 'uvis'
        else:
            raise KeyError(band, ' is not observed by HST for the target ', self.target_name)

        hst_data_folder = (Path(phangs_access_config.phangs_config_dict['hst_data_path']) /
                           helper_func.FileTools.target_names_no_zeros(target=self.target_name) /
                           (instrument + band.lower()))

        file_name = '%s_%s_%s_exp_drc_%s.fits' % (helper_func.FileTools.target_names_no_zeros(target=self.target_name),
                                                  instrument, band.lower(), file_type)

        return Path(hst_data_folder) / file_name

    def get_hst_ha_img_file_name(self):
        """
        hst H-alpha narrowband observation

        Returns
        -------
        data_file_path : ``Path``
        """

        if self.target_ha_name not in phangs_info.hst_ha_obs_band_dict.keys():
            raise LookupError(self.target_ha_name, ' has no H-alpha observation ')

        hst_data_folder = Path(phangs_access_config.phangs_config_dict['hst_ha_data_path'])

        file_name = ('%s_%s_%s_exp_drc_sci.fits' %
                     (helper_func.FileTools.target_names_no_zeros(target=self.target_ha_name),
                      helper_func.BandTools.get_hst_ha_instrument(target=self.target_ha_name),
                      helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name)))

        return Path(hst_data_folder) / file_name

    def get_hst_ha_cont_sub_img_file_name(self):
        """
        hst H-alpha continuum subtracted observation

        Returns
        -------
        data_file_path : ``Path``
        """

        if self.target_ha_name not in phangs_info.hst_ha_obs_band_dict.keys():
            raise LookupError(self.target_ha_name, ' has no H-alpha observation ')

        hst_data_folder = (Path(phangs_access_config.phangs_config_dict['hst_ha_cont_sub_data_path']) /
                           phangs_access_config.phangs_config_dict['hst_ha_cont_sub_ver'])

        if os.path.isfile(hst_data_folder /
                          ('%s_hst_ha.fits' % helper_func.FileTools.target_names_no_zeros(target=self.target_ha_name))):
            file_name = '%s_hst_ha.fits' % helper_func.FileTools.target_names_no_zeros(target=self.target_ha_name)
        elif os.path.isfile(hst_data_folder / ('%s_hst_%s_contsub.fits' %
                                               (helper_func.FileTools.target_names_no_zeros(target=self.target_ha_name),
                                                helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name)))):
            file_name = ('%s_hst_%s_contsub.fits' %
                         (helper_func.FileTools.target_names_no_zeros(target=self.target_ha_name),
                          helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name)))
        else:
            raise KeyError('No H-alpha continuum subtracted product found for ', self.target_ha_name)

        return Path(hst_data_folder) / file_name

    def get_jwst_img_file_name(self, instrument, band):
        """

        Parameters
        ----------
        instrument : str
        band : str

        Returns
        -------
        data_file_path : Path
        """

        nircam_data_folder = (Path(phangs_access_config.phangs_config_dict['%s_data_path' % instrument]) /
                              phangs_access_config.phangs_config_dict['%s_data_ver' % instrument] /
                              self.target_name)

        file_name = '%s_%s_lv3_%s_i2d_anchor.fits' % (self.target_name, instrument, band.lower())

        return Path(nircam_data_folder) / Path(file_name)

    def get_astrosat_img_file_name(self, band):
        """

        Parameters
        ----------
        band : str
        Returns
        -------
        data_file_path : Path
        """

        astrosat_data_folder = (Path(phangs_access_config.phangs_config_dict['astrosat_data_path']) /
                                phangs_access_config.phangs_config_dict['astrosat_data_ver'] /
                                'release')

        file_name = '%s_%s_bkg_subtracted_mw_corrected.fits' % (self.target_name.upper(), band[:-1].upper())

        return Path(astrosat_data_folder) / Path(file_name)

    def load_hst_band(self, band, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        file_name : str
        """
        # load the band observations
        if file_name is None:
            file_name = self.get_hst_img_file_name(band=band)
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)
        # rescale image to needed unit
        img_data *= helper_func.UnitTools.get_hst_img_conv_fct(img_header=img_header, img_wcs=img_wcs,
                                                               flux_unit=flux_unit)
        self.hst_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                    '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                    '%s_pixel_area_size_sr_img' % band:
                                        img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

        if load_err:
            err_file_name = self.get_hst_img_file_name(band=band, file_type='err')
            err_data, err_header, err_wcs = helper_func.FileTools.load_img(file_name=err_file_name)
            # rescale image to needed unit
            err_data *= helper_func.UnitTools.get_hst_img_conv_fct(img_header=err_header, img_wcs=err_wcs,
                                                                   flux_unit=flux_unit)
            self.hst_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                        '%s_wcs_err' % band: err_wcs, '%s_unit_err' % band: flux_unit,
                                        '%s_pixel_area_size_sr_err' % band:
                                            img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def load_hst_ha_band(self, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        load_err : bool
        flux_unit : str
        file_name : str
        """
        # load the band observations
        if file_name is None:
            file_name = self.get_hst_ha_img_file_name()
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)
        # rescale image to needed unit
        img_data *= helper_func.UnitTools.get_hst_img_conv_fct(img_header=img_header, img_wcs=img_wcs,
                                                               flux_unit=flux_unit)
        band = helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name)
        self.hst_ha_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                       '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                       '%s_pixel_area_size_sr_img' % band:
                                           img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
        if load_err:
            # TO DO: get errors !
            raise NotImplementedError('Uncertainties are not yet available for HST H-alpha observations')

    def load_hst_ha_cont_sub_band(self, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        load_err : bool
        flux_unit : str
        file_name : str
        """
        # load the band observations
        if file_name is None:
            file_name = self.get_hst_ha_cont_sub_img_file_name()
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)
        # rescale image to needed unit
        img_data *= helper_func.UnitTools.get_hst_img_conv_fct(img_header=img_header, img_wcs=img_wcs,
                                                               flux_unit=flux_unit)
        band = helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name)
        self.hst_ha_bands_data.update({'%s_cont_sub_data_img' % band: img_data,
                                       '%s_cont_sub_header_img' % band: img_header,
                                       '%s_cont_sub_wcs_img' % band: img_wcs,
                                       '%s_cont_sub_unit_img' % band: flux_unit,
                                       '%s_cont_sub_pixel_area_size_sr_img' % band:
                                           (img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg)})
        if load_err:
            # TO DO: get errors !
            raise NotImplementedError('Uncertainties are not yet available for HST H-alpha observations')

    def load_nircam_band(self, band, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        file_name : str

        """
        # load the band observations
        if file_name is None:
            file_name = self.get_jwst_img_file_name(instrument='nircam', band=band)
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number='SCI')

        img_data *= helper_func.UnitTools.get_jwst_conv_fact(img_wcs=img_wcs, flux_unit=flux_unit)

        self.nircam_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                       '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                       '%s_pixel_area_size_sr_img' % band:
                                           img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
        if load_err:
            err_data, err_header, err_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number='ERR')
            err_data *= helper_func.UnitTools.get_jwst_conv_fact(img_wcs=img_wcs, flux_unit=flux_unit)
            self.nircam_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                           '%s_wcs_err' % band: img_wcs, '%s_unit_err' % band: flux_unit,
                                           '%s_pixel_area_size_sr_err' % band:
                                               img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def load_miri_band(self, band, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        file_name : str

        """
        # load the band observations
        if file_name is None:
            file_name = self.get_jwst_img_file_name(instrument='miri', band=band)
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number='SCI')

        img_data *= helper_func.UnitTools.get_jwst_conv_fact(img_wcs=img_wcs, flux_unit=flux_unit)

        self.miri_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                     '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                     '%s_pixel_area_size_sr_img' % band:
                                         img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
        if load_err:
            err_data, err_header, err_wcs = helper_func.FileTools.load_img(file_name=file_name, hdu_number='ERR')
            err_data *= helper_func.UnitTools.get_jwst_conv_fact(img_wcs=img_wcs, flux_unit=flux_unit)
            self.miri_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                         '%s_wcs_err' % band: img_wcs, '%s_unit_err' % band: flux_unit,
                                         '%s_pixel_area_size_sr_err' % band:
                                             img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def load_astrosat_band(self, band, load_err=False, flux_unit='Jy', file_name=None):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        file_name : str
        """
        # load the band observations
        if file_name is None:
            file_name = self.get_astrosat_img_file_name(band=band)
        img_data, img_header, img_wcs = helper_func.FileTools.load_img(file_name=file_name)

        conversion_factor = helper_func.UnitTools.get_astrosat_conv_fact(img_wcs=img_wcs, band=band,
                                                                         flux_unit=flux_unit)
        img_data *= conversion_factor

        self.astrosat_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                         '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                         '%s_pixel_area_size_sr_img' % band:
                                             img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})
        if load_err:
            # the uncertainties are estimated to be approximately 5% in Hassani+2024 2024ApJS..271....2H
            self.astrosat_bands_data.update({'%s_data_err' % band: img_data * 0.05, '%s_header_err' % band: img_header,
                                             '%s_wcs_err' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                             '%s_pixel_area_size_sr_err' % band:
                                                 img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg})

    def get_phangs_obs_band_list(self):
        """
        assemble all observed photometry bands
        """
        band_list = []
        # HST broad band images
        band_list += helper_func.BandTools.get_hst_obs_band_list(target=self.target_name)
        # check if HST H-alpha observation is available:
        if helper_func.BandTools.check_hst_ha_obs(target=self.target_ha_name):
            hst_ha_band = helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name)
            band_list += [hst_ha_band]
            band_list += [hst_ha_band + '_cont_sub']
        # nircam
        band_list += helper_func.BandTools.get_nircam_obs_band_list(target=self.target_name)
        # miri
        band_list += helper_func.BandTools.get_miri_obs_band_list(target=self.target_name)
        # astrosat
        band_list += helper_func.BandTools.get_astrosat_obs_band_list(target=self.target_name)

        return band_list

    def load_phangs_bands(self, band_list=None, flux_unit='Jy', load_err=False, load_hst=True, load_hst_ha=True,
                          load_nircam=True, load_miri=True, load_astrosat=True):
        """
        wrapper to load all available HST, HST-H-alpha, NIRCAM and MIRI observations into the constructor
        This function checks if the band is already loaded and skips the loading if this is the case

        Parameters
        ----------
        band_list : list or str
        flux_unit : str
        load_err : bool
        load_hst: bool
        load_hst_ha: bool
        load_nircam: bool
        load_miri: bool
        load_astrosat: bool

        """
        # if only one band should be loaded
        if isinstance(band_list, str):
            band_list = [band_list]
        # if band list is none we get a list with all observed bands in order of wavelength
        if band_list is None:
            band_list = self.get_phangs_obs_band_list()

        # load bands
        for band in band_list:
            # check hst
            if band in (phangs_info.hst_obs_band_dict[self.target_name]['acs_wfc1_observed_bands'] +
                        phangs_info.hst_obs_band_dict[self.target_name]['wfc3_uvis_observed_bands']):
                # check if band is already loaded
                if ((('%s_data_img' % band) not in self.hst_bands_data) |
                        ((('%s_data_err' % band) not in self.hst_bands_data) & load_err)):
                    if load_hst:
                        self.load_hst_band(band=band, flux_unit=flux_unit, load_err=load_err)
                else:
                    continue
            # check hst H-alpha
            elif band in ['F657N', 'F658N']:
                # check if band is already loaded
                if (('%s_data_img' % band not in self.hst_ha_bands_data) |
                        (('%s__data_err' % band not in self.hst_ha_bands_data) & load_err)):
                    if load_hst_ha:
                        self.load_hst_ha_band(flux_unit=flux_unit, load_err=load_err)
                else:
                    continue
            # check hst H-alpha continuum subtracted
            # check hst H-alpha
            elif band in ['F657N_cont_sub', 'F658N_cont_sub']:
                # check if band is already loaded
                if (('%s_data_img' % band not in self.hst_ha_bands_data) |
                        (('%s_data_err' % band not in self.hst_ha_bands_data) & load_err)):
                    if load_hst_ha:
                        self.load_hst_ha_cont_sub_band(flux_unit=flux_unit, load_err=load_err)
                else:
                    continue
            # check nircam
            elif band in phangs_info.jwst_obs_band_dict[self.target_name]['nircam_observed_bands']:
                # check if band is already loaded
                if ((('%s_data_img' % band) not in self.nircam_bands_data) |
                        ((('%s_data_err' % band) not in self.nircam_bands_data) & load_err)):
                    if load_nircam:
                        self.load_nircam_band(band=band, flux_unit=flux_unit, load_err=load_err)
                else:
                    continue
            # check miri
            elif band in phangs_info.jwst_obs_band_dict[self.target_name]['miri_observed_bands']:
                # check if band is already loaded
                if ((('%s_data_img' % band) not in self.miri_bands_data) |
                        ((('%s_data_err' % band) not in self.miri_bands_data) & load_err)):
                    if load_miri:
                        self.load_miri_band(band=band, flux_unit=flux_unit, load_err=load_err)
                else:
                    continue
            # check astrosat
            elif band in phangs_info.astrosat_obs_band_dict[self.target_name]['observed_bands']:
                # check if band is already loaded
                if ((('%s_data_img' % band) not in self.astrosat_bands_data) |
                        ((('%s_data_err' % band) not in self.astrosat_bands_data) & load_err)):
                    if load_astrosat:
                        self.load_astrosat_band(band=band, flux_unit=flux_unit, load_err=load_err)
                else:
                    continue
            else:
                raise KeyError('Band is not found in possible band lists')

    def change_phangs_band_units(self, band_list=None, new_unit='MJy/sr'):
        """

        Parameters
        ----------
        band_list : list
        new_unit : str
        """
        if band_list is None:
            band_list = self.get_phangs_obs_band_list()

        for band in band_list:
            # check if band was loaded!
            if ('%s_img_cutout' % band) in (list(self.hst_bands_data.keys()) + list(self.hst_ha_bands_data.keys()) +
                                            list(self.nircam_bands_data.keys()) + list(self.miri_bands_data.keys()) +
                                            list(self.miri_bands_data.keys())):

                self.change_band_unit(band=band, new_unit=new_unit)

    def change_band_unit(self, band, new_unit='MJy/sr'):
        """
        will change loaded data to the needed unit. This will directly change all data saved in the constructor
        Parameters
        ----------
        band : str
        new_unit : str
            this can be :
            'mJy', 'Jy', 'MJy/sr' or 'erg A-1 cm-2 s-1'
        """

        # first we need to make sure what was the old unit and for which instrument.
        # Furthermore, we need the wavelength for some transformations
        if band in helper_func.BandTools.get_hst_obs_band_list(target=self.target_name):
            instrument = 'hst'
            band_wave = (
                helper_func.BandTools.get_hst_band_wave(band=band, instrument=helper_func.BandTools.get_hst_instrument(
                    target=self.target_name, band=band), unit='angstrom'))
        elif ((band == helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name)) |
              (band == (helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name) + '_cont_sub'))):
            instrument = 'hst_ha'
            band_wave = (
                helper_func.BandTools.get_hst_band_wave(
                    band=helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name),
                    instrument=helper_func.BandTools.get_hst_ha_instrument(target=self.target_ha_name),
                    unit='angstrom'))
        elif band in helper_func.BandTools.get_nircam_obs_band_list(target=self.target_name):
            instrument = 'nircam'
            band_wave = helper_func.BandTools.get_jwst_band_wave(band=band, unit='angstrom')
        elif band in helper_func.BandTools.get_miri_obs_band_list(target=self.target_name):
            instrument = 'miri'
            band_wave = helper_func.BandTools.get_jwst_band_wave(band=band, instrument='miri', unit='angstrom')

        elif band in helper_func.BandTools.get_astrosat_obs_band_list(target=self.target_name):
            instrument = 'astrosat'
            band_wave = helper_func.BandTools.get_astrosat_band_wave(band=band, unit='angstrom')
        else:
            raise KeyError('the band <%s> is not under the observed bands!' % band)

        # now we create a conversion factor
        # get the old unit
        old_unit = getattr(self, '%s_bands_data' % instrument)['%s_unit_img' % band]
        # get also pixel sizes
        pixel_size = getattr(self, '%s_bands_data' % instrument)['%s_pixel_area_size_sr_img' % band]
        # check if units are in the list of possible transformations
        assert old_unit in ['mJy', 'Jy', 'MJy/sr', 'erg A-1 cm-2 s-1']
        assert new_unit in ['mJy', 'Jy', 'MJy/sr', 'erg A-1 cm-2 s-1']

        conversion_factor = 1
        if old_unit != new_unit:
            # now first change the conversion factor to Jy
            if old_unit == 'mJy':
                conversion_factor *= 1e-3
            elif old_unit == 'MJy/sr':
                conversion_factor *= (1e6 * pixel_size)
            elif old_unit == 'erg A-1 cm-2 s-1':
                # The conversion from erg A-1 cm-2 s-1 is well described in
                # https://www.physicsforums.com/threads/unit-conversion-flux-densities.742561/
                # se also
                # https://www.physicsforums.com/threads/unit-conversion-of-flux-jansky-to-erg-s-cm-a-simplified-guide.927166/
                # we use fv dv = fλ dλ
                # fλ = fv dv/dλ
                # and because v = c/λ...
                # fλ = fv*c / λ^2
                # thus the conversion factor is:
                conversion_factor = 1e23 * 1e-8 * (band_wave ** 2) / (speed_of_light * 1e2)
                # the speed of light is in m/s the factor 1-e2 changes it to cm/s
                # the factor 1e8 changes Angstrom to cm (the Angstrom was in the nominator therefore it is 1/1e-8)

            # now convert to new unit
            if new_unit == 'mJy':
                conversion_factor *= 1e3
            elif new_unit == 'MJy/sr':
                conversion_factor *= 1e-6 / pixel_size
            elif new_unit == 'erg A-1 cm-2 s-1':
                conversion_factor *= 1e-23 * 1e8 * (speed_of_light * 1e2) / (band_wave ** 2)

        # change data
        getattr(self, '%s_bands_data' % instrument)['%s_data_img' % band] *= conversion_factor
        getattr(self, '%s_bands_data' % instrument)['%s_unit_img' % band] = new_unit

    def get_band_cutout_dict(self, ra_cutout, dec_cutout, cutout_size, include_err=False, band_list=None):
        """

        Parameters
        ----------
        ra_cutout : float
        dec_cutout : float
        cutout_size : float, tuple or list
            Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.
        include_err : bool
        band_list : list

        Returns
        -------
        cutout_dict : dict
        each element in dictionary is of type astropy.nddata.Cutout2D object
        """
        # geta list with all observed bands in order of wavelength
        if band_list is None:
            band_list = self.get_phangs_obs_band_list()

        if not isinstance(cutout_size, list):
            cutout_size = [cutout_size] * len(band_list)

        cutout_pos = SkyCoord(ra=ra_cutout, dec=dec_cutout, unit=(u.degree, u.degree), frame='fk5')
        cutout_dict = {'cutout_pos': cutout_pos}
        cutout_dict.update({'band_list': band_list})

        for band, band_index in zip(band_list, range(len(band_list))):
            if band in helper_func.BandTools.get_hst_obs_band_list(target=self.target_name):
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.hst_bands_data['%s_data_img' % band],
                                                   wcs=self.hst_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.get_img_cutout(img=self.hst_bands_data['%s_data_err' % band],
                                                       wcs=self.hst_bands_data['%s_wcs_err' % band],
                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})
            if band == helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name):
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.hst_ha_bands_data['%s_data_img' % band],
                                                   wcs=self.hst_ha_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.get_img_cutout(img=self.hst_ha_bands_data['%s_data_err' % band],
                                                       wcs=self.hst_ha_bands_data['%s_wcs_err' % band],
                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})
            if band == (helper_func.BandTools.get_hst_ha_band(target=self.target_ha_name) + '_cont_sub'):
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.hst_ha_bands_data['%s_data_img' % band],
                                                   wcs=self.hst_ha_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.get_img_cutout(img=self.hst_ha_bands_data['%s_data_err' % band],
                                                       wcs=self.hst_ha_bands_data['%s_wcs_err' % band],
                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})

            elif band in helper_func.BandTools.get_nircam_obs_band_list(target=self.target_name):
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.nircam_bands_data['%s_data_img' % band],
                                                   wcs=self.nircam_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.get_img_cutout(img=self.nircam_bands_data['%s_data_err' % band],
                                                       wcs=self.nircam_bands_data['%s_wcs_err' % band],
                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})

            elif band in helper_func.BandTools.get_miri_obs_band_list(target=self.target_name):
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.miri_bands_data['%s_data_img' % band],
                                                   wcs=self.miri_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.get_img_cutout(img=self.miri_bands_data['%s_data_err' % band],
                                                       wcs=self.miri_bands_data['%s_wcs_err' % band],
                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})
            elif band in helper_func.BandTools.get_astrosat_obs_band_list(target=self.target_name):
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.astrosat_bands_data['%s_data_img' % band],
                                                   wcs=self.astrosat_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
        return cutout_dict

    def get_hst_median_exp_time(self, band):
        """
        Function to calculate the median exposure time of HST observations
        Parameters
        ----------
        band : str

        Returns
        -------
        median_exp_time : float
        """
        exp_file_name = self.get_hst_img_file_name(band=band, file_type='wht')
        data, header, wcs = helper_func.FileTools.load_img(file_name=exp_file_name)
        return np.nanmedian(data[data != 0])
