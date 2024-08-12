"""
Construct a data access structure for all kind of spectroscopic data products
"""
import os.path
from pathlib import Path
import numpy as np

from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from scipy.constants import c as speed_of_light

from phangs_data_access import phangs_access_config, helper_func, phangs_info, phys_params


class SpecAccess:
    """
    Access class to organize all kind of spectroscopic data available
    """

    def __init__(self, target_name=None):
        """

        Parameters
        ----------
        target_name : str
            Default None. Target name
        """

        self.target_name = target_name

        # loaded data dictionaries
        self.muse_dap_map_data = {}
        self.muse_datacube_data = {}

        # get path to observation coverage hulls
        self.path2obs_cover_gull = (Path(__file__).parent.parent.absolute() / 'meta_data' / 'obs_coverage' /
                                    'data_output')

        super().__init__()

    def get_muse_data_file_name(self, data_prod='MUSEDAP', res='copt', ssp_model='fiducial'):
        """

        Parameters
        ----------
        data_prod : str
        res : str
        ssp_model : str
            This one is only for the copt DAPMAP data products

        Return
        ---------
        fiel_path : str
        """

        # get folder path
        file_path = (Path(phangs_access_config.phangs_config_dict['muse_data_path']) /
                     phangs_access_config.phangs_config_dict['muse_data_ver'] / res / data_prod)
        if (res == 'copt') & (data_prod == 'MUSEDAP'):
            file_path /= ssp_model

        # get file name
        if data_prod == 'MUSEDAP':
            if res == 'copt':
                file_name = '%s-%.2fasec_MAPS.fits' % (self.target_name.upper(),
                                                       phangs_info.muse_obs_res_dict[self.target_name]['copt_res'])
            elif res == 'native':
                file_name = '%s_MAPS.fits' % self.target_name.upper()
            elif res == '150pc':
                file_name = '%s-150pc_MAPS.fits' % self.target_name.upper()
            elif res == '15asec':
                file_name = '%s-15asec_MAPS.fits' % self.target_name.upper()
            else:
                raise KeyError(res, ' must be copt, native, 150pc or 15asec')
        elif data_prod == 'datacubes':
            if res == 'copt':
                file_name = '%s-%.2fasec.fits' % (self.target_name.upper(),
                                                       phangs_info.muse_obs_res_dict[self.target_name]['copt_res'])
            elif res == 'native':
                file_name = '%s.fits' % self.target_name.upper()
            elif res == '150pc':
                file_name = '%s-150pc.fits' % self.target_name.upper()
            elif res == '15asec':
                file_name = '%s-15asec.fits' % self.target_name.upper()
            else:
                raise KeyError(res, ' must be copt, native, 150pc or 15asec')
        else:
            raise KeyError(data_prod, ' must be either DAPMAP or datacubes')

        return file_path / file_name

    def load_muse_dap_map(self, res='copt', ssp_model='fiducial', map_type='HA6562_FLUX'):
        """

        Parameters
        ----------
        res : str
        ssp_model : str
            This one is only for the copt DAP results
        map_type : str

        """
        file_path = self.get_muse_data_file_name(res=res, ssp_model=ssp_model)

        # get MUSE data
        muse_hdu = fits.open(file_path)
        muse_map_data = muse_hdu[map_type].data
        muse_map_wcs = WCS(muse_hdu[map_type].header)
        muse_hdu.close()

        if res == 'copt':
            data_identifier = res + '_' + ssp_model
        else:
            data_identifier = res

        self.muse_dap_map_data.update({
            'dap_map_data_%s_%s' % (data_identifier, map_type): muse_map_data,
            'dap_map_wcs_%s_%s' % (data_identifier, map_type): muse_map_wcs
        })

    def load_muse_cube(self, res='copt'):
        """

        Parameters
        ----------
        res : str

        """
        file_path = self.get_muse_data_file_name(data_prod='datacubes', res=res)
        # get MUSE data
        muse_hdu = fits.open(file_path)
        # get header
        hdr = muse_hdu['DATA'].header
        # get wavelength
        wave_muse = hdr['CRVAL3'] + np.arange(hdr['NAXIS3']) * hdr['CD3_3']
        # get data and variance cube
        data_cube_muse = muse_hdu['DATA'].data
        var_cube_muse = muse_hdu['STAT'].data
        # get WCS
        wcs_3d_muse = WCS(hdr)
        wcs_2d_muse = wcs_3d_muse.celestial

        muse_hdu.close()

        self.muse_datacube_data.update({
            'wave_%s' % res: wave_muse,
            'data_cube_%s' % res: data_cube_muse,
            'var_cube_%s' % res: var_cube_muse,
            'hdr_%s' % res: hdr,
            'wcs_3d_%s' % res: wcs_3d_muse,
            'wcs_2d_%s' % res: wcs_2d_muse

        })

    def get_muse_obs_coverage_hull_dict(self):
        """
        Function to load the coverage dict of MUSE observations

        Returns
        -------
        coverage_mask : dict
        """
        return np.load(self.path2obs_cover_gull / ('%s_muse_obs_hull_dict.npy' % self.target_name),
                       allow_pickle=True).item()

    def check_coords_covered_by_muse(self, ra, dec, res='copt', max_dist_dist2hull_arcsec=2):
        """
        Function to check if coordinate points are inside MUSE observation

        Parameters
        ----------
        ra : float or ``np.ndarray``
        dec : float or ``np.ndarray``
        res : str
        max_dist_dist2hull_arcsec : float

        Returns
        -------
        coverage_dict : ``np.ndarray``
        """

        band_hull_dict = self.get_muse_obs_coverage_hull_dict()[res]
        coverage_mask = np.zeros(len(ra), dtype=bool)
        hull_data_ra = np.array([])
        hull_data_dec = np.array([])

        for hull_idx in band_hull_dict.keys():
            ra_hull = band_hull_dict[hull_idx]['ra']
            dec_hull = band_hull_dict[hull_idx]['dec']
            hull_data_ra = np.concatenate([hull_data_ra, ra_hull])
            hull_data_dec = np.concatenate([hull_data_dec, dec_hull])
            coverage_mask += helper_func.GeometryTools.check_points_in_polygon(x_point=ra, y_point=dec,
                                                                               x_data_hull=ra_hull,
                                                                               y_data_hull=dec_hull)

        coverage_mask *= helper_func.GeometryTools.flag_close_points2ensemble(
            x_data=ra, y_data=dec, x_data_ensemble=hull_data_ra,y_data_ensemble=hull_data_dec,
            max_dist2ensemble=max_dist_dist2hull_arcsec/3600)

        return coverage_mask

