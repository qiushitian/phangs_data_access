"""
Access structure for HST star cluster access
"""
import os.path
import numpy as np
from pathlib import Path
from astropy.table import Table, hstack

from phangs_data_access import phangs_access_config, helper_func, phangs_info
from phangs_data_access.sample_access import SampleAccess


class ClusterCatAccess:
    """
    This class is the basis to access star cluster catalogs the catalog content is described in the papers:
    Maschmann et al. 2024 and Thilker et al. 2024
    """

    def __init__(self):
        """

        """
        self.phangs_hst_cluster_cat_data_path = (
            phangs_access_config.phangs_config_dict)['phangs_hst_cluster_cat_data_path']
        self.phangs_hst_cluster_cat_release = (
            phangs_access_config.phangs_config_dict)['phangs_hst_cluster_cat_release']
        self.phangs_hst_cluster_cat_ver = (
            phangs_access_config.phangs_config_dict)['phangs_hst_cluster_cat_ver']

        self.hst_cc_data = {}

        super().__init__()

    def open_hst_cluster_cat(self, target, classify='human', cluster_class='class12', cat_type='obs'):
        """

        Function to load cluster catalogs into the constructor

        Parameters
        ----------
        target : str
        classify : str
        cluster_class : str
        cat_type : str

        Returns
        -------
        ``astropy.table.Table``
        """

        # assemble file name and path
        # get all instruments involved
        instruments = ''
        if phangs_info.phangs_hst_obs_band_dict[target]['acs_wfc1_observed_bands']:
            instruments += 'acs'
            if phangs_info.phangs_hst_obs_band_dict[target]['wfc3_uvis_observed_bands']:
                instruments += '-uvis'
        else:
            instruments += 'uvis'

        if cluster_class == 'candidates':
            file_string = Path('hlsp_phangs-cat_hst_%s_%s_multi_%s_obs-sed-candidates.fits' %
                               (instruments, target, self.phangs_hst_cluster_cat_ver))

        else:
            if classify == 'human':
                classify_str = 'human'
            elif classify == 'ml':
                classify_str = 'machine'
            else:
                raise KeyError('classify must be human or ml')

            if cluster_class == 'class12':
                cluster_str = 'cluster-class12'
            elif cluster_class == 'class3':
                cluster_str = 'compact-association-class3'
            else:
                raise KeyError('cluster_class must be class12 or class3')

            file_string = Path('hlsp_phangs-cat_hst_%s_%s_multi_v1_%s-%s-%s.fits'
                               % (instruments, target, cat_type, classify_str, cluster_str))

        # folder defined by catalog version
        folder_str = Path(self.phangs_hst_cluster_cat_release + '/catalogs')
        cluster_dict_path = Path(self.phangs_hst_cluster_cat_data_path) / folder_str

        file_path = cluster_dict_path / file_string
        if not os.path.isfile(file_path):
            print(file_path, ' not found ! ')
            raise FileNotFoundError('there is no HST cluster catalog for the target ', target,
                                    ' make sure that the file ', file_path, ' exists')
        return Table.read(file_path)

    def load_hst_cluster_cat(self, target_list=None, classify_list=None, cluster_class_list=None):
        """

        Function to load Phangs sample table into the constructor.
        Required to access global sample data

        Parameters
        ----------
        target_list : list
        classify_list : list
        cluster_class_list : list

        Returns
        -------
        None
        """
        if target_list is None:
            target_list = phangs_info.hst_cluster_catalog_target_list

        if classify_list is None:
            classify_list = ['human', 'ml']

        if cluster_class_list is None:
            cluster_class_list = ['class12', 'class3']

        for target in target_list:
            for classify in classify_list:
                for cluster_class in cluster_class_list:
                    if cluster_class in ['class12', 'class3']:
                        cluster_catalog_obs = self.open_hst_cluster_cat(target=target, classify=classify,
                                                                        cluster_class=cluster_class)
                        cluster_catalog_sed = self.open_hst_cluster_cat(target=target, classify=classify,
                                                                        cluster_class=cluster_class, cat_type='sed')

                        names_obs = list(cluster_catalog_obs.colnames)
                        names_sed = list(cluster_catalog_sed.colnames)
                        # get list of columns which are double
                        all_names_list = names_obs + names_sed
                        identifier_names = []
                        [identifier_names.append(x) for x in all_names_list if all_names_list.count(x) == 2 and
                         x not in identifier_names]

                        sed_name_list_double_id = identifier_names + names_sed
                        unique_sed_names = []
                        [unique_sed_names.append(x) for x in sed_name_list_double_id
                         if sed_name_list_double_id.count(x) == 1 and x not in unique_sed_names]

                        cluster_catalog = hstack([cluster_catalog_obs, cluster_catalog_sed[unique_sed_names]])
                    elif cluster_class == 'candidates':
                        cluster_catalog = self.open_hst_cluster_cat(target=target, classify=classify,
                                                                    cluster_class=cluster_class, cat_type='obs_sed')
                    else:
                        raise KeyError(cluster_class, ' not understood')
                    self.hst_cc_data.update({str(target) + '_' + classify + '_' + cluster_class: cluster_catalog})

    def check_load_hst_cluster_cat(self, target, classify, cluster_class):
        """
        check if catalog was loaded
        Parameters
        ----------
        target : str
        classify : str
        cluster_class : str
        """
        if not (str(target) + '_' + classify + '_' + cluster_class) in self.hst_cc_data.keys():
            self.load_hst_cluster_cat(target_list=[target], classify_list=[classify],
                                      cluster_class_list=[cluster_class])

    def get_hst_cc_phangs_candidate_id(self, target, classify='human', cluster_class='class12'):
        """
        candidate ID, can be used to connect cross identify with the initial candidate sample
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['ID_PHANGS_CANDIDATE'])

    def get_hst_cc_phangs_cluster_id(self, target, classify='human', cluster_class='class12'):
        """
        Phangs ID
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['ID_PHANGS_CLUSTER'])

    def get_hst_cc_index(self, target, classify='human', cluster_class='class12'):
        """
        running index for each individual catalog
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['INDEX'])

    def get_hst_cc_coords_pix(self, target, classify='human', cluster_class='class12'):
        """
        cluster X and Y coordinates for the PHANGS HST image products.
        These images are re-drizzled and therefore valid for all bands
        """
        x = np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_X'])
        y = np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_Y'])
        return x, y

    def get_hst_cc_coords_world(self, target, classify='human', cluster_class='class12'):
        """
        cluster coordinates RA and dec [Unit is degree]
        """
        ra = np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_RA'])
        dec = np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_DEC'])
        return ra, dec

    def get_hst_cc_class_human(self, target, classify='human', cluster_class='class12'):
        """
        Human classification
        1 [class 1] single peak, circularly symmetric, with radial profile more extended relative to point source
        2 [class 2] tar cluster – similar to Class 1, but elongated or asymmetric
        3 [class 3] compact stellar association – asymmetric, multiple peaks
        4 and above [class 4] not a star cluster or compact stellar association
        (e.g. image artifacts, background galaxies, individual stars or pairs of stars)
        For a more detailed description of other class numbers see readme of the catalog data release
        https://archive.stsci.edu/hlsps/phangs-cat/dr4/hlsp_phangs-cat_hst_multi_all_multi_v1_readme.txt
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_CLUSTER_CLASS_HUMAN'])

    def get_hst_cc_class_ml_vgg(self, target, classify='human', cluster_class='class12'):
        """
        Machine learning classification
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_CLUSTER_CLASS_ML_VGG'])

    def get_hst_cc_class_ml_vgg_qual(self, target, classify='human', cluster_class='class12'):
        """
        Estimated accuracy of machine learning classification
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_CLUSTER_CLASS_ML_VGG_QUAL'])

    def get_hst_ccd_class(self, target, classify='human', cluster_class='class12'):
        """
        Classification based in U-B vs. V-I color-color diagram (See Maschmann et al. 2024) Section 4.4
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_COLORCOLOR_CLASS_UBVI'])

    def get_hst_cc_age(self, target, classify='human', cluster_class='class12'):
        """
        Age see Thilker et al. 2024 [unit is Myr]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_age'])

    def get_hst_cc_age_err(self, target, classify='human', cluster_class='class12'):
        """
        Upper and lower uncertainty of Ages [unit is Myr]
        """
        return (np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_age_limlo']),
                np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_age_limhi']))

    def get_hst_cc_mstar(self, target, classify='human', cluster_class='class12'):
        """
        stellar mass [unit M_sun]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_mass'])

    def get_hst_cc_mstar_err(self, target, classify='human', cluster_class='class12'):
        """
        Upper and lower uncertainty of stellar mass [unit M_sun]

        """
        return (np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_mass_limlo']),
                np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_mass_limhi']))

    def get_hst_cc_ebv(self, target, classify='human', cluster_class='class12'):
        """
        Dust attenuation measured in E(B-V) [unit mag]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_ebv'])

    def get_hst_cc_ebv_err(self, target, classify='human', cluster_class='class12'):
        """
        Upper and lower uncertainty of dust attenuation measured in E(B-V) [unit mag]
        """
        return (np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_ebv_limlo']),
                np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['SEDfix_ebv_limhi']))

    def get_hst_cc_ir4_age(self, target, classify='human', cluster_class='class12'):
        """
        Old age estimation without decision tree [unit Age]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_AGE_MINCHISQ'])

    def get_hst_cc_ir4_age_err(self, target, classify='human', cluster_class='class12'):
        """
        Uncertainties of old age estimation without decision tree [unit Age]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_AGE_MINCHISQ_ERR'])

    def get_hst_cc_ir4_mstar(self, target, classify='human', cluster_class='class12'):
        """
        Old stellar mass estimation without decision tree [unit M_sun]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_MASS_MINCHISQ'])

    def get_hst_cc_ir4_mstar_err(self, target, classify='human', cluster_class='class12'):
        """
        Uncertainties of old stellar mass estimation without decision tree [unit M_sun]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_MASS_MINCHISQ_ERR'])

    def get_hst_cc_ir4_ebv(self, target, classify='human', cluster_class='class12'):
        """
        Old dust attenuation E(B-V) estimation without decision tree [unit mag]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_EBV_MINCHISQ'])

    def get_hst_cc_ir4_ebv_err(self, target, classify='human', cluster_class='class12'):
        """
        Uncertainties of old dust attenuation E(B-V) estimation without decision tree [unit mag]
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_EBV_MINCHISQ_ERR'])

    def get_hst_cc_ci(self, target, classify='human', cluster_class='class12'):
        """
        V-band concentration index, difference in magnitudes measured in 1 pix and 3 pix radii apertures.
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_CI'])

    def get_hst_cc_ir4_min_chi2(self, target, classify='human', cluster_class='class12'):
        """
        Old minimal reduced chi-square
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_REDUCED_MINCHISQ'])

    def get_hst_cc_cov_flag(self, target, classify='human', cluster_class='class12'):
        """
        Integer denoting the number of bands with no coverage for object
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]['PHANGS_NO_COVERAGE_FLAG'])

    def get_hst_cc_non_det_flag(self, target, classify='human', cluster_class='class12'):
        """
        Integer denoting the number of bands in which the photometry for the object was below the requested
        signal-to-noise ratio (S/N=1). 0 indicates all five bands had detections. A value of 1 and 2 means the object
         was detected in four and three bands, respectively. By design, this flag cannot be higher than 2.
        """
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_NON_DETECTION_FLAG'])

    @staticmethod
    def filter_name2hst_band(target, filter_name):
        """
        Method to get from band-pass filter names to the HST filter names used for this observation.
        """
        if filter_name == 'NUV':
            return 'F275W'
        elif filter_name == 'U':
            return 'F336W'
        elif filter_name == 'B':
            if 'F438W' in phangs_info.phangs_hst_obs_band_dict[target]['wfc3_uvis_observed_bands']:
                return 'F438W'
            else:
                return 'F435W'
        elif filter_name == 'V':
            return 'F555W'
        elif filter_name == 'I':
            return 'F814W'
        else:
            raise KeyError(filter_name, ' is not available ')

    def get_hst_cc_band_flux(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        Flux in a specific band [unit is mJy]
        """
        band = self.filter_name2hst_band(target=target, filter_name=filter_name)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_%s_mJy' % band.upper()])

    def get_hst_cc_band_flux_err(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        Uncertainty of flux in a specific band [unit is mJy]
        """
        band = self.filter_name2hst_band(target=target, filter_name=filter_name)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_%s_mJy_ERR' % band.upper()])

    def get_hst_cc_band_sn(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        Signa-to-noise in a specific band
        """
        return (self.get_hst_cc_band_flux(target=target, filter_name=filter_name, classify=classify, cluster_class=cluster_class) /
                self.get_hst_cc_band_flux_err(target=target, filter_name=filter_name, classify=classify, cluster_class=cluster_class))

    def get_hst_cc_band_detect_mask(self, target, filter_name, sn=3, classify='human', cluster_class='class12'):
        """
        get boolean mask for objects detected at a signal-to-noise ratio (sn)
        """
        return np.array(self.get_hst_cc_band_sn(target=target, filter_name=filter_name, classify=classify,
                                                 cluster_class=cluster_class) > sn)

    def get_hst_cc_band_vega_mag(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        magnitude [unit is Vega mag]
        """
        band = self.filter_name2hst_band(target=target, filter_name=filter_name)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_%s_VEGA' % band.upper()])

    def get_hst_cc_band_vega_mag_err(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        Uncertainty of magnitude. Since there is only a specific offset between AB and Vega magnitude systems,
        this is also valid for AB magnitudes [unit is mag]
        """
        band = self.filter_name2hst_band(target=target, filter_name=filter_name)
        return np.array(self.hst_cc_data[str(target) + '_' + classify + '_' + cluster_class]
                        ['PHANGS_%s_VEGA_ERR' % band.upper()])

    def get_hst_cc_band_ab_mag(self, target, filter_name, classify='human', cluster_class='class12'):
        """
        magnitude [unit is AB mag]
        """
        flux = self.get_hst_cc_band_flux(target=target, filter_name=filter_name, classify=classify,
                                         cluster_class=cluster_class)
        return helper_func.conv_mjy2ab_mag(flux=flux)

    def get_hst_cc_color(self, target, filter_name_1, filter_name_2, mag_sys='vega', classify='human',
                         cluster_class='class12'):
        """
        get magnitude difference between two bands also called color.

        Parameters
        ----------
        target : str
        filter_name_1: str
        filter_name_2: str
        mag_sys: str
        classify: str
        cluster_class: str
        """
        assert (filter_name_1 in ['NUV', 'U', 'B', 'V', 'I']) & (filter_name_2 in ['NUV', 'U', 'B', 'V', 'I'])
        assert mag_sys in ['vega', 'ab']
        band_mag_1 = (getattr(self, 'get_hst_cc_band_%s_mag' % mag_sys)
                      (target=target, filter_name=filter_name_1, classify=classify, cluster_class=cluster_class))
        band_mag_2 = (getattr(self, 'get_hst_cc_band_%s_mag' % mag_sys)
                      (target=target, filter_name=filter_name_2, classify=classify, cluster_class=cluster_class))
        return band_mag_1 - band_mag_2

    def get_hst_cc_color_err(self, target, filter_name_1, filter_name_2, classify='human', cluster_class='class12'):
        """
        Uncertainty of color.

        Parameters
        ----------
        target : str
        filter_name_1: str
        filter_name_2: str
        classify: str
        cluster_class: str
        """
        assert (filter_name_1 in ['NUV', 'U', 'B', 'V', 'I']) & (filter_name_2 in ['NUV', 'U', 'B', 'V', 'I'])
        band_mag_err_1 = (self.get_hst_cc_band_vega_mag_err
                          (target=target, filter_name=filter_name_1, classify=classify, cluster_class=cluster_class))
        band_mag_err_2 = (self.get_hst_cc_band_vega_mag_err
                          (target=target, filter_name=filter_name_1, classify=classify, cluster_class=cluster_class))

        return np.sqrt(band_mag_err_1 ** 2 + band_mag_err_2 ** 2)

    def get_quick_access(self, target='all', classify='human', cluster_class='class123',
                         save_quick_access=True, reload=False):
        """
        Function to quickly access catalog data. The loaded data are stored locally in a dictionary and one can
        directly access them via key-words

        Parameters
        ----------
        target : str
        classify: str
        cluster_class: str
        save_quick_access: bool
            if you want to save the dictionary for easier access next time you are loading this.
            However, for this the keyword phangs_hst_cluster_cat_quick_access_path must be specified in the
            phangs_config_dict
        reload: bool
            In case you want to reload this dictionary for example due to an update
        """
        quick_access_dict_path = \
            (Path(phangs_access_config.phangs_config_dict['phangs_hst_cluster_cat_quick_access_path']) /
             ('quick_access_dict_%s_%s_%s.npy' % (target, classify, cluster_class)))

        if os.path.isfile(quick_access_dict_path) and not reload:
            return np.load(quick_access_dict_path, allow_pickle=True).item()
        else:
            phangs_cluster_id = np.array([])
            phangs_candidate_id = np.array([])
            index = np.array([])
            target_name = np.array([], dtype=str)
            ra = np.array([])
            dec = np.array([])
            x = np.array([])
            y = np.array([])
            cluster_class_hum = np.array([])
            cluster_class_ml = np.array([])
            cluster_class_ml_qual = np.array([])

            color_vi_vega = np.array([])
            color_nuvu_vega = np.array([])
            color_ub_vega = np.array([])
            color_bv_vega = np.array([])

            color_vi_ab = np.array([])
            color_nuvu_ab = np.array([])
            color_ub_ab = np.array([])
            color_bv_ab = np.array([])

            color_vi_err = np.array([])
            color_nuvu_err = np.array([])
            color_ub_err = np.array([])
            color_bv_err = np.array([])

            detect_nuv = np.array([], dtype=bool)
            detect_u = np.array([], dtype=bool)
            detect_b = np.array([], dtype=bool)
            detect_v = np.array([], dtype=bool)
            detect_i = np.array([], dtype=bool)

            v_mag_vega = np.array([])
            abs_v_mag_vega = np.array([])
            v_mag_ab = np.array([])
            abs_v_mag_ab = np.array([])

            ccd_class = np.array([])
            age = np.array([])
            mstar = np.array([])
            ebv = np.array([])

            if target == 'all':
                target_list = phangs_info.hst_cluster_catalog_target_list
            else:
                target_list = [target]

            # add which cluster classes need to be added
            cluster_class_list = []
            if '12' in cluster_class:
                cluster_class_list.append('class12')
            if '3' in cluster_class:
                cluster_class_list.append('class3')

            for target in target_list:

                for cluster_class in cluster_class_list:

                    phangs_cluster_id = np.concatenate([phangs_cluster_id,
                                                        self.get_hst_cc_phangs_cluster_id(target=target,
                                                                                          cluster_class=cluster_class,
                                                                                          classify=classify)])
                    phangs_candidate_id = np.concatenate([phangs_candidate_id,
                                                          self.get_hst_cc_phangs_candidate_id(target=target,
                                                                                              cluster_class=cluster_class,
                                                                                              classify=classify)])
                    index = np.concatenate([index, self.get_hst_cc_index(target=target, cluster_class=cluster_class,
                                                                         classify=classify)])
                    ra_, dec_ = self.get_hst_cc_coords_world(target=target, cluster_class=cluster_class,
                                                             classify=classify)
                    ra = np.concatenate([ra, ra_])
                    dec = np.concatenate([dec, dec_])
                    x_, y_ = self.get_hst_cc_coords_pix(target=target, cluster_class=cluster_class, classify=classify)
                    x = np.concatenate([x, x_])
                    y = np.concatenate([y, y_])
                    target_name = np.concatenate([target_name, [target]*len(ra_)])
                    cluster_class_hum = np.concatenate([cluster_class_hum,
                                                        self.get_hst_cc_class_human(target=target,
                                                                                    cluster_class=cluster_class,
                                                                                    classify=classify)])
                    cluster_class_ml = np.concatenate([cluster_class_ml,
                                                       self.get_hst_cc_class_ml_vgg(target=target,
                                                                                    cluster_class=cluster_class,
                                                                                    classify=classify)])
                    cluster_class_ml_qual =\
                        np.concatenate([cluster_class_ml_qual,
                                        self.get_hst_cc_class_ml_vgg_qual(target=target, cluster_class=cluster_class,
                                                                          classify=classify)])

                    color_vi_vega = np.concatenate([color_vi_vega,
                                                    self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                          filter_name_1='V', filter_name_2='I',
                                                                          classify=classify)])
                    color_nuvu_vega = np.concatenate([color_nuvu_vega,
                                                      self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                            filter_name_1='NUV', filter_name_2='U',
                                                                            classify=classify)])
                    color_ub_vega = np.concatenate([color_ub_vega,
                                                    self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                          filter_name_1='U', filter_name_2='B',
                                                                          classify=classify)])
                    color_bv_vega = np.concatenate([color_bv_vega,
                                                    self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                          filter_name_1='B', filter_name_2='V',
                                                                          classify=classify)])

                    color_vi_ab = np.concatenate([color_vi_ab,
                                                  self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                        filter_name_1='V', filter_name_2='I',
                                                                        mag_sys='ab', classify=classify)])
                    color_nuvu_ab = np.concatenate([color_nuvu_ab,
                                                    self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                          filter_name_1='NUV', filter_name_2='U',
                                                                          mag_sys='ab', classify=classify)])
                    color_ub_ab = np.concatenate([color_ub_ab,
                                                  self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                        filter_name_1='U', filter_name_2='B',
                                                                        mag_sys='ab', classify=classify)])
                    color_bv_ab = np.concatenate([color_bv_ab,
                                                  self.get_hst_cc_color(target=target, cluster_class=cluster_class,
                                                                        filter_name_1='B', filter_name_2='V',
                                                                        mag_sys='ab', classify=classify)])

                    color_vi_err = np.concatenate([color_vi_err,
                                                   self.get_hst_cc_color_err(target=target, cluster_class=cluster_class,
                                                                             filter_name_1='V', filter_name_2='I',
                                                                             classify=classify)])
                    color_nuvu_err = np.concatenate([color_nuvu_err,
                                                     self.get_hst_cc_color_err(target=target,
                                                                               cluster_class=cluster_class,
                                                                               filter_name_1='NUV', filter_name_2='U',
                                                                               classify=classify)])
                    color_ub_err = np.concatenate([color_ub_err,
                                                   self.get_hst_cc_color_err(target=target, cluster_class=cluster_class,
                                                                             filter_name_1='U', filter_name_2='B',
                                                                             classify=classify)])
                    color_bv_err = np.concatenate([color_bv_err,
                                                   self.get_hst_cc_color_err(target=target, cluster_class=cluster_class,
                                                                             filter_name_1='B', filter_name_2='V',
                                                                             classify=classify)])

                    detect_nuv = np.concatenate([detect_nuv,
                                                 self.get_hst_cc_band_detect_mask(target=target, filter_name='NUV',
                                                                                  cluster_class=cluster_class,
                                                                                  classify=classify)])
                    detect_u = np.concatenate([detect_u,
                                               self.get_hst_cc_band_detect_mask(target=target, filter_name='U',
                                                                                cluster_class=cluster_class,
                                                                                classify=classify)])
                    detect_b = np.concatenate([detect_b,
                                               self.get_hst_cc_band_detect_mask(target=target, filter_name='B',
                                                                                cluster_class=cluster_class,
                                                                                classify=classify)])
                    detect_v = np.concatenate([detect_v,
                                               self.get_hst_cc_band_detect_mask(target=target, filter_name='V',
                                                                                cluster_class=cluster_class,
                                                                                classify=classify)])
                    detect_i = np.concatenate([detect_i,
                                               self.get_hst_cc_band_detect_mask(target=target, filter_name='I',
                                                                                cluster_class=cluster_class,
                                                                                classify=classify)])

                    # get V-band magnitude and absolute magnitude
                    v_mag_vega_ = self.get_hst_cc_band_vega_mag(target=target, filter_name='V',
                                                                cluster_class=cluster_class, classify=classify)
                    v_mag_ab_ = self.get_hst_cc_band_ab_mag(target=target, filter_name='V', cluster_class=cluster_class,
                                                            classify=classify)

                    v_mag_vega = np.concatenate([v_mag_vega, v_mag_vega_])
                    v_mag_ab = np.concatenate([v_mag_ab, v_mag_ab_])
                    # get distance
                    sample_access = SampleAccess()
                    print(target)
                    target_dist = sample_access.get_target_dist(
                        target=helper_func.FileTools.get_sample_table_target_name(target=target))
                    abs_v_mag_vega = np.concatenate([abs_v_mag_vega,
                                                     helper_func.UnitTools.conv_mag2abs_mag(mag=v_mag_vega_,
                                                                                            dist=target_dist)])
                    abs_v_mag_ab = np.concatenate([abs_v_mag_ab,
                                                   helper_func.UnitTools.conv_mag2abs_mag(mag=v_mag_ab_,
                                                                                          dist=target_dist)])

                    ccd_class = np.concatenate([ccd_class, self.get_hst_ccd_class(target=target,
                                                                                  cluster_class=cluster_class,
                                                                                  classify=classify)])

                    age = np.concatenate([age, self.get_hst_cc_age(target=target, cluster_class=cluster_class,
                                                                   classify=classify)])
                    mstar = np.concatenate([mstar, self.get_hst_cc_mstar(target=target, cluster_class=cluster_class,
                                                                         classify=classify)])
                    ebv = np.concatenate([ebv, self.get_hst_cc_ebv(target=target, cluster_class=cluster_class,
                                                                   classify=classify)])

            quick_access_dict = {
                'phangs_cluster_id': phangs_cluster_id,
                'phangs_candidate_id': phangs_candidate_id,
                'index': index,
                'target_name': target_name,
                'ra': ra,
                'dec': dec,
                'x': x,
                'y': y,
                'cluster_class_hum': cluster_class_hum,
                'cluster_class_ml': cluster_class_ml,
                'cluster_class_ml_qual': cluster_class_ml_qual,
                'color_vi_vega': color_vi_vega,
                'color_nuvu_vega': color_nuvu_vega,
                'color_ub_vega': color_ub_vega,
                'color_bv_vega': color_bv_vega,
                'color_vi_ab': color_vi_ab,
                'color_nuvu_ab': color_nuvu_ab,
                'color_ub_ab': color_ub_ab,
                'color_bv_ab': color_bv_ab,
                'color_vi_err': color_vi_err,
                'color_nuvu_err': color_nuvu_err,
                'color_ub_err': color_ub_err,
                'color_bv_err': color_bv_err,
                'detect_nuv': detect_nuv,
                'detect_u': detect_u,
                'detect_b': detect_b,
                'detect_v': detect_v,
                'detect_i': detect_i,
                'v_mag_vega': v_mag_vega,
                'abs_v_mag_vega': abs_v_mag_vega,
                'v_mag_ab': v_mag_ab,
                'abs_v_mag_ab': abs_v_mag_ab,
                'ccd_class': ccd_class,
                'age': age,
                'mstar': mstar,
                'ebv': ebv
            }

            if save_quick_access:
                np.save(quick_access_dict_path, quick_access_dict)
            return quick_access_dict
