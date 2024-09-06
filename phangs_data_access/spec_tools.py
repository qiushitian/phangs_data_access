"""
Tools for spectrsocopic analysis
"""
import numpy as np
import ppxf.ppxf_util as util

from astropy import constants as const
speed_of_light_kmps = const.c.to('km/s').value
from os import path

from ppxf.ppxf import ppxf
import ppxf.sps_util as lib
from urllib import request
from TardisPipeline.readData.MUSE_WFM import get_MUSE_polyFWHM


class SpecTools:
    def __init__(self):
        pass

    @staticmethod
    def get_target_ned_redshift(target):
        """
        Function to get redshift from NED with astroquery
        Parameters
        ----------

        Returns
        -------
        redshift : float
        """

        from astroquery.ipac.ned import Ned
        # get the center of the target
        ned_table = Ned.query_object(target)

        return ned_table['Redshift'][0]

    @staticmethod
    def get_target_sys_vel(target):
        """
        Function to get target systemic velocity based on NED redshift
        Parameters
        ----------

        Returns
        -------
        sys_vel : float
        """
        redshift = SpecTools.get_target_ned_redshift(target=target)
        return np.log(redshift + 1) * speed_of_light_kmps

    @staticmethod
    def fit_ppxf2spec(spec_dict, redshift, sps_name='fsps', age_range=None, metal_range=None):
        """

        Parameters
        ----------
        spec_dict : dict
        sps_name : str
            can be fsps, galaxev or emiles



        Returns
        -------
        dict
        """

        import matplotlib.pyplot as plt
        spec_dict['spec_flux'] *= 1e-20
        spec_dict['spec_flux_err'] *= 1e-20

        velscale = speed_of_light_kmps * np.diff(np.log(spec_dict['lam'][-2:]))[0]  # Smallest velocity step
        # print('velscale ', velscale)
        # velscale = speed_of_light_kmps*np.log(spec_dict['lam'][1]/spec_dict['lam'][0])
        # print('velscale ', velscale)
        # velscale = 10
        spectra_muse, ln_lam_gal, velscale = util.log_rebin(lam=spec_dict['lam'], spec=spec_dict['spec_flux'],
                                                            velscale=velscale)
        spectra_muse_err, ln_lam_gal, velscale = util.log_rebin(lam=spec_dict['lam'],
                                                                spec=spec_dict['spec_flux_err'], velscale=velscale)

        lsf_dict = {"lam": spec_dict['lam'], "fwhm": spec_dict['lsf']}
        # get new wavelength array
        lam_gal = np.exp(ln_lam_gal)

        # get stellar library
        ppxf_dir = path.dirname(path.realpath(lib.__file__))
        basename = f"spectra_{sps_name}_9.0.npz"
        filename = path.join(ppxf_dir, 'sps_models', basename)
        if not path.isfile(filename):
            url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
            request.urlretrieve(url, filename)

        sps = lib.sps_lib(filename=filename, velscale=velscale, fwhm_gal=lsf_dict, norm_range=[5070, 5950],
                          #wave_range=None,
                          age_range=age_range, metal_range=metal_range)
        reg_dim = sps.templates.shape[1:]  # shape of (n_ages, n_metal)
        stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)

        gas_templates, gas_names, line_wave = util.emission_lines(ln_lam_temp=sps.ln_lam_temp,
                                                                  lam_range_gal=spec_dict['lam_range'],
                                                                  FWHM_gal=get_MUSE_polyFWHM,
                                                                  limit_doublets=False)

        templates = np.column_stack([stars_templates, gas_templates])

        n_star_temps = stars_templates.shape[1]
        component = [0] * n_star_temps
        for line_name in gas_names:
            if '[' in line_name:
                component += [2]
            else:
                component += [1]

        gas_component = np.array(component) > 0  # gas_component=True for gas templates
        moments = [4, 4, 4]
        vel = speed_of_light_kmps * np.log(1 + redshift)  # eq.(8) of Cappellari (2017)
        start_gas = [vel, 150., 0, 0]  # starting guess
        start_star = [vel, 150., 0, 0]
        start = [start_star, start_gas, start_gas]

        # mask bad values
        mask = np.invert(np.isnan(spectra_muse_err))
        spectra_muse_err[np.isnan(spectra_muse_err)] = np.nanmean(spectra_muse_err)
        spectra_muse[np.isnan(spectra_muse)] = 0

        pp = ppxf(templates=templates, galaxy=spectra_muse, noise=spectra_muse_err, velscale=velscale, start=start,
                  moments=moments, degree=-1, mdegree=4, lam=lam_gal, lam_temp=sps.lam_temp,
                  reg_dim=reg_dim, component=component, gas_component=gas_component,
                  reddening=2.5, gas_reddening=0.0, gas_names=gas_names, mask=mask)

        light_weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
        light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
        light_weights /= light_weights.sum()  # Normalize to light fractions

        ages, met = sps.mean_age_metal(light_weights)
        mass2light = sps.mass_to_light(light_weights, redshift=redshift)

        wavelength = pp.lam
        total_flux = pp.galaxy
        total_flux_err = pp.noise

        best_fit = pp.bestfit
        gas_best_fit = pp.gas_bestfit
        continuum_best_fit = best_fit - gas_best_fit

        # get velocity of balmer component
        sol_kin_comp = pp.sol[0]
        balmer_kin_comp = pp.sol[1]
        forbidden_kin_comp = pp.sol[2]

        h_beta_rest_air = 4861.333
        h_alpha_rest_air = 6562.819

        balmer_redshift = np.exp(balmer_kin_comp[0] / speed_of_light_kmps) - 1

        observed_h_beta = h_beta_rest_air * (1 + balmer_redshift)
        observed_h_alpha = h_alpha_rest_air * (1 + balmer_redshift)

        observed_sigma_h_alpha = (balmer_kin_comp[1] / speed_of_light_kmps) * h_alpha_rest_air
        observed_sigma_h_alpha = np.sqrt(observed_sigma_h_alpha ** 2 + get_MUSE_polyFWHM(observed_h_alpha))
        observed_sigma_h_beta = (balmer_kin_comp[1] / speed_of_light_kmps) * h_beta_rest_air
        observed_sigma_h_beta = np.sqrt(observed_sigma_h_beta ** 2 + get_MUSE_polyFWHM(observed_h_beta))

        mask_ha = (wavelength > (observed_h_alpha - 3 * observed_sigma_h_alpha)) & (
                wavelength < (observed_h_alpha + 3 * observed_sigma_h_alpha))
        mask_hb = (wavelength > (observed_h_beta - 3 * observed_sigma_h_beta)) & (
                wavelength < (observed_h_beta + 3 * observed_sigma_h_beta))

        ha_line_comp = (total_flux - continuum_best_fit)[mask_ha]
        ha_cont_comp = continuum_best_fit[mask_ha]
        ha_wave_comp = wavelength[mask_ha]
        delta_lambda_ha = np.mean((ha_wave_comp[1:] - ha_wave_comp[:-1]) / 2)
        ha_ew = np.sum(((ha_cont_comp - ha_line_comp) / ha_cont_comp) * delta_lambda_ha)

        hb_line_comp = (total_flux - continuum_best_fit)[mask_hb]
        hb_cont_comp = continuum_best_fit[mask_hb]
        hb_wave_comp = wavelength[mask_hb]
        delta_lambda_hb = np.mean((hb_wave_comp[1:] - hb_wave_comp[:-1]) / 2)
        hb_ew = np.sum(((hb_cont_comp - hb_line_comp) / hb_cont_comp) * delta_lambda_hb)

        # gas_phase_metallicity
        flux_ha = pp.gas_flux[pp.gas_names == 'Halpha']
        flux_hb = pp.gas_flux[pp.gas_names == 'Hbeta']
        flux_nii = pp.gas_flux[pp.gas_names == '[OIII]5007_d']
        flux_oiii = pp.gas_flux[pp.gas_names == '[NII]6583_d']

        o3n2 = np.log10((flux_oiii / flux_hb) / (flux_nii / flux_ha))
        gas_phase_met = 8.73 - 0.32 * o3n2[0]
        # plt.plot(hb_wave_comp, hb_line_comp)
        # plt.plot(hb_wave_comp, hb_cont_comp)
        # plt.show()
        # exit()
        #
        # # exit()
        # plt.errorbar(wavelength, total_flux, yerr=total_flux_err)
        # plt.plot(wavelength, continuum_best_fit)
        # plt.scatter(wavelength[left_idx_ha[0][0]], continuum_best_fit[left_idx_ha[0][0]])
        # plt.scatter(wavelength[right_idx_ha[0][0]], continuum_best_fit[right_idx_ha[0][0]])
        # plt.plot(wavelength, continuum_best_fit + gas_best_fit)
        # plt.plot(wavelength, gas_best_fit)
        # plt.plot([observed_nii_1, observed_nii_1], [np.min(total_flux), np.max(total_flux)])
        # plt.plot([observed_h_alpha, observed_h_alpha], [np.min(total_flux), np.max(total_flux)])
        # plt.plot([observed_nii_2, observed_nii_2], [np.min(total_flux), np.max(total_flux)])
        # plt.show()
        #

        return {
            'wavelength': wavelength, 'total_flux': total_flux, 'total_flux_err': total_flux_err,
            'best_fit': best_fit, 'gas_best_fit': gas_best_fit, 'continuum_best_fit': continuum_best_fit,
            'ages': ages, 'met': met, 'mass2light': mass2light,
            'star_red': pp.dust[0]['sol'][0], 'gas_red': pp.dust[1]['sol'][0],
            'sol_kin_comp': sol_kin_comp, 'balmer_kin_comp': balmer_kin_comp, 'forbidden_kin_comp': forbidden_kin_comp,
            'ha_ew': ha_ew, 'hb_ew': hb_ew, 'gas_phase_met': gas_phase_met
        }

        #
        #
        #
        # plt.figure(figsize=(17, 6))
        # plt.subplot(111)
        # pp.plot()
        # plt.show()
        #

        #
        # exit()



#####################
##### Code dump #####
#####################
# started to develop an alternative fit for the Tardis pipeline
#
# def fit_tardis2spec(spec_dict, velocity, hdr, sps_name='fsps', age_range=None, metal_range=None, name='explore1'):
#     """
#
#     Parameters
#     ----------
#     spec_dict : dict
#     sps_name : str
#         can be fsps, galaxev or emiles
#
#
#
#     Returns
#     -------
#     dict
#     """
#     from os import path
#     # import ppxf.sps_util as lib
#     # from urllib import request
#     # from ppxf.ppxf import ppxf
#
#     import matplotlib.pyplot as plt
#
#     from TardisPipeline.utilities import util_ppxf, util_ppxf_stellarpops, util_sfh_quantities, util_ppxf_emlines
#     import TardisPipeline as tardis_module
#     codedir = os.path.dirname(os.path.realpath(tardis_module.__file__))
#
#     import ppxf.ppxf_util as util
#     from astropy.io import fits, ascii
#     from astropy import constants as const
#     from astropy.table import Table
#     import extinction
#
#     # tardis_path = '/home/egorov/Soft/ifu-pipeline/TardisPipeline/' # change to directory where you have installed DAP
#     ncpu = 20  # how many cpu would you like to use? (20-30 is fine for our server, but use no more than 8 for laptop)
#     # print(codedir+'/Templates/spectralTemplates/eMILES-noyoung/')
#     # exit()
#     configs = {  #'SSP_LIB': os.path.join(codedir, 'Templates/spectralTemplates/eMILES-noyoung/'),
#         #'SSP_LIB_SFH': os.path.join(codedir, 'Templates/spectralTemplates/eMILES-noyoung/'),
#         'SSP_LIB': codedir + '/Templates/spectralTemplates/CB07_chabrier-young-selection-MetalPoorRemoved/',
#         # stellar library to use
#         'SSP_LIB_SFH': codedir + '/Templates/spectralTemplates/CB07_chabrier-young-selection-MetalPoorRemoved/',
#         # stellar library to use
#         # 'SSP_LIB': codedir+'/Templates/spectralTemplates/eMILES-noyoung/',  # stellar library to use
#         'NORM_TEMP': 'LIGHT', 'REDSHIFT': velocity, 'MOM': 4, 'MC_PPXF': 0, 'PARALLEL': 1,
#         'ADEG': 12,
#         'ADEG_SFH': 12,
#         'MDEG': 0,
#         'MDEG_SFH': 0,
#         'MDEG_EMS': 24,
#         'NCPU': ncpu,
#         'ROOTNAME': name,
#         'SPECTRUM_SIZE': abs(hdr['CD1_1']) * 3600.,  # spaxel size in arcsec
#         # 'EMI_FILE': os.path.join(codedir, '/Templates/configurationTemplates/emission_lines.setup'),
#         'MC_PPXF_SFH': 10,
#         'EMI_FILE': codedir + '/Templates/configurationTemplates/emission_lines.setup',  # set of emission lines to fit
#         'SKY_LINES_RANGES': codedir + '/Templates/configurationTemplates/sky_lines_ranges.setup',
#         'OUTDIR': 'data_output/',
#         'MASK_WIDTH': 150,
#         'GAS_MOMENTS': 4}
#
#     velscale = speed_of_light_kmps * np.diff(np.log(spec_dict['lam'][-2:]))[0]  # Smallest velocity step
#     log_spec, logLam, velscale = util.log_rebin(lam=spec_dict['lam_range'], spec=spec_dict['spec_flux'],
#                                                 velscale=velscale)
#     c1 = fits.Column(name='LOGLAM', array=logLam, format='D')
#     c2 = fits.Column(name='LOGSPEC', array=log_spec, format='D')
#     t = fits.BinTableHDU.from_columns([c1, c2])
#     t.writeto('{}{}-ppxf_obsspec.fits'.format(configs['OUTDIR'], name), overwrite=True)
#     log_err, _, _ = util.log_rebin(spec_dict['lam_range'], spec_dict['spec_flux_err'], velscale=velscale)
#     ww = ~np.isfinite(log_spec) | ~np.isfinite(log_err) | (log_err <= 0)
#     log_err[ww] = 9999
#     log_spec[ww] = 0.
#     # # the DAP fitting routines expect log_spec and log_err to be 2D arrays containing N spectra,
#     # # here we add a dummy dimension since we are fitting only one spectrum
#     # # to fit more than one spectrum at the same time these lines can be easily adapted
#     log_err = np.expand_dims(log_err, axis=1)
#     log_spec = np.expand_dims(log_spec, axis=1)
#
#     # define the LSF of the MUSE data
#     LSF = get_MUSE_polyFWHM(np.exp(logLam), version="udf10")
#
#     # define the velocity scale in kms
#     velscale = (logLam[1] - logLam[0]) * speed_of_light_kmps
#
#     # this is the stellar kinematics ppxf wrapper function
#     ppxf_result = util_ppxf.runModule_PPXF(configs=configs,  #tasks='',
#                                            logLam=logLam,
#                                            log_spec=log_spec, log_error=log_err,
#                                            LSF=LSF)  #, velscale=velscale)
#     util_ppxf_emlines.runModule_PPXF_emlines(configs=configs,  #tasks='',
#                                              logLam=logLam,
#                                              log_spec=log_spec, log_error=log_err,
#                                              LSF=LSF, ppxf_results=ppxf_result)
#
#     # exit()
#     util_ppxf_stellarpops.runModule_PPXF_stellarpops(configs, logLam, log_spec, log_err, LSF, np.arange(1), ppxf_result)
#     masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err = util_sfh_quantities.compute_sfh_relevant_quantities(
#         configs)
#     print(masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err)
#
#     # read the output file which contains the best-fit from the emission lines fitting stage
#     ppxf_bestfit_gas = configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf-bestfit-emlines.fits'
#     hdu3 = fits.open(ppxf_bestfit_gas)
#     bestfit_gas = hdu3['FIT'].data["BESTFIT"][0]
#     mask = (hdu3['FIT'].data['BESTFIT'][0] == 0)
#     gas_templ = hdu3['FIT'].data["GAS_BESTFIT"][0]
#
#     ppxf_bestfit = configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf-bestfit.fits'
#     hdu_best_fit = fits.open(ppxf_bestfit)
#     cont_fit = hdu_best_fit['FIT'].data["BESTFIT"][0]
#
#     # # reddening = ppxf_sfh_data['REDDENING']
#     # hdu_best_fit_sfh = fits.open('data_output/explore1_ppxf-bestfit.fits')
#     # print(hdu_best_fit_sfh.info())
#     # print(hdu_best_fit_sfh[1].data.names)
#     #
#     # print(hdu_best_fit_sfh['FIT'].data['BESTFIT'])
#     # print(hdu_best_fit_sfh['FIT'].data['BESTFIT'].shape)
#     # print(logLam.shape)
#     # print(spec_dict['lam'].shape)
#     # # exit()
#     # # hdu_best_fit = fits.open('data_output/explore1_templates_SFH_info.fits')
#     # # print(hdu_best_fit.info())
#     # # print(hdu_best_fit[1].data.names)
#     # # print(hdu_best_fit[1].data['Age'])
#
#     plt.plot(spec_dict['lam'], spec_dict['spec_flux'])
#     plt.plot(np.exp(logLam), cont_fit)
#     plt.plot(np.exp(logLam), gas_templ)
#     plt.plot(np.exp(logLam), cont_fit + gas_templ)
#     plt.show()
#
#     exit()
#     # this the ppxf wrapper function to simulataneously fit the continuum plus emission lines
#     # util_ppxf_emlines.runModule_PPXF_emlines(configs,# '',
#     #                                          logLam, log_spec,
#     #                                          log_err, LSF, #velscale,
#     #                                          np.arange(1), ppxf_result)
#     util_ppxf_emlines.runModule_PPXF_emlines(configs=configs,  #tasks='',
#                                              logLam=logLam,
#                                              log_spec=log_spec, log_error=log_err,
#                                              LSF=LSF, ppxf_results=ppxf_result)
#
#     emlines = configs['OUTDIR'] + configs['ROOTNAME'] + '_emlines.fits'
#     with fits.open(emlines) as hdu_emis:
#         ems = Table(hdu_emis['EMLDATA_DATA'].data)
#
#     # This is to include SFH results, NOT TESTED!
#     with fits.open(configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf_SFH.fits') as hdu_ppxf_sfh:
#         ppxf_sfh_data = hdu_ppxf_sfh[1].data
#         masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err = util_sfh_quantities.compute_sfh_relevant_quantities(
#             configs)
#         reddening = ppxf_sfh_data['REDDENING']
#         st_props = masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err, reddening
#
#     exit()
#
#     return ems, st_props
#
#     spectra_muse_err, ln_lam_gal, velscale = util.log_rebin(lam=spec_dict['lam_range'],
#                                                             spec=spec_dict['spec_flux_err'], velscale=velscale)
#
#     # print(sum(np.isnan(spec_dict['spec_flux'])))
#     # print(sum(np.isnan(spectra_muse)))
#     #
#     # plt.plot(ln_lam_gal, spectra_muse_err)
#     # plt.show()
#
#     lsf_dict = {"lam": spec_dict['lam'], "fwhm": spec_dict['lsf']}
#     # get new wavelength array
#     lam_gal = np.exp(ln_lam_gal)
#     # goodpixels = util.determine_goodpixels(ln_lam=ln_lam_gal, lam_range_temp=spec_dict['lam_range'], z=redshift)
#     goodpixels = None
#     # goodpixels = (np.isnan(spectra_muse) + np.isnan(spectra_muse_err) + np.isinf(spectra_muse) + np.isinf(spectra_muse_err))
#     # print(sum(np.invert(np.isnan(spectra_muse) + np.isnan(spectra_muse_err) + np.isinf(spectra_muse) + np.isinf(spectra_muse_err))))
#     # print(sum(((spectra_muse > 0) & (spectra_muse < 100000000000000))))
#
#     # get stellar library
#     ppxf_dir = path.dirname(path.realpath(lib.__file__))
#     basename = f"spectra_{sps_name}_9.0.npz"
#     filename = path.join(ppxf_dir, 'sps_models', basename)
#     if not path.isfile(filename):
#         url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
#         request.urlretrieve(url, filename)
#
#     sps = lib.sps_lib(filename=filename, velscale=velscale, fwhm_gal=lsf_dict, norm_range=[5070, 5950],
#                       wave_range=None,
#                       age_range=age_range, metal_range=metal_range)
#     reg_dim = sps.templates.shape[1:]  # shape of (n_ages, n_metal)
#     stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
#
#     gas_templates, gas_names, line_wave = util.emission_lines(ln_lam_temp=sps.ln_lam_temp,
#                                                               lam_range_gal=spec_dict['lam_range'],
#                                                               FWHM_gal=get_MUSE_polyFWHM)
#
#     templates = np.column_stack([stars_templates, gas_templates])
#
#     n_star_temps = stars_templates.shape[1]
#     component = [0] * n_star_temps
#     for line_name in gas_names:
#         if '[' in line_name:
#             component += [2]
#         else:
#             component += [1]
#
#     gas_component = np.array(component) > 0  # gas_component=True for gas templates
#
#     moments = [4, 4, 4]
#
#     vel = speed_of_light_kmps * np.log(1 + redshift)  # eq.(8) of Cappellari (2017)
#     start_gas = [vel, 150., 0, 0]  # starting guess
#     start_star = [vel, 150., 0, 0]
#     print(start_gas)
#     start = [start_star, start_gas, start_gas]
#
#     pp = ppxf(templates=templates, galaxy=spectra_muse, noise=spectra_muse_err, velscale=velscale, start=start,
#               moments=moments, degree=-1, mdegree=4, lam=lam_gal, lam_temp=sps.lam_temp,  #regul=1/rms,
#               reg_dim=reg_dim, component=component, gas_component=gas_component,  #reddening=0,
#               gas_names=gas_names, goodpixels=goodpixels)
#
#     light_weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
#     light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
#     light_weights /= light_weights.sum()  # Normalize to light fractions
#
#     # light_weights = pp.weights[~gas_component]      # Exclude weights of the gas templates
#     # light_weights = light_weights.reshape(reg_dim)
#
#     ages, met = sps.mean_age_metal(light_weights)
#     mass2light = sps.mass_to_light(light_weights, redshift=redshift)
#
#     return {'pp': pp, 'ages': ages, 'met': met, 'mass2light': mass2light}
#
#     # wavelength = pp.lam
#     # total_flux = pp.galaxy
#     # total_flux_err = pp.noise
#     #
#     # best_fit = pp.bestfit
#     # gas_best_fit = pp.gas_bestfit
#     # continuum_best_fit = best_fit - gas_best_fit
#     #
#     # plt.errorbar(wavelength, total_flux, yerr=total_flux_err)
#     # plt.plot(wavelength, continuum_best_fit + gas_best_fit)
#     # plt.plot(wavelength, gas_best_fit)
#     # plt.show()
#     #
#     #
#     #
#     #
#     # plt.figure(figsize=(17, 6))
#     # plt.subplot(111)
#     # pp.plot()
#     # plt.show()
#     #
#     # plt.figure(figsize=(9, 3))
#     # sps.plot(light_weights)
#     # plt.title("Light Weights Fractions");
#     # plt.show()
#     #
#     # exit()