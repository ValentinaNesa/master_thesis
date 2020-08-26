import numpy as np
from climada.engine import Impact
from scipy.sparse import csr_matrix

import sys
sys.path.append('../../src/write_entities/')
from define_if import call_impact_functions
from define_hazard import call_hazard

# import for calc_mortality function to make it climada friendly
from climada.entity.exposures.base import INDICATOR_IF, INDICATOR_CENTR
import logging
LOGGER = logging.getLogger(__name__)
from scipy import sparse
from climada.util.config import CONFIG


# This function calls a random hazard and impact function and take the given exposures to calculate an impact.

def calculate_impact_mortality(directory_hazard, scenario, year, exposures, uncertainty_variable='all',
                     kanton=None, save_median_mat=False):
    """compute the impacts once:

                Parameters:
                    directory_hazard (str): directory to a folder containing one tasmax folder with all the data files
                    scenario (str): scenario for which to compute the hazards
                    year(str): year for which to compute the hazards
                    exposures(Exposures): the exposures which stay fixed for all runs
                    uncertainty_variable(str): variable for which to consider the uncertainty. Default: 'all'
                    kanton (str or None): Name of canton. Default: None (all of Switzerland)
                    save_median_mat (bool): rather we save the impact matrix . Default = True

                Returns:
                    Dictionary of impact loss and dictionary of impact matrices if specified
                      """

    impact_dict = {}
    matrices = {} if save_median_mat else None

    hazard = call_hazard(directory_hazard, scenario, year, uncertainty_variable=uncertainty_variable, kanton=kanton)
    ####################################################################################################

    error = uncertainty_variable == 'impactfunction' or uncertainty_variable == 'all' # this sentence evaluates to the correct boolean

    if_hw_set = call_impact_functions(error)

    for key, grid in exposures.items():  # calculate impact for each type of exposure
        impact = Impact()
        #impact.calc(grid, if_hw_set, hazard['heat'], save_mat=save_median_mat)
        calc_mortality(impact, grid, if_hw_set, hazard['heat'], save_mat=save_median_mat)

        impact_dict[key] = np.sum(impact.at_event)

        if save_median_mat:
            matrices[key] = csr_matrix(impact.imp_mat.sum(axis=0))
        # sum all events to get one 1xgridpoints matrix per type of exposures

    del hazard

    if save_median_mat:
        output = [impact_dict, matrices]
    else:
        output = [impact_dict]

    return output

###############################################################################

# modify Impact object

def calc_mortality(impact, exposures, impact_funcs, hazard, save_mat=False):
        """Compute impact of an hazard to exposures.

        Parameters:
            impact (Impact): impact, object to be modified
            exposures (Exposures): exposures
            impact_funcs (ImpactFuncSet): impact functions
            hazard (Hazard): hazard
            self_mat (bool): self impact matrix: events x exposures
        """
        # 1. Assign centroids to each exposure if not done
        assign_haz = INDICATOR_CENTR + hazard.tag.haz_type
        if assign_haz not in exposures:
            exposures.assign_centroids(hazard)
        else:
            LOGGER.info('Exposures matching centroids found in %s', assign_haz)

        # 2. Initialize values
        impact.unit = exposures.value_unit
        impact.event_id = hazard.event_id
        impact.event_name = hazard.event_name
        impact.date = hazard.date
        impact.coord_exp = np.stack([exposures.latitude.values,
                                   exposures.longitude.values], axis=1)
        impact.frequency = hazard.frequency
        impact.at_event = np.zeros(hazard.intensity.shape[0])
        impact.eai_exp = np.zeros(exposures.value.size)
        impact.tag = {'exp': exposures.tag, 'if_set': impact_funcs.tag,
                    'haz': hazard.tag}
        impact.crs = exposures.crs

        # Select exposures with positive value and assigned centroid
        exp_idx = np.where(np.logical_and(exposures.value > 0, \
                           exposures[assign_haz] >= 0))[0]
        if exp_idx.size == 0:
            LOGGER.warning("No affected exposures.")

        num_events = hazard.intensity.shape[0]
        LOGGER.info('Calculating damage for %s assets (>0) and %s events.',
                    exp_idx.size, num_events)

        # Get damage functions for this hazard
        if_haz = INDICATOR_IF + hazard.tag.haz_type
        haz_imp = impact_funcs.get_func(hazard.tag.haz_type)
        if if_haz not in exposures and INDICATOR_IF not in exposures:
            LOGGER.error('Missing exposures impact functions %s.', INDICATOR_IF)
            raise ValueError
        if if_haz not in exposures:
            LOGGER.info('Missing exposures impact functions for hazard %s. ' +\
                        'Using impact functions in %s.', if_haz, INDICATOR_IF)
            if_haz = INDICATOR_IF

        # Check if deductible and cover should be applied
        insure_flag = False
        if ('deductible' in exposures) and ('cover' in exposures) \
        and exposures.cover.max():
            insure_flag = True

        if save_mat:
            impact.imp_mat = sparse.lil_matrix((impact.date.size, exposures.value.size))

        # 3. Loop over exposures according to their impact function
        tot_exp = 0
        for imp_fun in haz_imp:
            # get indices of all the exposures with this impact function
            exp_iimp = np.where(exposures[if_haz].values[exp_idx] == imp_fun.id)[0]
            tot_exp += exp_iimp.size
            exp_step = int(CONFIG['global']['max_matrix_size']/num_events)
            if not exp_step:
                LOGGER.error('Increase max_matrix_size configuration parameter'
                             ' to > %s', str(num_events))
                raise ValueError
            # separte in chunks
            chk = -1
            for chk in range(int(exp_iimp.size/exp_step)):
                exp_impact_mortality(impact, \
                    exp_idx[exp_iimp[chk*exp_step:(chk+1)*exp_step]],\
                    exposures, hazard, imp_fun, insure_flag)
            exp_impact_mortality(impact, exp_idx[exp_iimp[(chk+1)*exp_step:]],\
                exposures, hazard, imp_fun, insure_flag)

        if not tot_exp:
            LOGGER.warning('No impact functions match the exposures.')
        impact.aai_agg = sum(impact.at_event * hazard.frequency)

        if save_mat:
            impact.imp_mat = impact.imp_mat.tocsr()
            
###############################################################################
            
def exp_impact_mortality(impact, exp_iimp, exposures, hazard, imp_fun, insure_flag):
    """Compute impact for inpute exposure indexes and impact function.
    
    Parameters:
        impact (Impact): impact, object to be modified
        exp_iimp (np.array): exposures indexes
        exposures (Exposures): exposures instance
        hazard (Hazard): hazard instance
        imp_fun (ImpactFunc): impact function instance
        insure_flag (bool): consider deductible and cover of exposures
    """
    if not exp_iimp.size:
        return
    
    # get assigned centroids
    icens = exposures[INDICATOR_CENTR + hazard.tag.haz_type].values[exp_iimp]
    
    # get affected intensities
    inten_val = hazard.intensity[:, icens]
    # get affected fractions
    fract = hazard.fraction[:, icens]
    # impact = fraction * mdr * value
    inten_val.data = imp_fun.calc_mdr(inten_val.data)
    
    # print('\nNEW')
    # print('inten_val', inten_val)
    # print('fract', fract)
    # print('exposures.value.values[exp_iimp]', exposures.value.values[exp_iimp])
    
    matrix = fract.multiply(inten_val).multiply(exposures.value.values[exp_iimp])
    # matrix = impact_mortality(impact)
    
    if insure_flag and matrix.nonzero()[0].size:
        inten_val = hazard.intensity[:, icens].todense()
        paa = np.interp(inten_val, imp_fun.intensity, imp_fun.paa)
        matrix = np.minimum(np.maximum(matrix - \
            exposures.deductible.values[exp_iimp] * paa, 0), \
            exposures.cover.values[exp_iimp])
        impact.eai_exp[exp_iimp] += np.sum(np.asarray(matrix) * \
            hazard.frequency.reshape(-1, 1), axis=0)
    else:
        impact.eai_exp[exp_iimp] += np.squeeze(np.asarray(np.sum( \
            matrix.multiply(hazard.frequency.reshape(-1, 1)), axis=0)))
    
    impact.at_event += np.squeeze(np.asarray(np.sum(matrix, axis=1)))
    impact.tot_value += np.sum(exposures.value.values[exp_iimp])
    if not isinstance(impact.imp_mat, list):
        impact.imp_mat[:, exp_iimp] = matrix
        
###############################################################################

# def impact_mortality(impact):
#     print('IMPACT')
#     print(type(impact))
#     print(impact)
#     return matrix