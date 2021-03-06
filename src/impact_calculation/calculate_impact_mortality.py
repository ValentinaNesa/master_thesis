import numpy as np
import pandas as pd
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

    error = uncertainty_variable == 'impactfunction' or uncertainty_variable == 'all'  # this sentence evaluates to the correct boolean
    if_hw_set = call_impact_functions(error)
    
    for key, grid in exposures.items():  # calculate impact for each type of exposure
        impact = Impact()
        
        calc_mortality(impact, key, grid, if_hw_set, hazard['heat'], kanton=kanton, save_mat=save_median_mat)

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

def calc_mortality(impact, key, exposures, impact_funcs, hazard, kanton, save_mat=False):
    """Compute impact of an hazard to exposures.

    Parameters:
        impact (Impact): impact, object to be modified
        key (str): exposures names
        exposures (Exposures): exposures
        impact_funcs (ImpactFuncSet): impact functions
        hazard (Hazard): hazard
        kanton (str or None): Name of canton. Default: None (all of Switzerland)
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
        LOGGER.info('Missing exposures impact functions for hazard %s. ' + \
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
        exp_step = int(CONFIG['global']['max_matrix_size'] / num_events)
        if not exp_step:
            LOGGER.error('Increase max_matrix_size configuration parameter'
                         ' to > %s', str(num_events))
            raise ValueError
        # separte in chunks
        chk = -1
        for chk in range(int(exp_iimp.size / exp_step)):
            exp_impact_mortality(impact, \
                                 exp_idx[exp_iimp[chk * exp_step:(chk + 1) * exp_step]], \
                                 exposures, key, hazard, imp_fun, insure_flag, kanton)
        exp_impact_mortality(impact, exp_idx[exp_iimp[(chk + 1) * exp_step:]], \
                             exposures, key, hazard, imp_fun, insure_flag, kanton)

    if not tot_exp:
        LOGGER.warning('No impact functions match the exposures.')
    impact.aai_agg = sum(impact.at_event * hazard.frequency)

    if save_mat:
        impact.imp_mat = impact.imp_mat.tocsr()


###############################################################################

def exp_impact_mortality(impact, exp_iimp, exposures, key, hazard, imp_fun, insure_flag, kanton):
    """Compute impact for inpute exposure indexes and impact function.
    
    Parameters:
        impact (Impact): impact, object to be modified
        exp_iimp (np.array): exposures indexes
        exposures (Exposures): exposures instance
        key (str): exposures names
        hazard (Hazard): hazard instance
        imp_fun (ImpactFunc): impact function instance
        insure_flag (bool): consider deductible and cover of exposures
        kanton (str or None): Name of canton. Default: None (all of Switzerland)
    """
    if not exp_iimp.size:
        return   
    
    if kanton is None:
        kanton_name = 'CH'
    else:
        kanton_name = kanton
    
    directory = '../../input_data/impact_functions/'
    
    annual_deaths = pd.read_excel(''.join([directory, 'annual_deaths.xlsx']), sheet_name = key)
    # file containing the number of annual deaths per CH / Canton for each age category
    
    # PREPROCESSING STEP:
        
    # get assigned centroids
    icens = exposures[INDICATOR_CENTR + hazard.tag.haz_type].values[exp_iimp]
    # get affected intensities
    temperature_matrix = hazard.intensity[:, icens] # intensity of the hazard
    # get affected fractions
    fract = hazard.fraction[:, icens]  # frequency of the hazard
    # get exposure values
    exposure_values = exposures.value.values[exp_iimp] 

    expected_deaths = {}
    daily_deaths = annual_deaths[annual_deaths['Canton'] == kanton_name]['Annual_deaths'].values[0] / 365
    max_temp =  temperature_matrix.max()
    for value in range(22, int(np.ceil(max_temp)) + 1):
        expected_deaths[value] = daily_deaths / imp_fun.calc_mdr(value)
    #print(expected_deaths)

    # Compute impact matrix
    matrix = impact_mortality(temperature_matrix, exposure_values, icens, expected_deaths, imp_fun, fract.shape)

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

def impact_mortality(temperature_matrix, exposure_values, indices_of_centroids, expected_deaths, imp_fun, shape):

    mat = csr_matrix(shape)
    num_of_days = shape[0]
    num_of_cells = shape[1]

    # print('CELLS: ', num_of_cells)
    # print('DAYS: ', num_of_days)

    for cell in range(num_of_cells):
        temperature_occurrence_index = 0
        attributable_fraction_index = 1

        # 1: Create histogram
        histogram = {}

        for day in range(num_of_days):
            temperature_value = int(round(temperature_matrix[day, cell]))
            if temperature_value > 21.5:
                if temperature_value in histogram:
                    histogram[temperature_value][temperature_occurrence_index] += 1
                else:
                    histogram[temperature_value] = [0, 0]
                    histogram[temperature_value][temperature_occurrence_index] = 1

        # 2: Aggregate expected deaths according to cell histogram:
        average_deaths = 0
        for key in histogram.keys():
            average_deaths += expected_deaths[key] * exposure_values[cell]
        #print(average_deaths)

        # 3: Attributable Fraction:
        for key in histogram.keys():
            value = exposure_values[cell] * (imp_fun.calc_mdr(key) - 1)
            histogram[key][attributable_fraction_index] = value / (value + 1)

        # 4: Attributable Deaths:
        total_attributable_deaths = 0
        for key in histogram.keys():
            days_amount = histogram[key][temperature_occurrence_index]
            attributable_fraction = histogram[key][attributable_fraction_index]
            total_attributable_deaths += days_amount * attributable_fraction * average_deaths
        #print(total_attributable_deaths)

        # Store the proper information in the first day only (since we already aggregated per each day)
        mat[0, cell] = total_attributable_deaths

    return mat
