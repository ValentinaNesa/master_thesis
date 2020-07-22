import pickle
from ast import literal_eval
import sys

# append the pathways to all modules used in the model
sys.path.append('../../src/impact_calculation')
sys.path.append('../../src/write_entities')
sys.path.append('../../src/util/')

from impact_monte_carlo_parallel import impact_monte_carlo


def convert(string):  # function that converts 'lists' from the bash input (strings) to python lists
    li = list(string.split(","))
    return li


directory_output = '../../output/impact_ch/'  # where to save to output
directory_hazard = '../../input_data/hazard/CH2018'  # first input from the bash script, which is the directory to the temperature files.

n_mc = literal_eval('1')  # number of Monte Carlo runs

# check the third input, which determines if the input should be calculated for Switzerland,
# all cantons indepentently or for one specific canton:
kantons = ['Zürich']
directory_output = '../../output/impact_cantons/'

# get fourth input, the years for which to compute the impact
years_list = [2050]

# get fifth input, the scenarios for which to compute the impact
scenarios = ['RCP45'] #On the computer: CH2018 data only for the RCP4.5 scenario !!

# check if any age groups were given, or if the impact for all categories should be computed
age_group = None
groups_str = 'all_age_groups'

# determine if the median damage matrix should be saved as output
save_median_mat = True

# in this base model run, all uncertainties are taken into account.
# This is not the case in the sensibility testing code where all are taken one by one.
uncertainty_variables_list = ['all']
uncertainty = 'all_uncertainties'

for kanton in kantons:  # loop through given kantons, one file per element in the kantons loop will be produced.
    # If cantons only contains None, only one file corresponding to all of Switzerland is produced,
    # otherwise one per canton will be written.

    if kanton is None:
        kanton_name = 'CH'
    else:
        kanton_name = kanton

    # compute the impact. impact[0] is the loss for each category and Monte Carlo run, impact[0] is the impact matrix
    # for each category and Monte Carlo run

    IMPACT = impact_monte_carlo(directory_hazard, scenarios, years_list, n_mc,
                                uncertainty_variables_list=uncertainty_variables_list, kanton=kanton,
                                age_group=age_group, save_median_mat=save_median_mat)

    with open(''.join([directory_output, 'loss_', groups_str, '_', str(n_mc), 'mc_',
                       uncertainty, '_', kanton_name, '.pickle']), 'wb') as handle:
        pickle.dump(IMPACT[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    if save_median_mat:
        with open(''.join([directory_output, 'matrix_',
                           groups_str, '_', str(n_mc), 'mc_', uncertainty, '_', kanton_name,
                           '.pickle']) , 'wb') as handle:
            pickle.dump(IMPACT[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
