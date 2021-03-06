import numpy as np
import pandas as pd
from pandas import DataFrame

from climada.entity import Exposures

import sys
sys.path.append('../../src/util/')
from shapefile_masks import vector_shapefile_mask


# In[ ]:

def call_exposures(kanton=None, age_group=None, epsg_output=4326):
    """write the Exposures:

                    Parameters:
                        kanton (str or None): Name of canton. Default: None (all of Switzerland)
                        age_group (str or None): specific age group, as given in the "GIS_Data_code" of the age_categories.csv file. Default: None
                        epsg_output (int): EPSG code of the output. Default: 4326.

                    Returns:
                        Dictionary containing one Exposure per age category (ratio of pop. per hectare instead of the number of people)
                          """

    directory = '../../input_data/exposures/'
    exposures = {}  # dictionary of the exposures, where we will further put each category of Exposure as a key
    
    population_info = pd.read_csv(
        ''.join([directory, 'age_categories.csv']))  # file containing the information on the age categories
    
    population_loc = pd.read_csv(''.join([directory, 'STATPOP2018.csv']))
    # file containing the geographical location of the population by age group
    
    epsg_data = 2056  # espg of the population_loc data
    
    # get tot. population (CH/Canton)
    pop_values = population_loc[population_loc.columns[8:]]
    pop_hectare_ch = pop_values.sum(axis=1) # to sum over the columns
    pop_tot_ch = pop_hectare_ch.sum(axis=0) # to sum over the rows
    
    if kanton:
        shp_dir = '../../input_data/shapefiles/KANTONS_projected_epsg4326/' \
                      'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET_epsg4326.shp'
        
        pop_loc_ch = population_loc.copy()
        pop_loc_ch['longitude'] = np.asarray(pop_loc_ch['E_KOORD']).flatten()
        pop_loc_ch['latitude'] = np.asarray(pop_loc_ch['N_KOORD']).flatten()
        pop_loc_canton = vector_shapefile_mask(pop_loc_ch, shp_dir, kanton, epsg_data,
                                                          epsg_output)
                
        pop_values_canton = pop_loc_canton[pop_loc_canton.columns[8:]]
        pop_hectare_canton = pop_values_canton.sum(axis=1)
        pop_tot_canton = pop_hectare_canton.sum(axis=0)
        
        #print(pop_tot_canton)
        #print(pop_tot_ch)
        
    # get subset of the population data for each category
    # Under 75 years:
    U = population_info.loc[population_info['Age_category'] == 'U']
    # Above 75 years
    O = population_info.loc[population_info['Age_category'] == 'O']

    if age_group is None:
        groups = list(population_loc)[8:]  # take all age_groups
    else:
        groups = age_group # take only the given age_groups

    age_type = {}
    if_ref = {}
    exposures_name = set()

    for group in groups:
        category = population_info[population_info['GIS_Data_code'] == group]['Age_category'].values[0]

        if category == 'U':
            exposures_name.add('Under 75 years')
            age_type['Under 75 years'] = U
            if_ref['Under 75 years'] = 1

        if category == 'O':
            exposures_name.add('Over 75 years')
            age_type['Over 75 years'] = O
            if_ref['Over 75 years'] = 2

    for name in exposures_name:

        code_i_l = ['E_KOORD', 'N_KOORD']
        if age_group is None:
            code_i_l.extend(list(age_type[name]['GIS_Data_code']))
        else:
            code_i_l.extend(age_group)

        population_sum_intensity = DataFrame()  # dataframe with ratio of the pop. for each category
        population_loc_intensity = population_loc[code_i_l]

        population_sum_intensity['longitude'] = np.asarray(population_loc_intensity['E_KOORD']).flatten()
        population_sum_intensity['latitude'] = np.asarray(population_loc_intensity['N_KOORD']).flatten()
        population_sum_intensity['value'] = np.asarray(
            population_loc_intensity[population_loc_intensity.columns[2:]].sum(axis=1) / pop_tot_ch)
        n_exp = len(population_sum_intensity['value'])

        if kanton:  # test if a canton was specified, in that case
            # we first get a panda geodataframe and define the exposures slightly differently
            shp_dir = '../../input_data/shapefiles/KANTONS_projected_epsg4326/' \
                      'swissBOUNDARIES3D_1_3_TLM_KANTONSGEBIET_epsg4326.shp'

            population_sum_intensity = vector_shapefile_mask(population_sum_intensity, shp_dir, kanton, epsg_data,
                                                          epsg_output)

            population_sum_intensity['value'] = population_sum_intensity['value'] * pop_tot_ch / pop_tot_canton

            population_sum_intensity = Exposures(population_sum_intensity)  # define as Exposure class
            population_sum_intensity.set_lat_lon()
            n_exp = len(population_sum_intensity['value'])
            population_sum_intensity['if_heat'] = np.full((n_exp), if_ref[name], dtype=int)
            population_sum_intensity.value_unit = 'Number of people'
            population_sum_intensity.fillna(0)
            population_sum_intensity.check()

        else:  # normal case, for entire Switzerland

            population_sum_intensity = Exposures(population_sum_intensity)
            population_sum_intensity.set_geometry_points()
            population_sum_intensity.value_unit = 'Number of people'
            population_sum_intensity['if_heat'] = np.full((n_exp), if_ref[name], dtype=int)
            population_sum_intensity.crs = {'init': ''.join(['epsg:', str(epsg_data)])} # crs: Coordinate Reference Systems
            population_sum_intensity.check()
            population_sum_intensity.fillna(0)
            population_sum_intensity.to_crs(epsg=epsg_output, inplace=True)
        
        name_category = name
        exposures[name_category] = population_sum_intensity

    return exposures