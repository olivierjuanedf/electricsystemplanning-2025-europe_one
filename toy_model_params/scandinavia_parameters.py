from typing import Dict, List, Union

import pandas as pd

from common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS
from common.fuel_sources import FuelSource, FuelNames, DummyFuelNames
from include.dataset_builder import GenerationUnitData

GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]
gps_coords = (18.0686, 59.3293)

import numpy as np

def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSource], wind_on_shore_cf_data: pd.DataFrame,
                   wind_off_shore_cf_data: pd.DataFrame, solar_pv_cf_data: pd.DataFrame) -> List[GENERATOR_DICT_TYPE]:
    """
    Get list of generators to be set on a given node of a PyPSA model
    :param country_trigram: name of considered country, as a trigram (ex: "ben", "fra", etc.)
    :param fuel_sources
    :param wind_on_shore_cf_data
    :param wind_off_shore_cf_data
    :param solar_pv_cf_data
    N.B.
    (i) Better in this function to use CONSTANT names of the different fuel sources to avoid trouble
    in the code (i.e. GEN_UNITS_PYPSA_PARAMS, FuelNames and DummyFuelNames dataclasses = sort of dict.). If you prefer
    to directly use str you can Ctrl+click on the constants below and see the corresponding str (e.g.,
    'name' for GEN_UNITS_PYPSA_PARAMS.name)
    (ii) When default PyPSA values have to be used for the generator parameters they are not provided below -> e.g.,
    efficiency=1, committable=False (i.e., not switch on/off integer variables in the model),
    min_power_pu/max_power_pu=0/1, marginal_cost=0
    -> see field 'generator_params_default_vals' in file input/long_term_uc/pypsa_static_params.json
    """
    # get number of time-slots based on length of CF data 

    n_ts = len(wind_on_shore_cf_data['value'].values) 
    generators = [
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind_on_shore',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 39318.31,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_on_shore_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind_off_shore',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 2447.3,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_off_shore_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_solar_pv',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.solar,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 12108.45,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: solar_pv_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.solar].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro_run_of_river',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'hydro',
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 7204.48,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 1.0,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro_reservoir',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'Hydro',  # or FuelNames.hydro
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 44482.06,
            GEN_UNITS_PYPSA_PARAMS.max_hours: 2026, 
            GEN_UNITS_PYPSA_PARAMS.soc_init: 109136356.65,
            GEN_UNITS_PYPSA_PARAMS.inflow: np.ones(n_ts)
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_nuclear',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.uranium,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 11277,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost / 0.37,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_biofuel',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.biomass,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 3273.41,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost / 0.37,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hard_coal',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.coal,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 866.6,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost / 0.37,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_gas',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.gas,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 3324.86,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.gas].primary_cost / 0.5,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.5
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_oil',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.oil,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 1580,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.oil].primary_cost / 0.4,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
        },
          # this is a battery 

        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_batteries', 
            GEN_UNITS_PYPSA_PARAMS.carrier: 'Flexibility',  
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 4000, 
            GEN_UNITS_PYPSA_PARAMS.min_power_pu: -1, 
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: 1, 
            GEN_UNITS_PYPSA_PARAMS.max_hours: 2, 
            GEN_UNITS_PYPSA_PARAMS.soc_init: 1000, 
            GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.95,
            GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.95 
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_dsr',
            GEN_UNITS_PYPSA_PARAMS.carrier: 'dsr', # or FuelNames.demand_side_response
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 10524.60,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 1000.0  # DSR is usually very expensive (only used at peak)
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other_renewables',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_renewables,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 6988.23,
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other_non_renewables',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_non_renewables,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 735.5,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.other_non_renewables].primary_cost / 0.4,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
        },
        # {
        #     GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro_phs_open',
        #     GEN_UNITS_PYPSA_PARAMS.carrier: 'hydro',
        #     GEN_UNITS_PYPSA_PARAMS.nominal_power: 709.32,
        #     GEN_UNITS_PYPSA_PARAMS.efficiency: 0.75, # Typical round-trip efficiency
        #     GEN_UNITS_PYPSA_PARAMS.marginal_cost: 0.0
        # },
        # QUESTION: what is this - very necessary - last fictive asset?
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_failure',
            GEN_UNITS_PYPSA_PARAMS.carrier: DummyFuelNames.ac,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 1e10,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 1e5
        }
    ]
    return generators


def set_gen_as_list_of_gen_units_data(generators: List[GENERATOR_DICT_TYPE]) -> List[GenerationUnitData]:
    # add type of units
    for elt_gen in generators:
        elt_gen['type'] = f'{elt_gen["carrier"]}_agg'
    # then cas as list of GenerationUnitData objects
    return [GenerationUnitData(**elt_gen_dict) for elt_gen_dict in generators]


# ---------------------------------------------------------------------



# from typing import Dict, List, Union

# import pandas as pd

# from common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS
# from common.fuel_sources import FuelSource, FuelNames, DummyFuelNames
# from include.dataset_builder import GenerationUnitData

# GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]
# gps_coords = (18.0686, 59.3293)


# def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSource], wind_on_shore_cf_data: pd.DataFrame,
#                    wind_off_shore_cf_data: pd.DataFrame, solar_pv_cf_data: pd.DataFrame) -> List[GENERATOR_DICT_TYPE]:
#     """
#     Get list of generators to be set on a given node of a PyPSA model
#     :param country_trigram: name of considered country, as a trigram (ex: "ben", "fra", etc.)
#     :param fuel_sources
#     :param wind_on_shore_cf_data
#     :param wind_off_shore_cf_data
#     :param solar_pv_cf_data
#     N.B.
#     (i) Better in this function to use CONSTANT names of the different fuel sources to avoid trouble
#     in the code (i.e. GEN_UNITS_PYPSA_PARAMS, FuelNames and DummyFuelNames dataclasses = sort of dict.). If you prefer
#     to directly use str you can Ctrl+click on the constants below and see the corresponding str (e.g.,
#     'name' for GEN_UNITS_PYPSA_PARAMS.name)
#     (ii) When default PyPSA values have to be used for the generator parameters they are not provided below -> e.g.,
#     efficiency=1, committable=False (i.e., not switch on/off integer variables in the model),
#     min_power_pu/max_power_pu=0/1, marginal_cost=0
#     -> see field 'generator_params_default_vals' in file input/long_term_uc/pypsa_static_params.json
#     """
#     generators = [
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hard-coal',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.coal,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 866.6,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost / 0.37,
#             GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_biofuel',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.biomass,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 3273.41,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost / 0.37,
#             GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_nuclear',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.uranium,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 11277,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost / 0.37,
#             GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_gas',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.gas,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 3324.86,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.gas].primary_cost / 0.5,
#             GEN_UNITS_PYPSA_PARAMS.efficiency: 0.5
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_oil',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.oil,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 1580,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.oil].primary_cost / 0.4,
#             GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other-non-renewables',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_non_renewables,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 735.5,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.other_non_renewables].primary_cost / 0.4,
#             GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind-on-shore',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 39318.31,
#             GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_on_shore_cf_data['value'].values,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind-off-shore',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 2447.3,
#             GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_off_shore_cf_data['value'].values,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_solar-pv',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.solar,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 12108.45,
#             GEN_UNITS_PYPSA_PARAMS.max_power_pu: solar_pv_cf_data['value'].values,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.solar].primary_cost
#         },
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other-renewables',
#             GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_renewables,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 6988.23,
#         },
#         # QUESTION: what is this - very necessary - last fictive asset?
#         {
#             GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_failure',
#             GEN_UNITS_PYPSA_PARAMS.carrier: DummyFuelNames.ac,
#             GEN_UNITS_PYPSA_PARAMS.nominal_power: 1e10,
#             GEN_UNITS_PYPSA_PARAMS.marginal_cost: 1e5
#         }
#     ]
#     return generators


# def set_gen_as_list_of_gen_units_data(generators: List[GENERATOR_DICT_TYPE]) -> List[GenerationUnitData]:
#     # add type of units
#     for elt_gen in generators:
#         elt_gen['type'] = f'{elt_gen["carrier"]}_agg'
#     # then cas as list of GenerationUnitData objects
#     return [GenerationUnitData(**elt_gen_dict) for elt_gen_dict in generators]
