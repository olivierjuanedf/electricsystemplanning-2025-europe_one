from typing import Dict, List, Union

import pandas as pd
import numpy as np

from common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS
from common.fuel_sources import FuelSource, FuelNames, DummyFuelNames
from include.dataset_builder import GenerationUnitData

GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]
gps_coords = (51.1657, 10.4515)


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
    n_ts = len(wind_on_shore_cf_data['value'].values)

    generators = [
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hard-coal',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.coal,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 12786.24,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost * 0.37,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_gas',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.gas,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 32541.3,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.gas].primary_cost * 0.5,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.5
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_oil',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.oil,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 2821.33,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.oil].primary_cost * 0.4,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other-non-renewables',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_non_renewables,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 9080.02,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.other_non_renewables].primary_cost * 0.4,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind-on-shore',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 69017.4,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_on_shore_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind-off-shore',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 11105.0,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_off_shore_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_solar-pv',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.solar,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 88447.85,
            GEN_UNITS_PYPSA_PARAMS.max_power_pu: solar_pv_cf_data['value'].values,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.solar].primary_cost
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other-renewables',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_renewables,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 11056.79,
        },
        # QUESTION: what is this - very necessary - last fictive asset?
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_failure',
            GEN_UNITS_PYPSA_PARAMS.carrier: DummyFuelNames.ac,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 1e10,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: 1e5
        },
        {
            GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_lignite',
            GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.lignite,
            GEN_UNITS_PYPSA_PARAMS.nominal_power: 14687.0,
            GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.lignite].primary_cost * 0.37,
            GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
        },
  # this is a battery
        {GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_battery',
         GEN_UNITS_PYPSA_PARAMS.carrier: 'Flexibility', 
         GEN_UNITS_PYPSA_PARAMS.nominal_power: 4377.82,
         GEN_UNITS_PYPSA_PARAMS.min_power_pu: -1,
         GEN_UNITS_PYPSA_PARAMS.max_power_pu: 1,
         GEN_UNITS_PYPSA_PARAMS.max_hours: 2,
         GEN_UNITS_PYPSA_PARAMS.soc_init: 1000,
         GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.95,
         GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.95
        },
          # this is an hydro reservoir - with very fictive values!
        {GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro-reservoir',
         GEN_UNITS_PYPSA_PARAMS.carrier: 'Hydro', 
         GEN_UNITS_PYPSA_PARAMS.nominal_power: 10000,
         GEN_UNITS_PYPSA_PARAMS.max_hours: 1000,
         GEN_UNITS_PYPSA_PARAMS.soc_init: 1000000,
         GEN_UNITS_PYPSA_PARAMS.inflow: np.ones(n_ts)
        },  

    ]
    return generators

def set_gen_as_list_of_gen_units_data(generators: List[GENERATOR_DICT_TYPE]) -> List[GenerationUnitData]:
    # add type of units
    for elt_gen in generators:
        elt_gen['type'] = f'{elt_gen["carrier"]}_agg'
    # then cas as list of GenerationUnitData objects
    return [GenerationUnitData(**elt_gen_dict) for elt_gen_dict in generators]
