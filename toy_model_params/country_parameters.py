from typing import Dict, List, Union

import pandas as pd
import numpy as np
from common.constants.pypsa_params import GEN_UNITS_PYPSA_PARAMS
from common.fuel_sources import FuelSource, FuelNames, DummyFuelNames
from include.dataset_builder import GenerationUnitData

GENERATOR_DICT_TYPE = Dict[str, Union[float, int, str]]
gps_coords = (12.5674, 41.8719)


def get_generators(country_trigram: str, fuel_sources: Dict[str, FuelSource], wind_onshore_cf_data, wind_offshore_cf_data, solar_pv_cf_data) -> List[dict]:
    """
    Get list of generators to be set on a given node of a PyPSA model
    :param country_trigram: name of considered country, as a trigram (ex: "ben", "fra", etc.)
    :param fuel_sources
    """
    # List to be completed
    n_ts = len(wind_onshore_cf_data['value'].values)
    generators = [
        {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_nuclear',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.uranium,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 4415,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.uranium].primary_cost * 0.33,
                GEN_UNITS_PYPSA_PARAMS.efficiency: 0.33
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hard-coal',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.coal,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 2671,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.coal].primary_cost * 0.37,
                GEN_UNITS_PYPSA_PARAMS.efficiency: 0.37
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_gas',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.gas,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 18500,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.gas].primary_cost * 0.5,
                GEN_UNITS_PYPSA_PARAMS.efficiency: 0.5
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_biomass',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.biomass,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 2088,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.gas].primary_cost * 0.5,
                GEN_UNITS_PYPSA_PARAMS.efficiency: 0.5
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_oil',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.oil,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 140,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.oil].primary_cost * 0.4,
                GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other-non-renewables',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_non_renewables,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 5706,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.other_non_renewables].primary_cost * 0.4,
                GEN_UNITS_PYPSA_PARAMS.efficiency: 0.4
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind_onshore',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 10898,
                GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_onshore_cf_data['value'].values,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_wind_offshore',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.wind,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 8399,
                GEN_UNITS_PYPSA_PARAMS.max_power_pu: wind_offshore_cf_data['value'].values,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.wind].primary_cost
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_solar_pv',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.solar,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 47893,
                GEN_UNITS_PYPSA_PARAMS.max_power_pu: solar_pv_cf_data['value'].values,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: fuel_sources[FuelNames.solar].primary_cost
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_other-renewables',
                GEN_UNITS_PYPSA_PARAMS.carrier: FuelNames.other_renewables,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 1010,
            },
            # QUESTION: what is this - very necessary - last fictive asset?
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_failure',
                GEN_UNITS_PYPSA_PARAMS.carrier: DummyFuelNames.ac,
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 1e10,
                GEN_UNITS_PYPSA_PARAMS.marginal_cost: 1e5
            },
            # this is a battery 
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_battery', 
                GEN_UNITS_PYPSA_PARAMS.carrier: 'Flexibility',  
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 2591, 
                GEN_UNITS_PYPSA_PARAMS.min_power_pu: -1, 
                GEN_UNITS_PYPSA_PARAMS.max_power_pu: 1, 
                GEN_UNITS_PYPSA_PARAMS.max_hours: 3, 
                GEN_UNITS_PYPSA_PARAMS.soc_init: 1000, 
                GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.95, 
                GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.95 
            },
            {
                GEN_UNITS_PYPSA_PARAMS.name: f'{country_trigram}_hydro_closed_loop', 
                GEN_UNITS_PYPSA_PARAMS.carrier: 'Flexibility',  
                GEN_UNITS_PYPSA_PARAMS.nominal_power: 1250, 
                GEN_UNITS_PYPSA_PARAMS.min_power_pu: -1, 
                GEN_UNITS_PYPSA_PARAMS.max_power_pu: 1, 
                GEN_UNITS_PYPSA_PARAMS.max_hours: 4.6, 
                GEN_UNITS_PYPSA_PARAMS.soc_init: 1000, 
                GEN_UNITS_PYPSA_PARAMS.efficiency_store: 0.95, 
                GEN_UNITS_PYPSA_PARAMS.efficiency_dispatch: 0.95 
            }
         
        ]
    return generators


def set_gen_as_list_of_gen_units_data(generators: List[GENERATOR_DICT_TYPE]) -> List[GenerationUnitData]:
    # add type of units
    for elt_gen in generators:
        elt_gen['type'] = f'{elt_gen["carrier"]}_agg'
    # then cas as list of GenerationUnitData objects
    return [GenerationUnitData(**elt_gen_dict) for elt_gen_dict in generators]

