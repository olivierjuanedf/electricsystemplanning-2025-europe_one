from dataclasses import dataclass, asdict
from typing import Dict
import json


def dict_to_str(d: Dict[str, float]) -> str:
    str_sep = ', '
    key_val_lst = [f'{key}: {val}' for key, val in d.items()]
    return str_sep.join(key_val_lst)


@dataclass
class UCSummaryMetrics:
    ENERGY_UNIT = 'GWh'
    COST_UNIT = 'M€'
    CO2_EMIS_UNIT = '??'
    # Energy Not Served (ENS), as the sum of failure volumes over horizon simulated 
    # (attention, in GWh)
    per_country_ens: Dict[str, float]
    # Number of hours with nonzero failure volume (may be a criterion integrated in national energy regulation 
    # for capa. sizing, e.g. in France with maximally 3hours in average over a set of climatic scenarios) 
    per_country_n_failure_hours: Dict[str, float]
    total_cost: float  # over Europe
    total_operational_cost: float  # without failure (fictive) penalty cost
    total_co2_emissions: float  # over Europe
    per_country_total_cost: Dict[str, float] = None  # including failure penalty (fictive) cost
    per_country_total_operational_cost: Dict[str, float] = None  # without this cost
    per_country_co2_emissions: Dict[str, float] = None 

    def __repr__(self, europe_name: str = None) -> str:
        metric_sep = '\n-> '
        uc_summary_metrics_str = 'UCSummaryMetrics'
        if europe_name is not None:
            uc_summary_metrics_str += f'for {europe_name}'
        uc_summary_metrics_str += f'{metric_sep}Energy Not Served ({self.ENERGY_UNIT}): {dict_to_str(d=self.per_country_ens)}'
        uc_summary_metrics_str += f'{metric_sep}Number of failure hours: {dict_to_str(d=self.per_country_n_failure_hours)}'
        uc_summary_metrics_str += f'{metric_sep}Total cost (over Europe, {self.COST_UNIT}): {self.total_cost}'
        uc_summary_metrics_str += f'{metric_sep}Total operational cost (Europe, without failure penalty incl., {self.COST_UNIT}): {self.total_operational_cost}'
        uc_summary_metrics_str += f'{metric_sep}Total CO2 emissions (over Europe, {self.CO2_EMIS_UNIT}): {self.total_co2_emissions}'
        if self.per_country_total_cost is not None:
            uc_summary_metrics_str += f'{metric_sep}Total cost ({self.COST_UNIT}): {dict_to_str(d=self.per_country_total_cost)}'
        if self.per_country_total_operational_cost is not None:
            uc_summary_metrics_str += f'{metric_sep}Total operational cost (without failure penalty incl., {self.COST_UNIT}): {dict_to_str(d=self.per_country_total_operational_cost)}'
        if self.per_country_co2_emissions is not None:
            uc_summary_metrics_str += f'{metric_sep}Total CO2 emissions ({self.CO2_EMIS_UNIT}): {dict_to_str(d=self.per_country_co2_emissions)}'
        return uc_summary_metrics_str
    
    def json_dump(self, file: str):
        summary_dict = asdict(self)
        # remove None values
        summary_dict = {key: val for key, val in summary_dict.items() if val is not None}
        with open(file, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f)
