from dataclasses import dataclass, asdict
from typing import Dict
import json


def dict_to_str(d: Dict[str, float], unit: str = None) -> str:
    str_sep = '\n* '
    d_str = ''
    unit_suffix = f' ({unit})' if unit is not None else ''
    for key, val in d.items():
        d_str += f'{str_sep}{key}: {val}{unit_suffix}'
    return d_str


@dataclass
class UCSummaryMetrics:
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
        uc_summary_metrics_str += f'{metric_sep}Energy Not Served: {dict_to_str(d=self.per_country_ens, unit='GWh')}'
        uc_summary_metrics_str += f'{metric_sep}Number of failure hours: {dict_to_str(d=self.per_country_n_failure_hours)}'
        uc_summary_metrics_str += f'{metric_sep}Total cost (over Europe): {self.total_cost} (€)'
        uc_summary_metrics_str += f'{metric_sep}Total operational cost (Europe, without failure penalty incl.): {self.total_operational_cost} (€)'
        uc_summary_metrics_str += f'{metric_sep}Total CO2 emissions (over Europe): {self.total_co2_emissions} (??)'
        if self.per_country_total_cost is not None:
            uc_summary_metrics_str += f'{metric_sep}Total cost: {dict_to_str(d=self.per_country_total_cost, unit='€')}'
        if self.per_country_total_operational_cost is not None:
            uc_summary_metrics_str += f'{metric_sep}Total operational cost (without failure penalty incl.): {dict_to_str(d=self.per_country_total_operational_cost, unit='€')}'
        if self.per_country_co2_emissions is not None:
            uc_summary_metrics_str += f'{metric_sep}Total CO2 emissions: {dict_to_str(d=self.per_country_co2_emissions, unit='??')}'
        return uc_summary_metrics_str
    
    def json_dump(self, file: str):
        summary_dict = asdict(self)
        # remove None values
        summary_dict = {key: val for key, val in summary_dict.items() if val is not None}
        with open(file, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f)
