import os
import anomaly

from dataclasses import dataclass
from typing import Dict, Any, Iterable

CLEAR_WINNER_THRESHOLD = float(os.environ.get('CLEAR_WINNER_THRESHOLD', 0.2))  # 20%

@dataclass
class Alert:
    timestamp               : str
    rat                     : str
    xcgi                    : str
    anomaly_type            : anomaly.Type
    kpi_snapshot            : Dict[str, Any]
    kpi_significance_scores : Dict[str, Any] = None
    classification_scores   : Dict[str, Any] = None

    @property
    def id(self) -> str:
        return f'ANOMALY-ALERT-{self.rat}-{self.xcgi}-{self.timestamp}'

def calculate_significance_scores(kpi_names: Iterable[str], kpi_metadata: Dict[str, Any]) -> Dict[str, float]:
    return {
        kpi_name: kpi_metadata[kpi_name]['reconstruction_error'] / kpi_metadata[kpi_name]['reconstruction_threshold']
        for kpi_name in kpi_names if kpi_name in kpi_metadata}

def calculate_severity(significance_scores: Iterable[float], kpi_count: int) -> float:
    return sum(significance_scores) / kpi_count if significance_scores else 0.0
