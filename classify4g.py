import logging
import detect

from datetime import datetime
from .classify import *

UPLINK_INTERFERENCE_KPIS = [
    'pmRadioRecInterferencePwr_mean',   # Derived
    'pmRadioRecInterferencePwr_std',    # Derived
    'pmRadioRecInterferencePwr_median', # Derived
    'pmRadioRecInterferencePwr_p95',    # Derived
    'pmSinrPuschDistr_mean',            # Derived
    'pmSinrPuschDistr_std',             # Derived
    'pmSinrPuschDistr_median',          # Derived
    'pmSinrPuschDistr_p95',             # Derived
    'uplink_harq_failure_rate',         # Derived
    'pmErabRelAbnormalEnb'              # Raw
]

MASS_EVENT_KPIS = [
    'pmRrcConnEstabAtt',     # Raw
    'pmActiveUeDlMax',       # Raw
    'pmPrbUtilDl',           # Raw
    'rrc_success_rate',      # Derived
    'pmPdcchCceUtil_mean',   # Derived
    'pmPdcchCceUtil_median', # Derived
    'pmPdcchCceUtil_p95'     # Derived
]

logger = logging.getLogger(__name__)

def classify_anomaly4g(event: detect.Event) -> Alert | None:
    def cell_outage_detected():
        return any({kpis['pmCellDowntimeMan'], kpis['pmCellDowntimeAuto']})

    def receiver_alive():
        return any({
            kpis['pmRadioRecInterferencePwr_mean'],
            kpis['pmRadioRecInterferencePwr_std'],
            kpis['pmRadioRecInterferencePwr_median'],
            kpis['pmRadioRecInterferencePwr_p95']})

    def has_connection_attempt():
        return any({kpis['pmRaAttCbra'], kpis['pmRrcConnEstabAtt']})

    ecgi = event.xcgi
    kpis = event.kpi_snapshot

    # Check if hard failures are already captured by existing KPIs.
    if cell_outage_detected():
        logger.info(f"Cell {ecgi} in outage mode, detected by pmCellDowntime counters")
        return None
    if kpis['pmCellSleepTime'] != 0:
        logger.info(f"Cell {ecgi} in sleep mode, detected by pmCellSleepTime counter")
        return None

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Detect hard failures first
    anomaly_type = None
    if kpis['pmRrcConnEstabAtt'] == 0 and kpis['pmActiveUeDlSum'] == 0 and not receiver_alive():
        anomaly_type = anomaly.Type.CELL_OUTAGE
    elif kpis['pmRrcConnEstabSucc'] == 0 and has_connection_attempt() and receiver_alive():
        anomaly_type = anomaly.Type.SLEEPING_CELL_STUCK
    elif not has_connection_attempt() and receiver_alive():
        anomaly_type = anomaly.Type.SLEEPING_CELL_SILENT
    if anomaly_type is not None:
        return Alert(
            timestamp=timestamp,
            rat='4G',
            xcgi=ecgi,
            anomaly_type=anomaly_type,
            kpi_snapshot=event.kpi_snapshot)

    # Detect degradation failures
    uplink_interference_scores = calculate_significance_scores(UPLINK_INTERFERENCE_KPIS, event.kpi_metadata)
    uplink_interference_score = calculate_severity(uplink_interference_scores.values(), len(UPLINK_INTERFERENCE_KPIS))
    mass_event_scores = calculate_significance_scores(MASS_EVENT_KPIS, event.kpi_metadata)
    mass_event_score = calculate_severity(mass_event_scores.values(), len(MASS_EVENT_KPIS))
    logger.info(f'{uplink_interference_score=}, {mass_event_score=}')
    difference = abs(uplink_interference_score - mass_event_score)
    average = (uplink_interference_score + mass_event_score) / 2
    if average > 0 and (difference / average) > CLEAR_WINNER_THRESHOLD:
        if uplink_interference_score > mass_event_score:
            anomaly_type = anomaly.Type.UPLINK_INTERFERENCE
        else:
            anomaly_type = anomaly.Type.MASS_EVENT
    else:
        anomaly_type = anomaly.Type.UNCLASSIFIED

    kpi_snapshot = {kpi_name: kpis[kpi_name] for kpi_name in UPLINK_INTERFERENCE_KPIS + MASS_EVENT_KPIS if kpi_name in kpis}
    kpi_significance_scores = uplink_interference_scores | mass_event_scores

    return Alert(
        timestamp=timestamp,
        rat='4G',
        xcgi=ecgi,
        anomaly_type=anomaly_type,
        kpi_snapshot=kpi_snapshot,
        kpi_significance_scores=kpi_significance_scores,
        classification_scores={
            'uplink_interference': uplink_interference_score,
            'mass_event': mass_event_score})
