"""Extract row-block weight evidence in a supervised worker process."""
import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gputracker.supervision import _write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True, help='JSON job specification')
    parser.add_argument('--result', type=Path, required=True, help='Terminal job outcome JSON path')
    cli_args = parser.parse_args()
    job_spec = json.loads(cli_args.request.read_text())
    from worker import HeartbeatReporter
    import pandas as pd
    from run_experiment import extract_row_block_evidence
    run_options = SimpleNamespace(**job_spec['options'])
    model_options = job_spec['row']
    heartbeat = HeartbeatReporter(os.environ.get('WORKER_HEARTBEAT_FILE'), model_options['model_id'])

    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    heartbeat.start(stage='fingerprint')
    status = 'failed'
    outcome = dict(model_id=model_options['model_id'], status=status, stage='fingerprint')
    try:
        status_csv_path = cli_args.result.with_suffix('.csv')
        extract_row_block_evidence(
            pd.DataFrame([model_options]), Path(job_spec['output_dir']), run_options,
            logging.getLogger(__name__), status_path=status_csv_path,
        )
        measurement_outcome = pd.read_csv(status_csv_path).fillna('').iloc[0]
        t1_status = measurement_outcome['t1_status']
        status = 'success' if t1_status in ('published', 'measured') else 'failed'
        outcome.update(status=status, t1_status=t1_status, reason=measurement_outcome.get('t1_error', ''))
    except BaseException as exc:
        outcome.update(status='failed', reason=type(exc).__name__, message=str(exc))
    finally:
        _write_json_atomic(cli_args.result, outcome)
        heartbeat.stop(state=status, stage='fingerprint')
    return 0 if status == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
