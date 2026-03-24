"""
CHAOS V1.0 -- Live ZeroMQ Inference Server

Listens on tcp://127.0.0.1:5555 for MT5 EA requests.
Supports two request types:
  - RAW_BARS: EA sends raw OHLCV, server computes features + inference
  - FEATURES: EA sends pre-computed 273-feature vector
  - HEARTBEAT: Connection health check

Also supports FILE_BRIDGE mode for environments where ZeroMQ DLL
cannot be loaded in MT5. In this mode, the server watches a shared
directory for request files.

Usage:
    python -u inference/live_zmq_server.py [--mode zmq|file_bridge]
    python -u inference/live_zmq_server.py --endpoint tcp://127.0.0.1:5555
"""
import os
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'
import sys
import json
import time
import logging
import argparse
import signal as sig
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('live_zmq_server')

PROJECT_ROOT = Path('G:/My Drive/chaos_v1.0')
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))


def load_production_handler():
    """Initialize the full inference pipeline with production config."""
    from replay.runners.run_replay import ModelLoader, RegimeSimulator, EnsembleEngine
    from risk.engine.portfolio_state import PortfolioState
    from risk.engine.exposure_controller import ExposureController
    from inference.live_inference_handler import LiveInferenceHandler, LiveFeatureEngine

    logger.info("Loading production config...")

    # Load configs
    with open(PROJECT_ROOT / 'replay' / 'config' / 'brain_quarantine.json') as f:
        quarantine_cfg = json.load(f)
    with open(PROJECT_ROOT / 'replay' / 'config' / 'portfolio_allocation.json') as f:
        portfolio_cfg = json.load(f)

    with open(PROJECT_ROOT / 'replay' / 'config' / 'defensive_mode.json') as f:
        defensive_cfg = json.load(f)
    with open(PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json') as f:
        instrument_specs = json.load(f)

    logger.info(f"Quarantine: {quarantine_cfg['quarantine_version']}")
    logger.info(f"Global quarantine: {quarantine_cfg['global_quarantine']}")

    # Load models (all pairs, production TFs)
    all_tfs = ['M5', 'M15', 'M30', 'H1']
    model_loader = ModelLoader(
        str(PROJECT_ROOT / 'models'),
        pairs=['AUDUSD', 'EURUSD', 'EURJPY', 'GBPJPY', 'GBPUSD',
               'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
        tfs=all_tfs,
    )

    # Apply quarantine at load time
    quarantined_count = 0
    for pair_tf, models in list(model_loader.models.items()):
        parts = pair_tf.split('_')
        pair, tf = parts[0], parts[1]
        block_key = pair_tf
        to_remove = []
        for brain_name in models:
            if brain_name in quarantine_cfg.get('global_quarantine', []):
                to_remove.append(brain_name)
            else:
                cond = quarantine_cfg.get('conditional_quarantine', {}).get(brain_name)
                if cond and cond.get('policy') == 'quarantined_globally_except':
                    if block_key not in cond.get('exceptions', []):
                        to_remove.append(brain_name)
        for b in to_remove:
            del models[b]
            quarantined_count += 1

    total_active = sum(len(v) for v in model_loader.models.values())
    logger.info(f"Models: {total_active} active, {quarantined_count} quarantined")

    # Initialize components
    regime_sim = RegimeSimulator(str(PROJECT_ROOT / 'regime' / 'regime_policy.json'))
    ensemble_engine = EnsembleEngine(str(PROJECT_ROOT / 'ensemble' / 'ensemble_config.json'))
    risk_controller = ExposureController(str(PROJECT_ROOT / 'risk' / 'config' / 'risk_policy.json'))
    portfolio_state = PortfolioState(
        str(PROJECT_ROOT / 'risk' / 'config' / 'instrument_specs.json'),
        str(PROJECT_ROOT / 'risk' / 'config' / 'correlation_groups.json'),
    )
    feature_engine = LiveFeatureEngine()

    handler = LiveInferenceHandler(
        model_loader=model_loader,
        regime_sim=regime_sim,
        ensemble_engine=ensemble_engine,
        risk_controller=risk_controller,
        portfolio_state=portfolio_state,
        quarantine_config=quarantine_cfg,
        portfolio_config=portfolio_cfg,
        feature_engine=feature_engine,
        defensive_config=defensive_cfg,
        instrument_specs=instrument_specs,
    )

    logger.info("Production handler ready")
    return handler


def run_zmq_server(endpoint="tcp://127.0.0.1:5555"):
    """Run ZeroMQ REP server."""
    import zmq

    handler = load_production_handler()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(endpoint)

    logger.info(f"CHAOS V1.0 Live Inference Server started on {endpoint}")
    logger.info("Waiting for MT5 EA connections...")

    running = True
    def shutdown(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received")
        running = False
    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    request_count = 0
    error_count = 0

    while running:
        try:
            if socket.poll(1000):  # 1 second timeout
                message = socket.recv_string()
                request = json.loads(message)

                # Heartbeat
                if request.get('type') == 'HEARTBEAT':
                    socket.send_string(json.dumps({
                        'status': 'ALIVE',
                        'uptime_requests': request_count,
                        'errors': error_count,
                        'timestamp': time.time(),
                    }))
                    continue

                # Inference request
                start = time.perf_counter()
                response = handler.process_request(request)
                latency = (time.perf_counter() - start) * 1000
                response['server_latency_ms'] = round(latency, 2)

                socket.send_string(json.dumps(response))
                request_count += 1

                pair = request.get('pair', request.get('symbol', '?'))
                tf = request.get('tf', request.get('timeframe', '?'))
                action = response.get('action', '?')
                signal = response.get('signal', 0)

                if request_count % 50 == 0:
                    logger.info(f"Requests: {request_count}, "
                               f"last: {pair}_{tf} -> {action} (signal={signal}, {latency:.1f}ms)")

        except json.JSONDecodeError as e:
            error_count += 1
            try:
                socket.send_string(json.dumps({'error': f'Invalid JSON: {str(e)}'}))
            except Exception:
                pass
        except Exception as e:
            error_count += 1
            logger.error(f"Server error: {e}", exc_info=True)
            try:
                socket.send_string(json.dumps({'error': str(e)}))
            except Exception:
                pass

    socket.close()
    context.term()
    logger.info(f"Server shutdown. Processed {request_count} requests, {error_count} errors.")


def run_file_bridge(bridge_dir=None):
    """
    File-based bridge for environments where ZeroMQ DLL is unavailable in MT5.
    EA writes request.json, server reads it, processes, writes response.json.
    """
    bridge_dir = Path(bridge_dir or PROJECT_ROOT / 'CHAOS_Bridge')
    bridge_dir.mkdir(parents=True, exist_ok=True)

    request_path = bridge_dir / 'request.json'
    response_path = bridge_dir / 'response.json'
    ack_path = bridge_dir / 'request.ack'

    handler = load_production_handler()

    logger.info(f"CHAOS V1.0 File Bridge Server started")
    logger.info(f"Bridge directory: {bridge_dir}")
    logger.info("Watching for request.json...")

    running = True
    def shutdown(signum, frame):
        nonlocal running
        running = False
    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    request_count = 0

    while running:
        try:
            if request_path.exists() and not ack_path.exists():
                # Read request
                try:
                    with open(request_path) as f:
                        request = json.load(f)
                except (json.JSONDecodeError, OSError):
                    time.sleep(0.05)  # File might still be writing
                    continue

                # Acknowledge (prevents re-read)
                ack_path.touch()

                # Heartbeat
                if request.get('type') == 'HEARTBEAT':
                    response = {
                        'status': 'ALIVE',
                        'uptime_requests': request_count,
                        'timestamp': time.time(),
                    }
                else:
                    # Process inference
                    start = time.perf_counter()
                    response = handler.process_request(request)
                    latency = (time.perf_counter() - start) * 1000
                    response['server_latency_ms'] = round(latency, 2)
                    request_count += 1

                # Write response atomically
                tmp_path = bridge_dir / 'response.tmp'
                with open(tmp_path, 'w') as f:
                    json.dump(response, f)
                tmp_path.replace(response_path)

                # Clean up request + ack
                try:
                    request_path.unlink()
                    ack_path.unlink()
                except OSError:
                    pass

                if request_count % 50 == 0:
                    logger.info(f"File bridge: {request_count} requests processed")

            else:
                time.sleep(0.1)  # Poll interval

        except Exception as e:
            logger.error(f"File bridge error: {e}", exc_info=True)
            time.sleep(1)

    logger.info(f"File bridge shutdown. Processed {request_count} requests.")


def main():
    parser = argparse.ArgumentParser(description='CHAOS V1.0 Live Inference Server')
    parser.add_argument('--mode', choices=['zmq', 'file_bridge'], default='zmq',
                       help='Communication mode (default: zmq)')
    parser.add_argument('--endpoint', default='tcp://127.0.0.1:5555',
                       help='ZeroMQ endpoint (default: tcp://127.0.0.1:5555)')
    parser.add_argument('--bridge-dir', default=None,
                       help='File bridge directory (default: CHAOS_Bridge/)')
    args = parser.parse_args()

    if args.mode == 'zmq':
        run_zmq_server(args.endpoint)
    else:
        run_file_bridge(args.bridge_dir)


if __name__ == '__main__':
    main()
