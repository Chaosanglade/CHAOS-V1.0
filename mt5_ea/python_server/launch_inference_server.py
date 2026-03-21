"""
CHAOS V1.0 -- Python Inference Server Launcher

Quick-start script for the live inference server.
Run this BEFORE starting the MT5 EA.

Usage:
    python launch_inference_server.py              # ZeroMQ mode (default)
    python launch_inference_server.py --file-bridge # File bridge mode (no ZMQ DLL needed)
"""
import os
import sys
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference'))

# Suppress warnings
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-bridge', action='store_true',
                       help='Use file bridge instead of ZeroMQ')
    parser.add_argument('--endpoint', default='tcp://127.0.0.1:5555')
    args = parser.parse_args()

    from inference.live_zmq_server import run_zmq_server, run_file_bridge

    if args.file_bridge:
        print("Starting CHAOS V1.0 in FILE BRIDGE mode...")
        run_file_bridge()
    else:
        print("Starting CHAOS V1.0 in ZeroMQ mode...")
        print(f"Endpoint: {args.endpoint}")
        print("Start MT5 EA after 'Waiting for connections...' appears.")
        run_zmq_server(args.endpoint)
