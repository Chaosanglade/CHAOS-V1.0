"""
Validates sample request/response against inference contract JSON Schemas.

# MT5 NEVER decides model logic. MT5 only:
# 1. Assembles features from price data
# 2. Sends ZeroMQ request
# 3. Receives decision
# 4. Executes with CoreArb DLL + risk constraints

Usage: python test_contracts.py
Requires: pip install jsonschema
"""
import json
import os
from jsonschema import validate, ValidationError, Draft7Validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_schema(filename):
    path = os.path.join(BASE_DIR, filename)
    with open(path) as f:
        return json.load(f)


def test_request_schema():
    """Validate a sample request against the inference_request schema."""
    schema = load_schema('inference_request.json')

    # Verify schema itself is valid
    Draft7Validator.check_schema(schema)

    # Sample valid request
    sample_request = {
        "request_id": 1,
        "timestamp": "2026-03-02T12:00:00Z",
        "symbol": "EURUSD",
        "timeframe": "M30",
        "regime_state": 1,
        "regime_confidence": 0.85,
        "features": [0.0] * 273  # 273 universal features
    }

    try:
        validate(instance=sample_request, schema=schema)
        print("  Request schema: PASS (valid request accepted)")
    except ValidationError as e:
        print(f"  Request schema: FAIL - {e.message}")
        return False

    # Test invalid request (wrong feature count)
    bad_request = sample_request.copy()
    bad_request["features"] = [0.0] * 100
    try:
        validate(instance=bad_request, schema=schema)
        print("  Request schema: FAIL (should have rejected wrong feature count)")
        return False
    except ValidationError:
        print("  Request schema: PASS (correctly rejected wrong feature count)")

    # Test invalid request (bad symbol)
    bad_request2 = sample_request.copy()
    bad_request2["symbol"] = "INVALID"
    try:
        validate(instance=bad_request2, schema=schema)
        print("  Request schema: FAIL (should have rejected invalid symbol)")
        return False
    except ValidationError:
        print("  Request schema: PASS (correctly rejected invalid symbol)")

    # Test invalid request (missing required field)
    bad_request3 = {"request_id": 1, "timestamp": "2026-03-02T12:00:00Z"}
    try:
        validate(instance=bad_request3, schema=schema)
        print("  Request schema: FAIL (should have rejected missing fields)")
        return False
    except ValidationError:
        print("  Request schema: PASS (correctly rejected missing fields)")

    return True


def test_response_schema():
    """Validate a sample response against the inference_response schema."""
    schema = load_schema('inference_response.json')

    # Verify schema itself is valid
    Draft7Validator.check_schema(schema)

    # Sample valid response
    sample_response = {
        "request_id": 1,
        "ensemble_key": "EURUSD_M30_ensemble",
        "signal": "LONG",
        "class_probs": [0.15, 0.25, 0.60],
        "confidence": 0.60,
        "models_voted": 18,
        "models_agreed": 14,
        "agreement_ratio": 0.778,
        "latency_ms": 12.5,
        "tf_confirmation": {
            "h1_bias": "LONG",
            "m30_signal": "LONG",
            "m15_confirm": "LONG",
            "aligned": True
        }
    }

    try:
        validate(instance=sample_response, schema=schema)
        print("  Response schema: PASS (valid response accepted)")
    except ValidationError as e:
        print(f"  Response schema: FAIL - {e.message}")
        return False

    # Test invalid response (bad signal)
    bad_response = sample_response.copy()
    bad_response["signal"] = "BUY"
    try:
        validate(instance=bad_response, schema=schema)
        print("  Response schema: FAIL (should have rejected invalid signal)")
        return False
    except ValidationError:
        print("  Response schema: PASS (correctly rejected invalid signal)")

    # Test FLAT response (error/regime denial case)
    flat_response = {
        "request_id": 2,
        "ensemble_key": "EURUSD_M30_ensemble",
        "signal": "FLAT",
        "class_probs": [0.0, 1.0, 0.0],
        "confidence": 1.0,
        "models_voted": 0,
        "models_agreed": 0,
        "agreement_ratio": 0.0,
        "latency_ms": 0.5
    }

    try:
        validate(instance=flat_response, schema=schema)
        print("  Response schema: PASS (FLAT response accepted)")
    except ValidationError as e:
        print(f"  Response schema: FAIL - {e.message}")
        return False

    return True


if __name__ == '__main__':
    print("Contract Validation Tests")
    print("=" * 50)

    req_ok = test_request_schema()
    resp_ok = test_response_schema()

    print()
    if req_ok and resp_ok:
        print("ALL CONTRACT TESTS PASSED")
    else:
        print("FAILURES DETECTED")
