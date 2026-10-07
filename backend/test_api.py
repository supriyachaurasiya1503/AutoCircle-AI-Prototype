import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[1]


def run_tests():
    print("==========================================")
    print("RUNNING END-TO-END FASTAPI API TEST SUITE")
    print("==========================================")

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    print(f"[PASS] Health Check: {res.json()}")

    # 2. Battery Predict Single
    payload = {
        "cycle_count": 800,
        "avg_temp": 28.0,
        "depth_of_discharge": 70.0,
        "fast_charge_frequency": 20.0,
        "internal_resistance": 110.0,
    }
    res = client.post("/api/battery/predict", json=payload)
    assert res.status_code == 200, f"Battery predict failed: {res.text}"
    print(f"[PASS] Battery SOH Predict: SOH={res.json()['soh']}% | Routing={res.json()['routing_decision']}")

    # 3. Disassembly Detect
    res = client.post("/api/disassembly/detect")
    assert res.status_code == 200, f"Disassembly failed: {res.text}"
    print(f"[PASS] Disassembly Detect: Components={len(res.json()['components_detected'])} | Time={res.json()['estimated_time_mins']} mins")

    # 4. Passport Generate
    p_payload = {"component_type": "battery_pack", "vin": "1HGBH41JXMN109186", "soh": 73.4}
    res = client.post("/api/passport/generate", json=p_payload)
    assert res.status_code == 200, f"Passport failed: {res.text}"
    print(f"[PASS] Digital Passport Generate: ID={res.json()['passport_id']} | Compliance={res.json()['compliance_badges']}")

    # 5. Sustainability Analyze
    s_payload = {"vehicles_processed": 2000, "second_life_rate": 0.6, "aluminium_recovery_purity": 0.94, "renewable_energy_ratio": 0.35}
    res = client.post("/api/sustainability/analyze", json=s_payload)
    assert res.status_code == 200, f"Sustainability failed: {res.text}"
    print(f"[PASS] Sustainability Analyze: Monthly CO2e={res.json()['monthly_co2e_avoided_tonnes']} tonnes | Tokens={res.json()['acct_tokens_earned']}")

    print("==========================================")
    print("ALL API ENDPOINTS PASSED VERIFICATION!")
    print("==========================================\n")


def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'


def test_battery_predict():
    payload = {
        'cycle_count': 800,
        'avg_temp': 28.0,
        'depth_of_discharge': 70.0,
        'fast_charge_frequency': 20.0,
        'internal_resistance': 110.0,
    }
    response = client.post('/api/battery/predict', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 'soh' in body
    assert 'routing_decision' in body


def test_battery_forecast_uses_saved_metrics_and_model():
    dataset = pd.read_csv(REPO_ROOT / 'data' / 'battery_cycle_level_dataset_CLEAN_FINAL.csv')
    history = dataset[dataset['battery_id'] == 'B0005'].sort_values('cycle').head(8)[
        ['cycle', 'voltage', 'temperature', 'capacity']
    ].to_dict(orient='records')

    metrics_response = client.get('/api/battery/metrics')
    forecast_response = client.post(
        '/api/battery/forecast',
        json={'battery_id': 'B0005', 'history': history},
    )

    assert metrics_response.status_code == 200
    assert forecast_response.status_code == 200
    metrics_payload = metrics_response.json()
    metrics = metrics_payload['known_battery_future_cycles']['exact_one_physical_cycle_test']
    forecast = forecast_response.json()
    assert forecast['next_cycle'] == 9
    assert forecast['test_rows'] == metrics['rows']
    assert forecast['test_coverage'] == metrics_response.json()['known_battery_future_cycles']['exact_one_physical_cycle_coverage']['fraction']
    assert forecast['mae_percentage_points'] == metrics['mae_percentage_points']
    selected_model = metrics_payload['known_battery_future_cycles']['selected_model']
    assert forecast['model_used'].lower().startswith(selected_model.lower())


def test_battery_forecast_rejects_missing_history_fields():
    response = client.post(
        '/api/battery/forecast',
        json={'battery_id': 'B0005', 'history': [{'cycle': 1, 'capacity': 1.86}]},
    )
    assert response.status_code == 422


def test_battery_forecast_rejects_unevaluated_multi_cycle_horizon():
    dataset = pd.read_csv(REPO_ROOT / 'data' / 'battery_cycle_level_dataset_CLEAN_FINAL.csv')
    history = dataset[dataset['battery_id'] == 'B0005'].sort_values('cycle').head(8)[
        ['cycle', 'voltage', 'temperature', 'capacity']
    ].to_dict(orient='records')
    response = client.post(
        '/api/battery/forecast',
        json={'battery_id': 'B0005', 'history': history, 'next_cycle': 11},
    )
    assert response.status_code == 422
    assert 'exactly one physical cycle ahead' in response.json()['detail']


def test_battery_samples_return_real_dataset_history():
    batteries_response = client.get('/api/battery/batteries')
    history_response = client.get('/api/battery/batteries/B0005/history?through_cycle=8')

    assert batteries_response.status_code == 200
    batteries = batteries_response.json()['batteries']
    battery_option = next(item for item in batteries if item['battery_id'] == 'B0005')
    assert battery_option['observed_cycles'][0] == 1
    assert battery_option['observed_cycles'][-1] == 168
    assert all(item['cycle_count'] >= 4 for item in batteries)
    assert history_response.status_code == 200
    history = history_response.json()['history']
    assert len(history) == 8
    assert history[0]['cycle'] == 1
    assert set(history[0]) == {'cycle', 'voltage', 'temperature', 'capacity'}


def test_battery_forecast_covers_all_routing_thresholds():
    dataset = pd.read_csv(REPO_ROOT / 'data' / 'battery_cycle_level_dataset_CLEAN_FINAL.csv')
    scenarios = [
        ('B0005', 8, 'Continue EV Use'),
        ('B0005', 168, 'Second-Life Storage'),
        ('B0006', 155, 'Recycle Now'),
    ]

    for battery_id, through_cycle, expected_route in scenarios:
        history = dataset[
            (dataset['battery_id'] == battery_id) & (dataset['cycle'] <= through_cycle)
        ].sort_values('cycle')[['cycle', 'voltage', 'temperature', 'capacity']].to_dict(orient='records')
        response = client.post(
            '/api/battery/forecast',
            json={'battery_id': battery_id, 'history': history},
        )
        assert response.status_code == 200, response.text
        assert response.json()['routing_decision'] == expected_route


def test_disassembly_detect():
    response = client.post('/api/disassembly/detect')
    assert response.status_code == 200
    body = response.json()
    assert 'components_detected' in body
    assert 'disassembly_sequence' in body


def test_passport_generate():
    payload = {
        'component_type': 'battery_pack',
        'vin': '1HGBH41JXMN109186',
        'soh': 73.4,
    }
    response = client.post('/api/passport/generate', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 'passport_id' in body
    assert 'compliance_badges' in body


def test_sustainability_analyze():
    payload = {
        'vehicles_processed': 2000,
        'second_life_rate': 0.6,
        'aluminium_recovery_purity': 0.94,
        'renewable_energy_ratio': 0.35,
    }
    response = client.post('/api/sustainability/analyze', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 'monthly_co2e_avoided_tonnes' in body
    assert body['monthly_co2e_avoided_tonnes'] >= 0


if __name__ == "__main__":
    run_tests()
