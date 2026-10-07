const API_BASE_URL = 'http://127.0.0.1:8000/api';

export async function predictBatterySOH(payload) {
  const res = await fetch(`${API_BASE_URL}/battery/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Battery prediction failed with status ${res.status}`);
  }
  return await res.json();
}

export async function predictBatteryForecast(payload) {
  const res = await fetch(`${API_BASE_URL}/battery/forecast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Battery forecast failed with status ${res.status}`);
  }
  return await res.json();
}

export async function fetchBatteryMetrics() {
  const res = await fetch(`${API_BASE_URL}/battery/metrics`);
  if (!res.ok) {
    throw new Error(`Battery metrics unavailable with status ${res.status}`);
  }
  return await res.json();
}

export async function fetchBatterySamples() {
  const res = await fetch(`${API_BASE_URL}/battery/batteries`);
  if (!res.ok) {
    throw new Error(`Battery samples unavailable with status ${res.status}`);
  }
  return await res.json();
}

export async function fetchBatteryHistory(batteryId, throughCycle) {
  const query = new URLSearchParams({ through_cycle: String(throughCycle) });
  const res = await fetch(`${API_BASE_URL}/battery/batteries/${encodeURIComponent(batteryId)}/history?${query}`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `Battery history unavailable with status ${res.status}`);
  }
  return await res.json();
}

export async function fetchDisassemblyPlan() {
  const res = await fetch(`${API_BASE_URL}/disassembly/detect`, { method: 'POST' });
  if (!res.ok) {
    throw new Error(`Disassembly detection failed with status ${res.status}`);
  }
  return await res.json();
}

export async function generateDigitalPassport(payload) {
  const res = await fetch(`${API_BASE_URL}/passport/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Passport generation failed with status ${res.status}`);
  }
  return await res.json();
}

export async function analyzeCarbonImpact(payload) {
  const res = await fetch(`${API_BASE_URL}/sustainability/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Sustainability analysis failed with status ${res.status}`);
  }
  return await res.json();
}
