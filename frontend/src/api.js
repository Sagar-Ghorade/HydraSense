/**
 * Frontend API client for HydraSense.
 *
 * All endpoints are relative to the FastAPI backend URL.
 * In development the Vite proxy forwards /api/* to the backend;
 * in production the backend URL is set via env.FREEBUFF_API_ORIGIN.
 */

import { fetchUtils } from "react-admin";

const API_BASE = process.env.REACT_APP_API_BASE || "/api";

/** Health check */
export const checkHealth = () => fetch(`${API_BASE}/health`).then(r => r.json());

/** List stations */
export const listStations = () => fetch(`${API_BASE}/stations`).then(r => r.json());

/** Historical readings (last 30 days) */
export const getHistoricalReadings = () =>
  fetch(`${API_BASE}/groundwater`).then(r => r.json());

/** Format a level value from the API for display */
export const formatLevel = (level) => `${Number(level).toFixed(3)} m`;

/** Make a prediction request */
export const makePrediction = (stationId, horizonDays) =>
  fetch(`${API_BASE}/predictions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ station_id: stationId, horizon_days: horizonDays }),
  }).then((r) => {
    if (!r.ok) return r.json().then((e) => Promise.reject(e));
    return r.json();
  });