from pathlib import Path

import numpy as np

from generate_london_osm_cycle import (
    apply_urban_events,
    compute_grade,
    compute_time_from_geometry,
    generate_speed_profile,
    get_elevation_profile,
    save_cycle_yaml,
)
from route_optimization import (
    load_optimized_route,
    optimize_route,
    save_optimized_route,
)
from utils import get_data_dir

# ==============================================================
# Config
# ==============================================================

DEFAULT_LOCATION = "London"


# ==============================================================
# Helpers
# ==============================================================


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad = np.radians([lat1, lon1])
    lat2_rad, lon2_rad = np.radians([lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (np.sin(dlat / 2)) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * (np.sin(dlon / 2)) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000 * c


def _segment_lengths(lat: list[float], lon: list[float]) -> list[float]:
    return [
        float(_haversine_distance_m(lat[i - 1], lon[i - 1], lat[i], lon[i]))
        for i in range(1, len(lat))
    ]


def _route_to_coords(route_data: dict) -> tuple[list[float], list[float]]:
    coords = route_data.get("coords", [])
    if not coords:
        raise ValueError("Baseline route is missing coordinate data.")
    lat = [point["lat"] for point in coords]
    lon = [point["lon"] for point in coords]
    if len(lat) < 2:
        raise ValueError("Baseline route must include at least two coordinates.")
    return lat, lon


def _update_route_summary(route_data: dict, distance_m: float, duration_s: float) -> bool:
    summary = route_data.setdefault("summary", {})
    updated = False
    if summary.get("distance_m") != distance_m:
        summary["distance_m"] = distance_m
        updated = True
    if summary.get("duration_s") is None or summary.get("duration_s") <= 0:
        summary["duration_s"] = duration_s
        updated = True
    return updated


def _needs_route_refresh(route_data: dict, expected_weights: dict[str, float]) -> bool:
    summary = route_data.get("summary", {})
    weights = summary.get("weights")
    duration_s = summary.get("duration_s")
    energy_est_wh = summary.get("energy_est_Wh")
    if weights != expected_weights:
        return True
    if duration_s is None or duration_s <= 0:
        return True
    if energy_est_wh is None or energy_est_wh <= 0:
        return True
    return False


# ==============================================================
# Main entry
# ==============================================================


def generate_london_osm_cycle_baseline(
    location: str = DEFAULT_LOCATION,
    route_filename: str | None = None,
    output_filename: str | None = None,
) -> Path:
    data_dir = get_data_dir()
    if route_filename is None:
        route_filename = f"{location}_baseline_route.json"
    if output_filename is None:
        output_filename = f"{location}_cycle_baseline.yaml"
    route_path = data_dir / route_filename
    expected_weights = {"energy": 0.0, "time": 1.0}
    if route_path.exists():
        route_data = load_optimized_route(route_filename)
        if _needs_route_refresh(route_data, expected_weights):
            route_data = optimize_route(
                origin=(51.5246, -0.1340),
                destination=(51.5033, -0.1133),
                weights=expected_weights,
            )
            save_optimized_route(route_data, route_filename)
    else:
        route_data = optimize_route(
            origin=(51.5246, -0.1340),
            destination=(51.5033, -0.1133),
            weights=expected_weights,
        )
        save_optimized_route(route_data, route_filename)

    lat, lon = _route_to_coords(route_data)

    elevation = get_elevation_profile(lat, lon)
    grade = compute_grade(elevation, lat, lon)
    speed = generate_speed_profile(lat, lon)
    speed = apply_urban_events(speed)
    t = compute_time_from_geometry(lat, lon, speed)

    segment_lengths = _segment_lengths(lat, lon)
    total_distance_m = float(sum(segment_lengths))
    duration_s = float(t[-1]) if len(t) else 0.0

    if _update_route_summary(route_data, total_distance_m, duration_s):
        save_optimized_route(route_data, route_filename)

    road_names = route_data.get("road_names") or ["Baseline Route"]
    route_json = {
        "features": [
            {
                "properties": {
                    "summary": {
                        "distance": total_distance_m,
                        "duration": duration_s,
                    }
                }
            }
        ]
    }

    output_path = data_dir / output_filename

    save_cycle_yaml(
        [
            {
                "filename": str(output_path),
                "name": "london_osm_urban_baseline",
                "description": "Baseline OSM route weighted entirely on time.",
                "t": t,
                "speed": speed,
                "grade": grade,
                "road_names": road_names,
                "segment_lengths": [float(length) for length in segment_lengths],
                "lat": lat,
                "lon": lon,
                "route_json": route_json,
                "routing_profile": route_data.get("summary", {}).get("network_type", "bike"),
            }
        ]
    )

    return output_path


# ==============================================================
# How to run
# ==============================================================
# python src/generate_london_osm_cycle_baseline.py


if __name__ == "__main__":
    output = generate_london_osm_cycle_baseline()
    print(f"Saved baseline cycle → {output}")
