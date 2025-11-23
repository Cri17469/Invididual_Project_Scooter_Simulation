import os
from pathlib import Path

import numpy as np
import yaml
import requests
import time
import openrouteservice as ors


# ==============================================================
# 1. Get real OSM route using OpenRouteService
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _read_api_key_from_env_file(var_name: str, env_filename: str = ".env"):
    """Return the first matching key from a simple KEY=VALUE .env style file."""
    env_path = PROJECT_ROOT / env_filename
    if not env_path.exists():
        return None

    try:
        with open(env_path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() == var_name:
                    return value.strip().strip('"').strip("'")
    except OSError:
        # Fallback to default behavior – this helper is best-effort only.
        return None

    return None


def _resolve_api_key_from_candidates(candidates: list[str], env_filename: str = ".env") -> str | None:
    """Try multiple env variable names before giving up."""
    for name in candidates:
        value = os.getenv(name)
        if value:
            return value

    for name in candidates:
        value = _read_api_key_from_env_file(name, env_filename)
        if value:
            return value

    return None


def resolve_ors_api_key(explicit_key: str | None = None) -> str:
    """Resolve the OpenRouteService API key from the caller/env/.env file."""
    if explicit_key:
        return explicit_key

    api_key = _resolve_api_key_from_candidates(["ORS_API_KEY"])

    if not api_key:
        raise EnvironmentError(
            "Missing OpenRouteService API key. Set ORS_API_KEY in the environment or .env file."
        )

    return api_key


def resolve_tomtom_api_key(explicit_key: str | None = None) -> str:
    """Resolve the TomTom Traffic API key from env or `.env`."""
    if explicit_key:
        return explicit_key

    candidates = ["TOM_API_KEY", "TOMTOM_API_KEY", "tom_api_key"]
    api_key = _resolve_api_key_from_candidates(candidates)

    if not api_key:
        names = ", ".join(candidates)
        raise EnvironmentError(
            f"Missing TomTom API key. Set one of {{{names}}} in the environment or .env file."
        )

    return api_key


def fetch_osm_route(start, end, api_key, profile="driving-car"):
    """
    Fetch real-world route between two coordinates (lat, lon).
    Returns:
        lat_list, lon_list, full_route_json
    """
    client = ors.Client(key=api_key)

    route = client.directions(
        coordinates=[(start[1], start[0]), (end[1], end[0])],
        profile=profile,
        format="geojson",
        instructions="true"
    )

    coords = route['features'][0]['geometry']['coordinates']
    lat = [c[1] for c in coords]
    lon = [c[0] for c in coords]

    return lat, lon, route


# ==============================================================
# 2. Extract road names from ORS route JSON
# ==============================================================

def extract_road_names(route_json):
    names = []

    segments = route_json['features'][0]['properties']['segments']
    for seg in segments:
        for step in seg["steps"]:
            nm = step.get("name", "").strip()
            if nm and nm not in names:
                names.append(nm)

    return names


def extract_segment_lengths(route_json):
    """
    Returns list of segment lengths in meters
    """
    lengths = []
    segments = route_json['features'][0]['properties']['segments']

    for seg in segments:
        for step in seg['steps']:
            lengths.append(step.get("distance", 0.0))

    return lengths


def _fetch_tomtom_flow_ratio(lat: float, lon: float, api_key: str):
    """Return congestion coefficient plus raw TomTom speeds.

    The Flow Segment Data API provides a current speed and a free-flow speed.
    We treat the ratio of the two as a multiplier for our synthetic speed
    profile.
    """

    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "point": f"{lat},{lon}",
        "unit": "KMPH",
        "key": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError, KeyError):
        return None, None, None

    flow = data.get("flowSegmentData", {})
    current_speed = flow.get("currentSpeed")
    free_speed = flow.get("freeFlowSpeed")

    if not current_speed or not free_speed:
        return None, current_speed, free_speed
    if free_speed <= 0:
        return None, current_speed, free_speed

    ratio = current_speed / free_speed
    return ratio, current_speed, free_speed


def compute_traffic_scaling(route_json, lat, lon, tomtom_api_key: str):
    """Derive per-point traffic scaling factors and step metadata."""

    scaling = np.ones(len(lat))
    steps_metadata = []

    segments = route_json['features'][0]['properties'].get('segments', [])

    for seg_idx, seg in enumerate(segments):
        for step_idx, step in enumerate(seg.get('steps', [])):
            way_points = step.get("way_points", [0, 0])
            start_idx = max(0, min(int(way_points[0]), len(lat) - 1))
            end_idx = max(0, min(int(way_points[1]), len(lat) - 1))
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx

            mid_idx = (start_idx + end_idx) // 2
            midpoint_lat = lat[mid_idx]
            midpoint_lon = lon[mid_idx]

            ratio, current_speed, free_speed = _fetch_tomtom_flow_ratio(
                midpoint_lat, midpoint_lon, tomtom_api_key
            )

            if ratio is None:
                ratio = 1.0

            ratio = float(np.clip(ratio, 0.2, 1.2))
            scaling[start_idx:end_idx + 1] = ratio

            steps_metadata.append({
                "road_name": step.get("name", ""),
                "segment_index": seg_idx,
                "step_index": step_idx,
                "way_points": [start_idx, end_idx],
                "traffic_scaling": ratio,
                "tomtom_current_speed_kmh": current_speed,
                "tomtom_free_flow_speed_kmh": free_speed,
            })

    return scaling, steps_metadata


# ==============================================================
# 3. Elevation using OpenElevation API (free)
# ==============================================================

def get_elevation_profile(lat, lon, chunk_size=80):
    elevation = []
    coords = list(zip(lat, lon))

    for i in range(0, len(coords), chunk_size):
        chunk = coords[i:i+chunk_size]

        locations = "|".join([f"{c[0]},{c[1]}" for c in chunk])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

        try:
            r = requests.get(url)
            data = r.json()

            for res in data["results"]:
                elevation.append(res["elevation"])

        except Exception as e:
            print(f"[WARNING] elevation chunk failed: {e}")
            elevation.extend([0] * len(chunk))

        time.sleep(0.2)  # avoid rate limit

    return np.array(elevation)


# ==============================================================
# 4. Compute grade (%) from elevation
# ==============================================================

def compute_grade(elevation, lat, lon):
    grade = [0]

    for i in range(1, len(elevation)):
        # haversine distance
        lat1, lon1 = np.radians([lat[i-1], lon[i-1]])
        lat2, lon2 = np.radians([lat[i], lon[i]])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
        c = 2 * np.arcsin(np.sqrt(a))
        dx = 6371000 * c  # meters

        dh = elevation[i] - elevation[i-1]

        g = (dh / dx * 100) if dx > 0 else 0
        g = np.clip(g, -6, 6)

        grade.append(g)

    return np.array(grade)


# ==============================================================
# 5. Generate London-style speed profile
# ==============================================================

def generate_speed_profile(lat, lon):
    """
    London-style base speed: 10–35 km/h waves + random variability.
    """
    n = len(lat)

    base = []
    for i in range(n):
        cyc = (np.sin(2*np.pi*i/n*4) + 1) / 2
        spd = 12 + cyc * 23  # 12–35 km/h typical
        base.append(spd)

    return np.array(base)


# ==============================================================
# 6. Apply city traffic behavior
# ==============================================================

def apply_urban_events(speed):
    speed = speed.copy()
    n = len(speed)

    # Traffic lights
    for i in range(n):
        if np.random.rand() < 0.005:
            dur = np.random.randint(5, 12)
            for j in range(i, min(i+dur, n)):
                speed[j] = 0

    # Bus/taxi blocking
    for i in range(n):
        if np.random.rand() < 0.003:
            drop = np.random.randint(8, 18)
            for j in range(i, min(i+8, n)):
                speed[j] = max(0, speed[j] - drop)

    # Pedestrian crossing
    for i in range(n):
        if np.random.rand() < 0.002:
            dur = np.random.randint(3, 7)
            for j in range(i, min(i+dur, n)):
                speed[j] = 0

    # Add natural noise
    noise = np.random.normal(0, 2.0, n)
    speed = np.clip(speed + noise, 0, 45)

    return speed


# ==============================================================
# 7. Save as cycle.yaml
# ==============================================================

def save_cycle_yaml(cycles):
    """Persist one or more cycle definitions to disk.

    Parameters
    ----------
    cycles : Iterable[dict]
        Each dictionary must define at least the keys `filename`, `t`,
        `speed`, `grade`, `road_names`, `segment_lengths`, `lat`, `lon`,
        and `route_json`. Optional metadata such as `name`, `description`,
        and `routing_profile` can also be supplied per cycle.
    """

    for cycle in cycles:
        filename = cycle["filename"]
        name = cycle.get("name", "cycle")
        description = cycle.get("description", "Generated drive cycle")
        routing_profile = cycle.get("routing_profile", "driving-car")

        data = {
            "name": name,
            "description": description,
            "time_s": cycle["t"].tolist(),
            "speed_kmh": cycle["speed"].tolist(),
            "grade_percent": cycle["grade"].tolist(),
            "traffic_scaling": cycle.get("traffic_scaling", np.ones_like(cycle["speed"])).tolist(),
            "traffic_steps": cycle.get("traffic_steps", []),
            # Extra metadata (ignored by loader)
            "road_names": cycle["road_names"],
            "segment_lengths_m": cycle["segment_lengths"],
            "route_coords": {
                "lat": cycle["lat"],
                "lon": cycle["lon"],
            },
            "osm_summary": {
                "distance_m": cycle["route_json"]["features"][0]["properties"]["summary"]["distance"],
                "duration_s": cycle["route_json"]["features"][0]["properties"]["summary"]["duration"],
                "routing_profile": routing_profile,
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)

        print(f"Saved realistic London cycle → {filename}")


# ==============================================================
# Main Entry
# ==============================================================

def generate_london_osm_cycle(api_key: str | None = None):
    # Example route: UCL → Waterloo
    start = (51.5246, -0.1340)  # UCL
    end   = (51.5033, -0.1133)  # Waterloo Station

    ORS_API_KEY = resolve_ors_api_key(api_key)
    tomtom_api_key = resolve_tomtom_api_key()

    # 1. Get OSM route + geometry
    lat, lon, route_json = fetch_osm_route(start, end, ORS_API_KEY)

    # 2. Extract metadata
    road_names = extract_road_names(route_json)
    segment_lengths = extract_segment_lengths(route_json)

    # 3. Elevation
    elevation = get_elevation_profile(lat, lon)

    # 4. Grade (%)
    grade = compute_grade(elevation, lat, lon)

    # 5. Speed profile (base + live traffic)
    speed = generate_speed_profile(lat, lon)
    speed = apply_urban_events(speed)

    traffic_scaling, traffic_steps = compute_traffic_scaling(
        route_json, lat, lon, tomtom_api_key
    )
    speed = np.clip(speed * traffic_scaling, 0, 45)

    # 6. Convert to drive cycle time (10s interval)
    t = np.arange(0, len(speed)*10, 10)

    # 7. Save cycle YAML (pass as list to allow multiple future cycles)
    save_cycle_yaml([
        {
            "filename": str(Path(__file__).parent.parent / "data" / "cycle.yaml"),
            "name": "london_osm_urban",
            "description": "OSM-based London route with traffic lights, congestion, elevation, and road metadata.",
            "t": t,
            "speed": speed,
            "grade": grade,
            "traffic_scaling": traffic_scaling,
            "traffic_steps": traffic_steps,
            "road_names": road_names,
            "segment_lengths": segment_lengths,
            "lat": lat,
            "lon": lon,
            "route_json": route_json,
            "routing_profile": "driving-car",
        }
    ])
    
    # 8. Save cycle YAML for another cycle (Canary Wharf → UCL)
    start2 = (51.5033, -0.0195)  # Canary Wharf
    end2 = (51.5246, -0.1340)    # UCL
    
    lat2, lon2, route_json2 = fetch_osm_route(start2, end2, ORS_API_KEY)
    road_names2 = extract_road_names(route_json2)
    segment_lengths2 = extract_segment_lengths(route_json2)
    elevation2 = get_elevation_profile(lat2, lon2)
    grade2 = compute_grade(elevation2, lat2, lon2)
    speed2 = generate_speed_profile(lat2, lon2)
    speed2 = apply_urban_events(speed2)
    t2 = np.arange(0, len(speed2)*10, 10)
    
    save_cycle_yaml([
        {
            "filename": str(Path(__file__).parent.parent / "data" / "cycle_canary_to_ucl.yaml"),
            "name": "london_osm_canary_to_ucl",
            "description": "OSM-based London route from Canary Wharf to UCL with traffic lights, congestion, elevation, and road metadata.",
            "t": t2,
            "speed": speed2,
            "grade": grade2,
            "road_names": road_names2,
            "segment_lengths": segment_lengths2,
            "lat": lat2,
            "lon": lon2,
            "route_json": route_json2,
            "routing_profile": "driving-car",
        }
    ])
    
    # 8. Save Cambridge cycle YAML
    start3 = (52.2061, 0.1333)  # Canary Wharf
    end3 = (51.5003, 0.0187)    # UCL
    
    lat3, lon3, route_json3 = fetch_osm_route(start3, end3, ORS_API_KEY)
    road_names3 = extract_road_names(route_json3)
    segment_lengths3 = extract_segment_lengths(route_json3)
    elevation3 = get_elevation_profile(lat3, lon3)
    grade3 = compute_grade(elevation3, lat3, lon3)
    speed3 = generate_speed_profile(lat3, lon3)
    speed3 = apply_urban_events(speed3)
    t3 = np.arange(0, len(speed3)*10, 10)
    
    save_cycle_yaml([
        {
            "filename": str(Path(__file__).parent.parent / "data" / "cambridge.yaml"),
            "name": "cambridge_osm_urban",
            "description": "OSM-based Cambridge route.",
            "t": t3,
            "speed": speed3,
            "grade": grade3,
            "road_names": road_names3,
            "segment_lengths": segment_lengths3,
            "lat": lat3,
            "lon": lon3,
            "route_json": route_json3,
            "routing_profile": "driving-car",
        }
    ])

if __name__ == "__main__":
    generate_london_osm_cycle()
