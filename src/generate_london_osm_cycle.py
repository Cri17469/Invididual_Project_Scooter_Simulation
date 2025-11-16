import numpy as np
import yaml
import requests
import time
import openrouteservice as ors
from pathlib import Path


# ==============================================================
# 1. Get real OSM route using OpenRouteService
# ==============================================================

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

def save_cycle_yaml(filename, t, speed, grade, road_names, segment_lengths, lat, lon, route_json):
    data = {
        "name": "london_osm_urban",
        "description": "OSM-based London route with traffic lights, congestion, elevation, and road metadata.",
        "time_s": t.tolist(),
        "speed_kmh": speed.tolist(),
        "grade_percent": grade.tolist(),

        # Extra metadata (ignored by loader)
        "road_names": road_names,
        "segment_lengths_m": segment_lengths,
        "route_coords": {
            "lat": lat,
            "lon": lon
        },
        "osm_summary": {
            "distance_m": route_json['features'][0]['properties']['summary']['distance'],
            "duration_s": route_json['features'][0]['properties']['summary']['duration'],
            "routing_profile": "driving-car"
        }
    }

    with open(filename, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    print(f"Saved realistic London cycle → {filename}")


# ==============================================================
# Main Entry
# ==============================================================

def generate_london_osm_cycle():
    # Example route: UCL → Waterloo
    start = (51.5246, -0.1340)  # UCL
    end   = (51.5033, -0.1133)  # Waterloo Station

    # TODO: replace with your real API key
    ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImMxOTJlZTZmOGM2OTRhNjc4MWVlYmFlODVkNWMwNmFkIiwiaCI6Im11cm11cjY0In0="

    # 1. Get OSM route + geometry
    lat, lon, route_json = fetch_osm_route(start, end, ORS_API_KEY)

    # 2. Extract metadata
    road_names = extract_road_names(route_json)
    segment_lengths = extract_segment_lengths(route_json)

    # 3. Elevation
    elevation = get_elevation_profile(lat, lon)

    # 4. Grade (%)
    grade = compute_grade(elevation, lat, lon)

    # 5. Speed profile
    speed = generate_speed_profile(lat, lon)
    speed = apply_urban_events(speed)

    # 6. Convert to drive cycle time (10s interval)
    t = np.arange(0, len(speed)*10, 10)

    # 7. Save cycle YAML
    save_cycle_yaml(
        filename=str(Path(__file__).parent.parent / "data" / "cycle.yaml"),
        t=t,
        speed=speed,
        grade=grade,
        road_names=road_names,
        segment_lengths=segment_lengths,
        lat=lat,
        lon=lon,
        route_json=route_json
    )


if __name__ == "__main__":
    generate_london_osm_cycle()
