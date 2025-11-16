import os
import time
import requests
import numpy as np
import yaml
import osmnx as ox
import openrouteservice as ors
from pathlib import Path

# ==========================================================
# 1. Fetch OSM route using OpenRouteService
# ==========================================================


def fetch_osm_route(start, end, api_key):
    """
    start/end = (lat, lon)
    returns coordinates list with many points
    """
    client = ors.Client(key=api_key)

    route = client.directions(
        coordinates=[(start[1], start[0]), (end[1], end[0])],
        profile='driving-car',
        format='geojson'
    )

    coords = route['features'][0]['geometry']['coordinates']
    # coords are [lon,lat]
    lat = [c[1] for c in coords]
    lon = [c[0] for c in coords]

    return lat, lon


# ==========================================================
# 2. Elevation using OSMnx + SRTM
# ==========================================================

def get_elevation_profile(lat, lon, chunk_size=80):
    """
    Use Open-Elevation (free) to fetch elevation for a list of coordinates.
    Returns elevation array (meters).
    """

    elevation = []

    coords = list(zip(lat, lon))

    for i in range(0, len(coords), chunk_size):
        chunk = coords[i:i+chunk_size]

        # format: lat,lon|lat,lon|...
        locations = "|".join([f"{c[0]},{c[1]}" for c in chunk])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"

        try:
            r = requests.get(url)
            data = r.json()

            for result in data["results"]:
                elevation.append(result["elevation"])

        except Exception as e:
            print(f"[WARNING] Elevation chunk failed: {e}. Using 0m.")
            elevation.extend([0]*len(chunk))

        time.sleep(0.2)  # avoid rate limit

    return np.array(elevation)

# ==========================================================
# 3. Generate speed profile based on real road maxspeed
# ==========================================================


def generate_speed_profile(lat, lon):
    """
    Converts OSM route geometry into a London-like speed pattern.
    - Base speed from OSM maxspeed (if available)
    - Add London behavior: congestion, slowdowns, signals
    """
    # Load driving graph around route
    G = ox.graph_from_point((lat[0], lon[0]), dist=2000, network_type='drive')

    # ⭐ Project graph to UTM (no sklearn required)
    G = ox.project_graph(G)

    speed = []

    for i in range(len(lat) - 1):
        try:
            # nearest nodes in projected graph
            u = ox.distance.nearest_nodes(G, lon[i], lat[i])
            v = ox.distance.nearest_nodes(G, lon[i+1], lat[i+1])

            # get maxspeed attribute
            maxspeed = G[u][v][0].get("maxspeed")
            if isinstance(maxspeed, list):
                ms = int(maxspeed[0])
            else:
                ms = int(maxspeed) if maxspeed else 30

        except Exception:
            ms = 30  # default fallback

        # London typical speeds (maxspeed rarely reached)
        base_speed = np.clip(np.random.normal(ms * 0.6, 4), 5, ms)
        speed.append(base_speed)

    speed.append(speed[-1])
    return np.array(speed)


# ==========================================================
# 4. Convert elevation to grade (%)
# ==========================================================

def compute_grade(elevation, lat, lon):
    grade = [0]

    for i in range(1, len(elevation)):
        dx = ox.distance.great_circle(
            lat[i-1], lon[i-1], lat[i], lon[i])  # meters
        dh = elevation[i] - elevation[i-1]
        g = (dh / dx) * 100 if dx > 1 else 0
        grade.append(np.clip(g, -6, 6))

    return np.array(grade)


# ==========================================================
# 5. Add urban factors: lights, bus stops, random events
# ==========================================================

def apply_urban_disruptions(speed):
    n = len(speed)
    speed = speed.copy()

    # Traffic lights
    for i in range(n):
        if np.random.random() < 0.005:
            dur = np.random.randint(3, 12)
            for j in range(i, min(i+dur, n)):
                speed[j] = 0

    # Bus blocking
    for i in range(n):
        if np.random.random() < 0.003:
            drop = np.random.randint(10, 20)
            for j in range(i, min(i+8, n)):
                speed[j] = max(0, speed[j] - drop)

    # Random pedestrian stop
    for i in range(n):
        if np.random.random() < 0.002:
            dur = np.random.randint(5, 10)
            for j in range(i, min(i+dur, n)):
                speed[j] = 0

    # Noise (natural variability)
    noise = np.random.normal(0, 2, n)
    speed = np.clip(speed + noise, 0, 45)

    return speed


# ==========================================================
# 6. Save cycle as YAML
# ==========================================================

def save_cycle_yaml(name, description, t, speed, grade, filename="cycle.yaml"):
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    filepath = data_dir / filename

    data = {
        "name": name,
        "description": description,
        "time_s": t.tolist(),
        "speed_kmh": speed.tolist(),
        "grade_percent": grade.tolist()
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    print(f"Saved to {filepath}")


# ==========================================================
# Main Logic
# ==========================================================

def generate_london_osm_cycle():
    # EXAMPLE: UCL → Waterloo Station
    start = (51.5246, -0.1340)  # UCL
    end = (51.5033, -0.1133)  # Waterloo Station

    API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImMxOTJlZTZmOGM2OTRhNjc4MWVlYmFlODVkNWMwNmFkIiwiaCI6Im11cm11cjY0In0="

    # 1. OSM route
    lat, lon = fetch_osm_route(start, end, API_KEY)

    # 2. elevation
    elevation = get_elevation_profile(lat, lon)

    # 3. speed profile base
    speed = generate_speed_profile(lat, lon)

    # 4. grade
    grade = compute_grade(elevation, lat, lon)

    # 5. add urban events
    speed = apply_urban_disruptions(speed)

    # 6. time
    t = np.arange(0, len(speed)*10, 10)

    # 7. save
    save_cycle_yaml(
        name="london_osm_urban",
        description="Real OSM London cycle with elevation + traffic behaviors",
        t=t,
        speed=speed,
        grade=grade,
        filename="cycle.yaml"
    )


if __name__ == "__main__":
    generate_london_osm_cycle()
