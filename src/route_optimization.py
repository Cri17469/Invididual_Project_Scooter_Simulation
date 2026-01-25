import json
import os
import time
from pathlib import Path
from typing import Iterable

import networkx as nx
import osmnx as ox
import numpy as np
import requests
import openrouteservice as ors
from shapely.geometry import Point


from energy_model import simulate_energy
from utils import get_data_dir
from vehicle_params import load_vehicle_params, VehicleParams

# ==============================================================
# Config
# ==============================================================

MAX_SPEED_KMH = 25.0
DEFAULT_WEIGHTS = {
    "energy": 5.0,
    "time": 1.0,
}
DEFAULT_NETWORK_TYPE = "bike"
DEFAULT_BUFFER_M = 3000
DEFAULT_GRAPH_FILENAME = "london_osm_graph.graphml"
DEFAULT_ROUTE_FILENAME = "optimized_route.json"


# ==============================================================
# Location helpers
# ==============================================================


def resolve_location(location: str | tuple[float, float] | list[float]) -> tuple[float, float]:
    if isinstance(location, str):
        ors_api_key = _resolve_ors_api_key()
        if ors_api_key:
            try:
                lat, lon = _geocode_with_ors(location, ors_api_key)
                return float(lat), float(lon)
            except (ValueError, ors.exceptions.ApiError, requests.RequestException):
                pass
        lat, lon = ox.geocode(location)
        return float(lat), float(lon)

    if isinstance(location, (tuple, list)) and len(location) == 2:
        return float(location[0]), float(location[1])

    raise ValueError("Location must be a place name string or (lat, lon) tuple.")


def _read_api_key_from_env_file(var_name: str, env_filename: str = ".env") -> str | None:
    env_path = Path(__file__).resolve().parent.parent / env_filename
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
        return None

    return None


def _resolve_ors_api_key() -> str | None:
    api_key = os.getenv("ORS_API_KEY")
    if api_key:
        return api_key
    return _read_api_key_from_env_file("ORS_API_KEY")


def _geocode_with_ors(location: str, api_key: str) -> tuple[float, float]:
    client = ors.Client(key=api_key)
    response = client.pelias_search(text=location)
    features = response.get("features", [])
    if not features:
        raise ValueError("OpenRouteService did not return any geocoding results.")
    coordinates = features[0].get("geometry", {}).get("coordinates")
    if not coordinates or len(coordinates) < 2:
        raise ValueError("OpenRouteService response missing coordinates.")
    lon, lat = coordinates[:2]
    return float(lat), float(lon)


def _bbox_from_points(points: Iterable[tuple[float, float]], buffer_m: float) -> tuple[float, float, float, float]:
    boxes = [
        ox.utils_geo.bbox_from_point(point, dist=buffer_m)
        for point in points
    ]
    north = max(box[0] for box in boxes)
    south = min(box[1] for box in boxes)
    east = max(box[2] for box in boxes)
    west = min(box[3] for box in boxes)
    return north, south, east, west


# ==============================================================
# Graph + cost model
# ==============================================================


def _normalize_speed_kph(speed_value: float | str | list | None, fallback: float) -> float:
    if speed_value is None:
        return fallback
    if isinstance(speed_value, list) and speed_value:
        speed_value = speed_value[0]
    try:
        return float(speed_value)
    except (TypeError, ValueError):
        return fallback


def simulate_edge_energy_wh(
    length_m: float,
    speed_kph: float,
    grade_percent: float,
    params: VehicleParams,
) -> float:
    if length_m <= 0.0 or speed_kph <= 0.0:
        return 0.0
    time_s = length_m / (speed_kph / 3.6)
    steps = max(int(round(time_s)) + 1, 2)
    t = np.linspace(0.0, time_s, steps)
    v = np.full_like(t, speed_kph / 3.6)
    theta = np.full_like(t, np.arctan(grade_percent / 100.0))
    cycle = {
        "t": t,
        "v": v,
        "theta": theta,
    }
    result = simulate_energy(cycle, params, plot=False)
    return max(float(result["E_Wh"]), 0.0)


def _fetch_node_elevations(graph: nx.MultiDiGraph, chunk_size: int = 80, pause_s: float = 0.2) -> None:
    nodes = list(graph.nodes(data=True))
    coords = [(node_id, data.get("y"), data.get("x")) for node_id, data in nodes]
    for i in range(0, len(coords), chunk_size):
        chunk = coords[i:i + chunk_size]
        locations = "|".join([f"{lat},{lon}" for _, lat, lon in chunk])
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            for (node_id, _, _), result in zip(chunk, data.get("results", []), strict=False):
                graph.nodes[node_id]["elevation"] = float(result.get("elevation", 0.0))
        except requests.RequestException:
            for node_id, _, _ in chunk:
                graph.nodes[node_id]["elevation"] = 0.0
        time.sleep(pause_s)


def _ensure_edge_grades(
    graph: nx.MultiDiGraph,
    graph_filename: str = DEFAULT_GRAPH_FILENAME,
) -> None:
    if any("elevation" not in data for _, data in graph.nodes(data=True)):
        _fetch_node_elevations(graph)
    ox.add_edge_grades(graph, add_absolute=True)
    data_dir = get_data_dir()
    graph_path = data_dir / graph_filename
    ox.save_graphml(graph, graph_path)


def _extract_grade_percent(edge_data: dict) -> float:
    grade = edge_data.get("grade")
    if grade is None:
        return 0.0
    return float(grade) * 100.0


def build_graph(
    origin: tuple[float, float],
    destination: tuple[float, float],
    network_type: str = DEFAULT_NETWORK_TYPE,
    buffer_m: float = DEFAULT_BUFFER_M,
    graph_filename: str = DEFAULT_GRAPH_FILENAME,
    use_cache: bool = True,
) -> nx.MultiDiGraph:
    data_dir = get_data_dir()
    graph_path = data_dir / graph_filename

    if use_cache and graph_path.exists():
        graph = ox.load_graphml(graph_path)
    else:
        north, south, east, west = _bbox_from_points([origin, destination], buffer_m)
        try:
            graph = ox.graph_from_bbox(
                bbox=(north, south, east, west),
                network_type=network_type,
                simplify=True,
            )
        except TypeError:
            graph = ox.graph_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                network_type=network_type,
                simplify=True,
            )
        ox.save_graphml(graph, graph_path)

    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    return graph


def optimize_route(
    origin: str | tuple[float, float] | list[float],
    destination: str | tuple[float, float] | list[float],
    weights: dict[str, float] | None = None,
    max_speed_kmh: float = MAX_SPEED_KMH,
    network_type: str = DEFAULT_NETWORK_TYPE,
    buffer_m: float = DEFAULT_BUFFER_M,
    graph_filename: str = DEFAULT_GRAPH_FILENAME,
) -> dict:
    resolved_origin = resolve_location(origin)
    resolved_destination = resolve_location(destination)
    weights = weights or DEFAULT_WEIGHTS
    params = load_vehicle_params()


    def nearest_node_fallback(graph: nx.MultiDiGraph, x: float, y: float) -> int:
        nodes = graph.nodes
        closest_node = None
        closest_dist = float("inf")
        for node_id, data in nodes.items():
            dx = data.get("x", 0.0) - x
            dy = data.get("y", 0.0) - y
            dist = dx * dx + dy * dy
            if dist < closest_dist:
                closest_dist = dist
                closest_node = node_id
        if closest_node is None:
            raise RuntimeError("Unable to determine nearest node for optimized route.")
        return int(closest_node)

    def resolve_nodes(graph: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, int, int]:
        graph_proj = ox.project_graph(graph)
        origin_x, origin_y = ox.projection.project_geometry(
            Point(resolved_origin[1], resolved_origin[0]),
            to_crs=graph_proj.graph["crs"],
        )[0].coords[0]
        dest_x, dest_y = ox.projection.project_geometry(
            Point(resolved_destination[1], resolved_destination[0]),
            to_crs=graph_proj.graph["crs"],
        )[0].coords[0]

        try:
            origin_node = ox.nearest_nodes(graph_proj, origin_x, origin_y)
            destination_node = ox.nearest_nodes(graph_proj, dest_x, dest_y)
        except ImportError:
            origin_node = nearest_node_fallback(graph_proj, origin_x, origin_y)
            destination_node = nearest_node_fallback(graph_proj, dest_x, dest_y)

        if not nx.has_path(graph_proj.to_undirected(), origin_node, destination_node):
            graph_proj = ox.utils_graph.get_largest_component(graph_proj, strongly=False)
            try:
                origin_node = ox.nearest_nodes(graph_proj, origin_x, origin_y)
                destination_node = ox.nearest_nodes(graph_proj, dest_x, dest_y)
            except ImportError:
                origin_node = nearest_node_fallback(graph_proj, origin_x, origin_y)
                destination_node = nearest_node_fallback(graph_proj, dest_x, dest_y)

        return graph_proj, origin_node, destination_node

    graph = build_graph(
        resolved_origin,
        resolved_destination,
        network_type=network_type,
        buffer_m=buffer_m,
        graph_filename=graph_filename,
    )
    _ensure_edge_grades(graph, graph_filename=graph_filename)
    graph_proj, origin_node, destination_node = resolve_nodes(graph)

    for u, v, k, data in graph_proj.edges(keys=True, data=True):
        length_m = float(data.get("length", 0.0))
        speed_kph = _normalize_speed_kph(data.get("speed_kph"), max_speed_kmh)
        speed_kph = min(speed_kph, max_speed_kmh)
        time_s = length_m / (speed_kph / 3.6) if speed_kph > 0 else 0.0
        grade_percent = _extract_grade_percent(data)
        energy_wh = simulate_edge_energy_wh(length_m, speed_kph, grade_percent, params)
        data["cost"] = weights.get("energy", 1.0) * energy_wh + weights.get("time", 1.0) * time_s
        data["energy_est_Wh"] = energy_wh
        data["time_est_s"] = time_s

    try:
        route_nodes = nx.shortest_path(graph_proj, origin_node, destination_node, weight="cost")
    except nx.NetworkXNoPath:
        graph = build_graph(
            resolved_origin,
            resolved_destination,
            network_type=network_type,
            buffer_m=buffer_m * 2,
            graph_filename=graph_filename,
            use_cache=False,
        )
        graph_proj, origin_node, destination_node = resolve_nodes(graph)
        route_nodes = nx.shortest_path(graph_proj, origin_node, destination_node, weight="cost")

    coords = [
        {
            "lat": float(graph.nodes[node]["y"]),
            "lon": float(graph.nodes[node]["x"]),
        }
        for node in route_nodes
    ]

    route_length_m = 0.0
    route_time_s = 0.0
    route_energy_est_Wh = 0.0
    road_names: list[str] = []
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_data = graph_proj.get_edge_data(u, v)
        if not edge_data:
            continue
        edge = edge_data[min(edge_data.keys())]
        route_length_m += float(edge.get("length", 0.0))
        route_time_s += float(edge.get("time_est_s", 0.0))
        route_energy_est_Wh += float(edge.get("energy_est_Wh", 0.0))
        edge_name = edge.get("name")
        if isinstance(edge_name, list):
            edge_name = edge_name[0] if edge_name else None
        if isinstance(edge_name, str):
            edge_name = edge_name.strip()
        if edge_name and edge_name not in road_names:
            road_names.append(edge_name)

    return {
        "origin": {
            "lat": resolved_origin[0],
            "lon": resolved_origin[1],
        },
        "destination": {
            "lat": resolved_destination[0],
            "lon": resolved_destination[1],
        },
        "nodes": route_nodes,
        "coords": coords,
        "road_names": road_names,
        "summary": {
            "distance_m": route_length_m,
            "duration_s": route_time_s,
            "energy_est_Wh": route_energy_est_Wh,
            "weights": weights,
            "max_speed_kmh": max_speed_kmh,
            "network_type": network_type,
        },
    }


def save_optimized_route(route_data: dict, filename: str = DEFAULT_ROUTE_FILENAME) -> Path:
    data_dir = get_data_dir()
    output_path = data_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(route_data, f, indent=2)
    return output_path


def load_optimized_route(filename: str = DEFAULT_ROUTE_FILENAME) -> dict:
    data_dir = get_data_dir()
    input_path = data_dir / filename
    if not input_path.exists():
        raise FileNotFoundError(f"Optimized route not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================
# How to run
# ==============================================================
# python src/route_optimization.py


if __name__ == "__main__":
    route = optimize_route(
        origin=(51.5246, -0.1340),
        destination=(51.5033, -0.1133),
        weights={"energy": 1.0, "time": 1.0},
    )
    output = save_optimized_route(route)
    print(f"Saved optimized route to {output}")
