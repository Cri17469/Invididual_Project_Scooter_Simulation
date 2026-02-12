# Invididual_Project_Scooter_Simulation
MECH0020 electrical scooter energy usage simulation

## OpenRouteService API key

The `generate_london_osm_cycle.py` utility needs a valid OpenRouteService token. The
route optimization helpers will also use the same `ORS_API_KEY` (when provided) for
geocoding place names. To keep the repository free of secrets, copy `.env.example` to
`.env` (which is ignored by git) and paste your key there, or export the variable
before running the script:

## 1) Theory of Velocity Simulation and Energy Simulation

The simulation is driven by the `simulate_energy(cycle, params, plot=False)` function in `src/energy_model.py`.
It expects a driving cycle dictionary with:

- `t`: time array (s)
- `v`: speed array (m/s)
- `theta`: road grade angle array (rad)

### Velocity-side dynamics used by the model

Inside `simulate_energy`, acceleration is calculated from the speed trace:

- `a = np.gradient(v, t)`

Then longitudinal forces are built from classical vehicle dynamics:

- Inertial force: `F_inert = m_total * a`
- Grade force: `F_grade = m_total * g * sin(theta)`
- Rolling force: `F_roll = Cr * m_total * g * cos(theta)`
- Aerodynamic force: `F_aero = 0.5 * rho_air * CdA * v^2`

Total traction force is:

- `F_trac = F_inert + F_grade + F_roll + F_aero`

and wheel power is:

- `P_wheel = F_trac * v`

This means the velocity profile (`v`) directly determines acceleration, forces, and finally power demand over time.

### Energy simulation theory implemented

Battery power is mapped from wheel power with motoring and regen efficiencies:

- If `P_wheel > 0`: `P_bat = P_wheel / eta_m`
- If `P_wheel <= 0`: `P_bat = eta_regen * P_wheel`

Then hard limits are applied:

- Regen charge limit: `P_bat >= -Pchg_max`
- Discharge limit: `P_bat <= Pdis_max`

Net battery energy is integrated with trapezoidal integration:

- `E_Wh = ∫ P_bat dt / 3600`

Distance and intensity metric:

- `distance_km = ∫ v dt / 1000`
- `Wh_per_km = E_Wh / distance_km`

The same function also decomposes energy components using helper functions:

- `signed_energy(power_trace)`
- `positive_energy(power_trace)`

It reports propulsion-side components in `component_energy_Wh`:

- inertial
- grade
- undulation
- rolling
- aerodynamic

Regenerative braking outputs are also computed in `simulate_energy`:

- `P_regen = max(-P_bat, 0)`
- `E_regen_Wh = ∫ P_regen dt / 3600`
- `regen_fraction = E_regen_Wh / braking_energy_raw`
- `regen_peak_power_W = max(P_regen)`

So the core function for both velocity-based dynamics and energy accounting is:

- `simulate_energy(...)` in `src/energy_model.py`

## 2) Modelling of Battery, Voltage, Regeneration, and Related Terms

Battery and electrical parameters are defined in `params/scooter_params.yaml` and loaded by:

- `load_vehicle_params(filename="scooter_params.yaml")` in `src/vehicle_params.py`

This function returns a `VehicleParams` dataclass used by the simulator.

### Battery model currently implemented

The implemented battery behavior in code is a **power-limited energy bucket + SoC integrator**:

1. **Capacity model**
   - Parameter: `Cap_Wh`
   - Used to convert power flow into SoC change.

2. **Charge/discharge power constraints**
   - `Pchg_max` (maximum regenerative charging power)
   - `Pdis_max` (maximum discharge power)
   - Enforced in `simulate_energy` using clamp operations.

3. **State of Charge (SoC) propagation**
   - Initialized from `soc0`
   - Time-step update in loop:
     - `dWh = P_bat[i] * dt[i] / 3600`
     - `SoC[i] = clip(SoC[i-1] - dWh / Cap_Wh, 0, 1)`

4. **Returned battery outputs**
   - `SoC_trace`
   - `final_SoC`
   - `E_Wh`, `Wh_per_km`

### Voltage modelling status in this codebase

`V_nom` (nominal voltage) is loaded into `VehicleParams`, but there is currently **no explicit dynamic voltage equation** (e.g., OCV(SOC), internal resistance drop, RC transient) in `simulate_energy`.

So at present:

- Voltage is a stored parameter (`V_nom`) available for future extensions.
- Energy and SoC are simulated directly from power integration.
- Current (`I = P/V`) is not explicitly solved in the present model.

### Regeneration modelling details

Regeneration is modelled in `simulate_energy` by:

- Applying motor regen efficiency (`eta_regen`) when wheel power is negative.
- Enforcing maximum charging power (`Pchg_max`).
- Integrating negative battery power as recovered energy (`E_regen_Wh`).
- Reporting recovery quality with `regen_fraction` and `regen_peak_power_W`.

### Main functions you can follow in code

- `simulate_energy(cycle, params, plot=False)` → full dynamics + battery/regen + SoC (`src/energy_model.py`)
- `load_vehicle_params(filename="scooter_params.yaml")` → reads battery/motor/environment parameters (`src/vehicle_params.py`)
- `build_cycle(...)` and `main()` in `src/main.py` → build comparison cycles and call `simulate_energy` for baseline/optimized routes

## 3) Analysis of the Time-Energy Weighted Routing Algorithm

Your time-energy weighted logic is implemented in `optimize_route(...)` inside `src/route_optimization.py`, where each graph edge is assigned a scalar cost and Dijkstra-style shortest-path selection is performed with `nx.shortest_path(..., weight="cost")`.

### Core optimization equation used in code

For each edge `(u, v, k)`, the code computes:

- `time_s = length_m / (speed_kph / 3.6)`
- `energy_wh = simulate_edge_energy_wh(length_m, speed_kph, grade_percent, params)`
- `cost = w_energy * energy_wh + w_time * time_s`

with:

- `w_energy = weights.get("energy", 1.0)`
- `w_time = weights.get("time", 1.0)`

This is a linear weighted-sum multi-objective model (energy + travel time) converted to one objective.

### How edge energy is estimated (function chain)

`simulate_edge_energy_wh(...)` builds a synthetic constant-speed edge cycle and reuses the same physics/energy model as your main simulation:

1. Build edge-local cycle arrays:
   - `t = np.linspace(0.0, time_s, steps)`
   - `v = np.full_like(t, speed_kph / 3.6)`
   - `theta = np.full_like(t, np.arctan(grade_percent / 100.0))`
2. Call `simulate_energy(cycle, params, plot=False)`.
3. Return `max(result["E_Wh"], 0.0)` so negative net edge energy is clipped to zero in routing cost.

Interpretation: downhill/regen segments can reduce net pack energy in physics space, but the route cost model does not allow negative edge cost (avoids pathological “energy-harvesting loops” in shortest-path search).

### Why this algorithm is practical

- **Consistent physics basis**: routing energy uses the exact same `simulate_energy(...)` model as end-to-end cycle evaluation.
- **Tunable behavior**: by changing `weights`, you smoothly shift between faster routes and lower-energy routes.
- **Robust graph search**: once each edge has a single `cost`, standard shortest-path is stable and efficient.

### Parameter and preprocessing effects on results

- Speed used per edge is normalized by `_normalize_speed_kph(...)` and capped by `max_speed_kmh`, which directly affects both `time_s` and `energy_wh`.
- Grade comes from `_extract_grade_percent(edge_data)`, so elevation quality/availability influences energy discrimination between routes.
- If no path exists, `optimize_route(...)` retries with a larger graph buffer, then reruns shortest path.

### Units and weighting caveat (important for essay discussion)

Because the cost is `Wh` + `s` weighted by raw coefficients, the coefficients must also serve as **unit conversion/tradeoff scalers**. In other words, `w_energy` and `w_time` are not only “preferences”; they implicitly convert and balance two different units.

A useful way to describe this in your essay:

- increasing `w_time` biases toward shorter duration even if Wh rises,
- increasing `w_energy` biases toward lower Wh even if travel time rises,
- the numerical ratio `w_time / w_energy` determines the effective value of one second relative to one watt-hour in the optimizer.

### Main functions to cite for this algorithm

- `optimize_route(...)` (edge cost construction + shortest path)
- `simulate_edge_energy_wh(...)` (edge-level physical energy estimate)
- `_normalize_speed_kph(...)` and `_extract_grade_percent(...)` (edge attribute normalization)
- `simulate_energy(...)` (physics + battery/regen backend reused by routing)
