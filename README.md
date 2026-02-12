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