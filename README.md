# Invididual_Project_Scooter_Simulation
MECH0020 electrical scooter energy usage simulation

## OpenRouteService API key

The `generate_london_osm_cycle.py` utility needs a valid OpenRouteService token. To keep
the repository free of secrets, copy `.env.example` to `.env` (which is ignored by git)
and paste your key there, or export the variable before running the script:

```bash
cp .env.example .env
echo "ORS_API_KEY=sk_your_real_token" >> .env

# or, alternatively
export ORS_API_KEY=sk_your_real_token
python src/generate_london_osm_cycle.py
```

The script now reads the key from the function argument, the environment, or the `.env`
file—whichever is provided first.

## Vehicle and rider parameters

The energy model now distinguishes between the scooter and rider masses so that the
impact of driver weight on grade, rolling, and inertial loads can be studied. Set the
`driver.mass_kg` field in `data/scooter_params.yaml` to explore different body weights.
Simulation output also reports a propulsion-only energy breakdown for grade (slope),
inertial acceleration, rolling resistance, and aerodynamic drag to highlight how road
incline and air resistance contribute to total consumption. For slope analysis, the
results now include net gravitational work (positive when climbing, negative when
descending) and the downhill assistance recovered by regen to explicitly show gravity's
effect on energy use.
