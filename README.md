# Invididual_Project_Scooter_Simulation
MECH0020 electrical scooter energy usage simulation

## OpenRouteService API key

The `generate_london_osm_cycle.py` utility needs a valid OpenRouteService token. The
route optimization helpers will also use the same `ORS_API_KEY` (when provided) for
geocoding place names. To keep the repository free of secrets, copy `.env.example` to
`.env` (which is ignored by git) and paste your key there, or export the variable
before running the script:
