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

## TomTom Traffic API key

`generate_london_osm_cycle.py` now scales the generated speed profile using live traffic
conditions from TomTom's Flow Segment Data API. Add your TomTom key to `.env` as
`TOM_API_KEY` (or `TOMTOM_API_KEY`/`tom_api_key`) so the generator can fetch the
congestion coefficients for each road segment.
