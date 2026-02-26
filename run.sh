#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_dir="${repo_root}/data"
location="London"
runs=30
speed_noise_std=0.2
output_file="paired_differences.yaml"
baseline_output="${data_dir}/${location}_cycle_baseline.yaml"
optimized_output="${data_dir}/${location}_cycle_optimized.yaml"

wait_for_file() {
  local file_path="$1"
  while [[ ! -s "${file_path}" ]]; do
    echo "Waiting for ${file_path} to be ready..."
    sleep 1
  done
}

mkdir -p "${data_dir}"
rm -f "${data_dir}"/*
echo "Cleared data directory: ${data_dir}"

echo "Generating baseline route..."
python "${repo_root}/src/generate_london_osm_cycle_baseline.py" --location "${location}"
wait_for_file "${baseline_output}"

echo "Generating optimized route..."
python "${repo_root}/src/generate_london_osm_cycle_optimized.py" --location "${location}"
wait_for_file "${optimized_output}"

echo "Running ${runs} paired simulations and exporting differences..."
python "${repo_root}/src/main.py" --location "${location}" --runs "${runs}" --speed-noise-std "${speed_noise_std}" --output "${output_file}"

rm -f "${baseline_output}" "${optimized_output}"
echo "Done. Saved required paired differences at ${data_dir}/${output_file}"
