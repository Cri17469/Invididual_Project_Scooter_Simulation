#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_dir="${repo_root}/data"
location="CrystalPalace-Greenwich"
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

echo "Running the results output script..."
python "${repo_root}/src/main.py" --location "${location}"

echo "Task completed. Detailed results are available in ${baseline_output} and ${optimized_output}."
