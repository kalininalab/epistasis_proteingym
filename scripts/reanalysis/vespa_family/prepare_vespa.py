#!/usr/bin/env python
import argparse
from pathlib import Path

ROOT = Path("/data/users/akolchina/epistasis_proteingym")
HERE = ROOT / "scripts/reanalysis/vespa_family"
BASE = ROOT / "results/reanalysis/tsuboyama/vespa_family"
parser = argparse.ArgumentParser()
parser.add_argument("--pilot", action="store_true")
args = parser.parse_args()
datasets = [x.stem for x in sorted((ROOT / "results/tables/intermediate/tsuboyama_epistatic").glob("*.csv"))]
if args.pilot:
    datasets = ["PIN1_HUMAN_Tsuboyama_2023_1I6C"]
tag = "vespa_pilot2" if args.pilot else "vespa_full"
directory = BASE / "condor" / tag
logs = directory / "logs"
logs.mkdir(parents=True, exist_ok=True)
common = f"""universe = vanilla
executable = {HERE}/run_vespa.sh
initialdir = {ROOT}
should_transfer_files = NO
getenv = False
log = {logs}/events.log
"""
(directory / "gpu.sub").write_text(common + f"""arguments = score $(dataset) $(limit) $(namespace)
request_cpus = 2
request_gpus = 1
request_memory = 28GB
request_disk = 8GB
+MaxRuntime = 259200
output = {logs}/gpu.$(ClusterId).out
error = {logs}/gpu.$(ClusterId).err
queue
""")
(directory / "cpu.sub").write_text(common + f"""arguments = fit $(dataset) none $(namespace)
request_cpus = 1
request_memory = 5GB
output = {logs}/cpu.$(ClusterId).out
error = {logs}/cpu.$(ClusterId).err
queue
""")
limit = "8" if args.pilot else "none"
lines = []
for i, dataset in enumerate(datasets):
    lines += [f"JOB S_{i:02d} {directory}/gpu.sub",
              f'VARS S_{i:02d} dataset="{dataset}" limit="{limit}" namespace="{tag}"',
              f"CATEGORY S_{i:02d} gpu",
              f"JOB F_{i:02d} {directory}/cpu.sub",
              f'VARS F_{i:02d} dataset="{dataset}" namespace="{tag}"',
              f"PARENT S_{i:02d} CHILD F_{i:02d}"]
lines += ["MAXJOBS gpu 50", "MAXJOBS cpu 10"]
(directory / "analysis.dag").write_text("\n".join(lines) + "\n")
print(directory / "analysis.dag")
