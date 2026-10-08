#!/usr/bin/env python
import argparse
from pathlib import Path
ROOT=Path("/data/users/akolchina/epistasis_proteingym"); HERE=ROOT/"scripts/reanalysis/vespa_family"; BASE=ROOT/"results/reanalysis/tsuboyama/vespa_family"
p=argparse.ArgumentParser(); p.add_argument("--pilot",action="store_true"); a=p.parse_args()
datasets=[x.stem for x in sorted((ROOT/"results/tables/intermediate/tsuboyama_epistatic").glob("*.csv"))]
if a.pilot: datasets=["PIN1_HUMAN_Tsuboyama_2023_1I6C"]
tag="vespag_pilot" if a.pilot else "vespag_full"; d=BASE/"condor"/tag; logs=d/"logs"; logs.mkdir(parents=True,exist_ok=True)
common=f"""universe = vanilla
executable = {HERE}/run_vespag.sh
initialdir = {ROOT}
should_transfer_files = NO
getenv = False
log = {logs}/events.log
"""
(d/"gpu.sub").write_text(common+f"""arguments = score $(dataset) $(limit) $(namespace)
request_cpus = 2
request_gpus = 1
request_memory = 36GB
request_disk = 8GB
+MaxRuntime = 259200
output = {logs}/gpu.$(ClusterId).out
error = {logs}/gpu.$(ClusterId).err
queue
""")
(d/"cpu.sub").write_text(common+f"""arguments = fit $(dataset) none $(namespace)
request_cpus = 1
request_memory = 5GB
output = {logs}/cpu.$(ClusterId).out
error = {logs}/cpu.$(ClusterId).err
queue
""")
limit="8" if a.pilot else "none"; lines=[]
for i,ds in enumerate(datasets):
    lines += [f"JOB S_{i:02d} {d}/gpu.sub",f'VARS S_{i:02d} dataset="{ds}" limit="{limit}" namespace="{tag}"',f"CATEGORY S_{i:02d} gpu",f"JOB F_{i:02d} {d}/cpu.sub",f'VARS F_{i:02d} dataset="{ds}" namespace="{tag}"',f"PARENT S_{i:02d} CHILD F_{i:02d}"]
lines += ["MAXJOBS gpu 50","MAXJOBS cpu 10"]; (d/"analysis.dag").write_text("\n".join(lines)+"\n"); print(d/"analysis.dag")
