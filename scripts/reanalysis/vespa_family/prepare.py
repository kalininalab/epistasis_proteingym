#!/usr/bin/env python
import argparse
from pathlib import Path
ROOT=Path('/data/users/akolchina/epistasis_proteingym'); HERE=ROOT/'scripts/reanalysis/vespa_family'; BASE=ROOT/'results/reanalysis/tsuboyama/vespa_family'
p=argparse.ArgumentParser(); p.add_argument('--pilot',action='store_true'); a=p.parse_args()
datasets=[x.stem for x in sorted((ROOT/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv'))]
if a.pilot: datasets=['PIN1_HUMAN_Tsuboyama_2023_1I6C']
tag='pilot' if a.pilot else 'full'; d=BASE/'condor'/tag; logs=d/'logs'; logs.mkdir(parents=True,exist_ok=True)
common=f'''universe = vanilla\nexecutable = {HERE}/run.sh\ninitialdir = {ROOT}\nshould_transfer_files = NO\ngetenv = False\nlog = {logs}/events.log\n'''
(d/'gpu.sub').write_text(common+f'''arguments = score $(dataset) $(limit) $(namespace)\nrequest_cpus = 2\nrequest_gpus = 1\nrequest_memory = 28GB\nrequest_disk = 8GB\n+MaxRuntime = 259200\noutput = {logs}/gpu.$(ClusterId).out\nerror = {logs}/gpu.$(ClusterId).err\nqueue\n''')
(d/'cpu.sub').write_text(common+f'''arguments = fit $(dataset) none $(namespace)\nrequest_cpus = 1\nrequest_memory = 5GB\noutput = {logs}/cpu.$(ClusterId).out\nerror = {logs}/cpu.$(ClusterId).err\nqueue\n''')
limit='8' if a.pilot else 'none'; lines=[]
for i,ds in enumerate(datasets):
    lines += [f'JOB S_{i:02d} {d}/gpu.sub',f'VARS S_{i:02d} dataset="{ds}" limit="{limit}" namespace="{tag}"',f'CATEGORY S_{i:02d} gpu',f'JOB F_{i:02d} {d}/cpu.sub',f'VARS F_{i:02d} dataset="{ds}" namespace="{tag}"',f'PARENT S_{i:02d} CHILD F_{i:02d}']
lines += ['MAXJOBS gpu 50','MAXJOBS cpu 10']; (d/'analysis.dag').write_text('\n'.join(lines)+'\n'); print(d/'analysis.dag')
