#!/usr/bin/env python
import argparse
from pathlib import Path
R=Path('/data/users/akolchina/epistasis_proteingym');H=R/'scripts/reanalysis/protssn_mm_pll';B=R/'results/reanalysis/tsuboyama/protssn_mm_pll'
p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();datasets=[x.stem for x in sorted((R/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv'))];datasets=['PIN1_HUMAN_Tsuboyama_2023_1I6C'] if a.pilot else datasets;tag='pilot' if a.pilot else 'full';d=B/'condor'/tag;l=d/'logs';l.mkdir(parents=True,exist_ok=True)
common=f'''universe = vanilla
executable = {H}/run.sh
initialdir = {R}
should_transfer_files = NO
getenv = False
log = {l}/events.log
'''
(d/'gpu.sub').write_text(common+f'''arguments = score $(dataset) $(namespace) $(limit)
request_cpus = 2
request_gpus = 1
request_memory = 32GB
request_disk = 12GB
requirements = (Gpus_GlobalMemoryMb >= 16000)
output = {l}/gpu.$(ClusterId).$(Process).out
error = {l}/gpu.$(ClusterId).$(Process).err
queue
''')
(d/'cpu.sub').write_text(common+f'''arguments = fit $(dataset) $(namespace)
request_cpus = 1
request_memory = 5GB
output = {l}/cpu.$(ClusterId).$(Process).out
error = {l}/cpu.$(ClusterId).$(Process).err
queue
''')
limit='8' if a.pilot else 'none';lines=[]
for i,ds in enumerate(datasets):
 lines += [f'JOB S{i} {d}/gpu.sub',f'VARS S{i} dataset="{ds}" namespace="{tag}" limit="{limit}"',f'CATEGORY S{i} gpu',f'JOB F{i} {d}/cpu.sub',f'VARS F{i} dataset="{ds}" namespace="{tag}"',f'CATEGORY F{i} cpu',f'PARENT S{i} CHILD F{i}']
lines += ['MAXJOBS gpu 10','MAXJOBS cpu 5'];(d/'analysis.dag').write_text('\n'.join(lines)+'\n');print(d/'analysis.dag')
