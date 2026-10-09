#!/usr/bin/env python
import argparse
from pathlib import Path
R=Path('/data/users/akolchina/epistasis_proteingym');H=R/'scripts/reanalysis/next_sitewise_mm_pll';B=R/'results/reanalysis/tsuboyama/next_sitewise_mm_pll'
p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();datasets=[x.stem for x in sorted((R/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv'))];datasets=['PIN1_HUMAN_Tsuboyama_2023_1I6C'] if a.pilot else datasets;tag='pilot' if a.pilot else 'full';d=B/'condor'/tag;l=d/'logs';l.mkdir(parents=True,exist_ok=True)
common=f'''universe = vanilla
executable = {H}/run.sh
initialdir = {R}
should_transfer_files = NO
getenv = False
log = {l}/events.log
'''
(d/'gpu.sub').write_text(common+f'''arguments = score $(dataset) $(model) $(namespace) $(limit)
request_cpus = 2
request_gpus = 1
request_memory = $(memory)
request_disk = 60GB
requirements = (Gpus_GlobalMemoryMb >= $(gpu_memory))
output = {l}/gpu.$(ClusterId).$(Process).out
error = {l}/gpu.$(ClusterId).$(Process).err
queue
''')
(d/'cpu.sub').write_text(common+f'''arguments = fit $(dataset) $(model) $(namespace)
request_cpus = 1
request_memory = 5GB
output = {l}/cpu.$(ClusterId).$(Process).out
error = {l}/cpu.$(ClusterId).$(Process).err
queue
''')
lines=[];limit='8' if a.pilot else 'none'
for mi,(model,mem,gmem) in enumerate((('MSA_Transformer_ensemble','32GB',10000),('xTrimoPGLM-10B-MLM','80GB',30000),('MULAN_small','16GB',10000))):
 for i,ds in enumerate(datasets):
  k=f'{mi}_{i}';lines += [f'JOB S{k} {d}/gpu.sub',f'VARS S{k} dataset="{ds}" model="{model}" namespace="{tag}" limit="{limit}" memory="{mem}" gpu_memory="{gmem}"',f'CATEGORY S{k} gpu',f'JOB F{k} {d}/cpu.sub',f'VARS F{k} dataset="{ds}" model="{model}" namespace="{tag}"',f'CATEGORY F{k} cpu',f'PARENT S{k} CHILD F{k}']
lines += ['MAXJOBS gpu 10','MAXJOBS cpu 5'];(d/'analysis.dag').write_text('\n'.join(lines)+'\n');print(d/'analysis.dag')
