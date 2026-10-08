#!/usr/bin/env python
import argparse
from pathlib import Path
R=Path('/data/users/akolchina/epistasis_proteingym');H=R/'scripts/reanalysis/structure_plm_mm_pll';B=R/'results/reanalysis/tsuboyama/structure_plm_mm_pll';p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();ds=[x.stem for x in sorted((R/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv'))];ds=['PIN1_HUMAN_Tsuboyama_2023_1I6C'] if a.pilot else ds;tag='pilot' if a.pilot else 'full';d=B/'condor'/tag;l=d/'logs';l.mkdir(parents=True,exist_ok=True);common=f'''universe = vanilla
executable = {H}/run.sh
initialdir = {R}
should_transfer_files = NO
getenv = False
log = {l}/events.log
''';(d/'gpu.sub').write_text(common+f'''arguments = score $(dataset) $(model) $(namespace) $(limit)
request_cpus = 2
request_gpus = 1
request_memory = $(memory)
request_disk = 12GB
output = {l}/gpu.$(ClusterId).$(Process).out
error = {l}/gpu.$(ClusterId).$(Process).err
queue
''');(d/'cpu.sub').write_text(common+f'''arguments = fit $(dataset) $(model) $(namespace)
request_cpus = 1
request_memory = 5GB
output = {l}/cpu.$(ClusterId).$(Process).out
error = {l}/cpu.$(ClusterId).$(Process).err
queue
''');lines=[];limit='8' if a.pilot else 'none'
for mi,m in enumerate(('MIF','MIFST','ProSST-1024','SaProt_35M_AF2')):
 mem='24GB' if m in ('MIFST','ProSST-1024') else '16GB'
 for i,x in enumerate(ds):
  k=f'{mi}_{i}';lines += [f'JOB S{k} {d}/gpu.sub',f'VARS S{k} dataset="{x}" model="{m}" namespace="{tag}" limit="{limit}" memory="{mem}"',f'CATEGORY S{k} gpu',f'JOB F{k} {d}/cpu.sub',f'VARS F{k} dataset="{x}" model="{m}" namespace="{tag}"',f'CATEGORY F{k} cpu',f'PARENT S{k} CHILD F{k}']
lines += ['MAXJOBS gpu 50','MAXJOBS cpu 10'];(d/'analysis.dag').write_text('\n'.join(lines)+'\n');print(d/'analysis.dag')
