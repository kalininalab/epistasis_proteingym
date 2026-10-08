#!/usr/bin/env python
import argparse
from pathlib import Path
R=Path('/data/users/akolchina/epistasis_proteingym');H=R/'scripts/reanalysis/carp_mm_pll';B=R/'results/reanalysis/tsuboyama/carp_mm_pll';p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();ds=[x.stem for x in sorted((R/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv'))];ds=['PIN1_HUMAN_Tsuboyama_2023_1I6C'] if a.pilot else ds;tag='pilot' if a.pilot else 'full';d=B/'condor'/tag;l=d/'logs';l.mkdir(parents=True,exist_ok=True);common=f'''universe = vanilla
executable = {H}/run.sh
initialdir = {R}
should_transfer_files = NO
getenv = False
log = {l}/events.log
''';(d/'gpu.sub').write_text(common+f'''arguments = score $(dataset) $(namespace) $(limit)
request_cpus = 2
request_gpus = 1
request_memory = 16GB
request_disk = 8GB
output = {l}/gpu.$(ClusterId).out
error = {l}/gpu.$(ClusterId).err
queue
''');(d/'cpu.sub').write_text(common+f'''arguments = fit $(dataset) $(namespace)
request_cpus = 1
request_memory = 5GB
output = {l}/cpu.$(ClusterId).out
error = {l}/cpu.$(ClusterId).err
queue
''');lines=[];limit='8' if a.pilot else 'none'
for i,x in enumerate(ds):lines += [f'JOB S{i} {d}/gpu.sub',f'VARS S{i} dataset="{x}" namespace="{tag}" limit="{limit}"',f'CATEGORY S{i} gpu',f'JOB F{i} {d}/cpu.sub',f'VARS F{i} dataset="{x}" namespace="{tag}"',f'CATEGORY F{i} cpu',f'PARENT S{i} CHILD F{i}']
lines += ['MAXJOBS gpu 50','MAXJOBS cpu 10'];(d/'analysis.dag').write_text('\n'.join(lines)+'\n');print(d/'analysis.dag')
