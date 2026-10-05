#!/usr/bin/env python
"""Prepare pilot or full HTCondor DAG, capped at ten concurrent GPUs."""
import argparse
from pathlib import Path
import pandas as pd
ROOT=Path('/data/users/akolchina/epistasis_proteingym'); HERE=ROOT/'scripts/reanalysis/multimodel_mm_pll'; BASE=ROOT/'results/reanalysis/tsuboyama/multimodel_mm_pll'
def main():
    p=argparse.ArgumentParser(); p.add_argument('--pilot',action='store_true'); a=p.parse_args(); models=pd.read_csv(HERE/'models.csv'); models=models[models.status.isin(['ready_to_submit','running_full'])]
    datasets=[p.stem for p in sorted((ROOT/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv'))]
    if a.pilot: datasets=['PIN1_HUMAN_Tsuboyama_2023_1I6C']
    tag='pilot' if a.pilot else 'full'; condor=BASE/'condor'/tag; logs=condor/'logs'; logs.mkdir(parents=True,exist_ok=True)
    common=f'''universe = vanilla\nexecutable = {HERE}/run.sh\ninitialdir = {ROOT}\nshould_transfer_files = NO\ngetenv = False\nlog = {logs}/events.log\n'''
    (condor/'score.sub').write_text(common+f'''arguments = score $(key) $(checkpoint) $(revision) $(default) $(dataset) $(max_doubles) $(namespace)\nrequest_cpus = 2\nrequest_gpus = 1\nrequest_memory = 16GB\nrequest_disk = 8GB\n+MaxRuntime = 172800\noutput = {logs}/score.$(ClusterId).$(ProcId).out\nerror = {logs}/score.$(ClusterId).$(ProcId).err\nqueue\n''')
    (condor/'cpu.sub').write_text(common+f'''arguments = $(arguments)\nrequest_cpus = 1\nrequest_memory = 5GB\nrequest_disk = 2GB\n+MaxRuntime = 86400\noutput = {logs}/cpu.$(ClusterId).$(ProcId).out\nerror = {logs}/cpu.$(ClusterId).$(ProcId).err\nqueue\n''')
    dag=[]; scores={}; max_doubles='8' if a.pilot else 'none'
    for di,dataset in enumerate(datasets):
        for mi,row in enumerate(models.itertuples(index=False)):
            name=f'S_{di:02d}_{mi:02d}'; scores[(dataset,row.model_key)]=name; dag += [f'JOB {name} {condor}/score.sub',f'VARS {name} key="{row.model_key}" checkpoint="{row.checkpoint}" revision="{row.revision}" default="{row.default_column}" dataset="{dataset}" max_doubles="{max_doubles}" namespace="{tag}"',f'CATEGORY {name} gpu']
        # Fit the two standalone representatives.
        for key in ['ESM1b','ESM2_150M']:
            name=f'F_{key}_{di:02d}'; dag += [f'JOB {name} {condor}/cpu.sub',f'VARS {name} arguments="fit {key} {dataset} {tag}"',f'CATEGORY {name} cpu',f'PARENT {scores[(dataset,key)]} CHILD {name}']
        # Average five ESM1v checkpoints, then fit the ensemble only.
        agg=f'A_ESM1v_{di:02d}'; fit=f'F_ESM1v_{di:02d}'; parents=' '.join(scores[(dataset,f'ESM1v_{i}')] for i in range(1,6))
        dag += [f'JOB {agg} {condor}/cpu.sub',f'VARS {agg} arguments="aggregate {dataset} {tag}"',f'CATEGORY {agg} cpu',f'PARENT {parents} CHILD {agg}',f'JOB {fit} {condor}/cpu.sub',f'VARS {fit} arguments="fit ESM1v_ensemble {dataset} {tag}"',f'CATEGORY {fit} cpu',f'PARENT {agg} CHILD {fit}']
    dag += ['MAXJOBS gpu 10','MAXJOBS cpu 10']; (condor/'analysis.dag').write_text('\n'.join(dag)+'\n')
    print(f'{tag}: {len(models)*len(datasets)} GPU score jobs, {4*len(datasets)} dependent CPU nodes; max 10 GPUs'); print(condor/'analysis.dag')
if __name__=='__main__': main()
