from pathlib import Path
ROOT=Path('/data/users/akolchina/epistasis_proteingym')
HERE=ROOT/'scripts/reanalysis/pll_esm2_650m'; BASE=ROOT/'results/reanalysis/tsuboyama/pll_esm2_650m'; CONDOR=BASE/'condor'; LOG=CONDOR/'logs'
LOG.mkdir(parents=True,exist_ok=True)
datasets=[p.stem for p in sorted((ROOT/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv'))]
(CONDOR/'datasets.txt').write_text('\n'.join(datasets)+'\n')
common=f'''universe = vanilla\nexecutable = {HERE}/run_full.sh\ninitialdir = {ROOT}\nshould_transfer_files = NO\ngetenv = False\nlog = {LOG}/events.log\n'''
(CONDOR/'score.sub').write_text(common+f'''arguments = score $(dataset)\nrequest_cpus = 2\nrequest_gpus = 1\nrequest_memory = 12GB\nrequest_disk = 5GB\noutput = {LOG}/score.$(ClusterId).out\nerror = {LOG}/score.$(ClusterId).err\nqueue\n''')
(CONDOR/'fit.sub').write_text(common+f'''arguments = fit $(dataset)\nrequest_cpus = 1\nrequest_memory = 4GB\nrequest_disk = 1GB\noutput = {LOG}/fit.$(ClusterId).out\nerror = {LOG}/fit.$(ClusterId).err\nqueue\n''')
dag=[]
for i,dataset in enumerate(datasets):
    score=f'SCORE_{i:02d}'; fit=f'FIT_{i:02d}'
    dag += [f'JOB {score} {CONDOR}/score.sub',f'VARS {score} dataset="{dataset}"',f'CATEGORY {score} pll_gpu',
            f'JOB {fit} {CONDOR}/fit.sub',f'VARS {fit} dataset="{dataset}"',f'CATEGORY {fit} pll_cpu',f'PARENT {score} CHILD {fit}']
dag += ['MAXJOBS pll_gpu 10','MAXJOBS pll_cpu 10']
(CONDOR/'analysis.dag').write_text('\n'.join(dag)+'\n')
print(f'Prepared {len(datasets)} GPU scoring jobs (max 10 concurrent) followed by {len(datasets)} CPU fits (max 10 concurrent).')
print(CONDOR/'analysis.dag')
