from pathlib import Path
ROOT=Path('/data/users/akolchina/epistasis_proteingym'); HERE=ROOT/'scripts/reanalysis/conditional_mm'
BASE=ROOT/'results/reanalysis/tsuboyama/conditional_mm_esm2_650m'; LOG=BASE/'condor/logs'; LOG.mkdir(parents=True,exist_ok=True)
datasets=[p.stem for p in sorted((ROOT/'results/tables/intermediate/tsuboyama_epistatic').glob('*.csv'))]
(BASE/'condor/datasets.txt').write_text('\n'.join(datasets)+'\n')
common=f'''universe = vanilla\ninitialdir = {ROOT}\nshould_transfer_files = NO\ngetenv = False\nlog = {LOG}/events.log\n'''
(BASE/'condor/score.sub').write_text(common+f'''executable = {HERE}/run.sh\narguments = score $(Process) 10\nrequest_cpus = 2\nrequest_gpus = 1\nrequest_memory = 12GB\nrequest_disk = 5GB\noutput = {LOG}/score.$(ClusterId).$(Process).out\nerror = {LOG}/score.$(ClusterId).$(Process).err\nqueue 10\n''')
(BASE/'condor/fit.sub').write_text(common+f'''executable = {HERE}/run.sh\narguments = fit $(dataset)\nrequest_cpus = 1\nrequest_memory = 4GB\nrequest_disk = 1GB\noutput = {LOG}/fit.$(ClusterId).$(Process).out\nerror = {LOG}/fit.$(ClusterId).$(Process).err\nqueue dataset from {BASE}/condor/datasets.txt\n''')
(BASE/'condor/analysis.dag').write_text(f'''JOB SCORE {BASE}/condor/score.sub\nJOB FIT {BASE}/condor/fit.sub\nPARENT SCORE CHILD FIT\n''')
print(BASE/'condor/analysis.dag')
