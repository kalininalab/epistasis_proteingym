from pathlib import Path

# Absolute path to the repo root (parent of the "src" folder)
repo_root = Path(__file__).resolve().parent.parent

# Commonly used folders
data_dir = repo_root / "data"
external_dir = repo_root / "external"
results_dir = repo_root / "results"
tables_dir = results_dir / "tables"
figures_dir = results_dir / "figures"

# Common subfolders
somermeyer_data_dir = data_dir / "somermeyer"
tsuboyama_data_dir = data_dir / "tsuboyama"

final_tables_dir = tables_dir / "final"
interm_tables_dir = tables_dir / "intermediate"

main_figures_dir = figures_dir / "main"
supp_figures_dir = figures_dir / "supplementary"

# Create all of them if they don't exist
for folder in [
    data_dir, external_dir, results_dir, figures_dir,
    somermeyer_data_dir, tsuboyama_data_dir,
    final_tables_dir, interm_tables_dir,
    main_figures_dir, supp_figures_dir
]:folder.mkdir(parents=True, exist_ok=True)