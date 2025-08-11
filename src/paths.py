from pathlib import Path

# Absolute path to the repo root (parent of the "src" folder)
repo_root = Path(__file__).resolve().parent.parent

# Commonly used folders
data_dir = repo_root / "data"
external_dir = repo_root / "external"
results_dir = repo_root / "results"
tables_dir = results_dir / "tables"
figures_dir = results_dir / "figures"

# Create them if they don't exist
for folder in [data_dir, external_dir, results_dir, tables_dir, figures_dir]:
    folder.mkdir(parents=True, exist_ok=True)
