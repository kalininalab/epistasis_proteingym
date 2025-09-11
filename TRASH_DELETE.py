from src.analysis import compute_result_table, select_best_models, build_tsuboyama_N_series
from src.paths import data_dir, interm_tables_dir, main_figures_dir, supp_figures_dir
from src.models.linear_regression import linear_regression
from src.models.mlp import mlp
from src.plotting import combined_plot_tsuboyama


results, counts = compute_result_table(data_dir, interm_tables_dir / "tsuboyama_epistatic", "tsuboyama")
counts.to_csv(interm_tables_dir / "counts.csv", index=True)

results = linear_regression(results, data_dir, interm_tables_dir / "tsuboyama_epistatic", "tsuboyama")

final_results = mlp(results, data_dir, interm_tables_dir / "tsuboyama_epistatic", "tsuboyama")
final_results.dropna(axis=1, how="all", inplace=True)
output_dir = interm_tables_dir / "models_evaluation" 
final_results.to_csv(output_dir / "tsuboyama_all_models.csv", index=True)

select_best_models(interm_tables_dir / "models_evaluation", "tsuboyama")

combined_plot_tsuboyama(interm_tables_dir, main_figures_dir)

build_tsuboyama_N_series(
    raw_tsuboyama_dir = interm_tables_dir / "tsuboyama_epistatic",
    model_root_dir    = data_dir,
    work_root_dir     = interm_tables_dir,
    figures_dir       = supp_figures_dir,
)
