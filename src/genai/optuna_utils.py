"""Optuna helpers: persistent SQLite studies, summary json/csv and plots for the report."""
from __future__ import annotations

import optuna

from .common import OUT_DIR, SEED, STUDY_DIR, mkdirs, save_json


def make_study(name: str, direction: str = "minimize", resume: bool = True) -> optuna.Study:
    mkdirs(STUDY_DIR)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=name, storage=f"sqlite:///{STUDY_DIR / (name + '.db')}", load_if_exists=resume,
        direction=direction, sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))


def run_study(study: optuna.Study, objective, n_trials: int, search_space: dict, extra: dict | None = None):
    """Runs only the missing trials (resumable), then writes summary + csv + figures."""
    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    remaining = max(0, n_trials - done)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    return summarize(study, search_space, extra)


def summarize(study: optuna.Study, search_space: dict, extra: dict | None = None) -> dict:
    out_dir = OUT_DIR / "optuna"
    mkdirs(out_dir)
    states = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    summary = {"study": study.study_name, "search_space": search_space, "n_trials": len(study.trials),
               "trial_states": states, "best_value": study.best_value, "best_params": study.best_params,
               "best_trial": study.best_trial.number, **(extra or {})}
    save_json(summary, out_dir / f"{study.study_name}_summary.json")
    study.trials_dataframe().to_csv(out_dir / f"{study.study_name}_trials.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(out_dir / f"{study.study_name}_history.png", dpi=150)
        plt.close("all")
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 4:
            plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(out_dir / f"{study.study_name}_importance.png", dpi=150)
            plt.close("all")
    except Exception as e:  # plots are nice-to-have
        print(f"[optuna] plot skipped: {e}")
    return summary
