# tools/summarize_mlflow.py
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

PRIMARY_METRIC = "correctness_score"  # métrica principal
PENALIZE = ["toxicity_score", "harmfulness_score"]  # penalizaciones
SECONDARY = ["relevance_score", "coherence_score"]  # desempates
ALIAS_OK = ["lc_is_correct"]  # por si usas también QAEvalChain como referencia

def load_runs_df(exp_prefix="eval_"):
    client = MlflowClient()
    exps = [e for e in client.search_experiments() if e.name.startswith(exp_prefix)]
    if not exps:
        raise SystemExit("No hay experimentos eval_")
    # Toma el último experimento por fecha de creación (o elige por nombre exacto)
    exp = sorted(exps, key=lambda e: e.creation_time, reverse=True)[0]
    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
    rows = []
    for r in runs:
        params = r.data.params
        metrics = r.data.metrics
        row = {
            "run_id": r.info.run_id,
            "prompt_version": params.get("prompt_version"),
            "chunk_size": int(params.get("chunk_size", 0) or 0),
            "chunk_overlap": int(params.get("chunk_overlap", 0) or 0),
        }
        for k, v in metrics.items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)

def rank_configs(df: pd.DataFrame):
    # Agrupa por config y calcula promedios
    metric_cols = [c for c in df.columns if c.endswith("_score")] + ALIAS_OK
    g = (df.groupby(["prompt_version","chunk_size"], dropna=False)
            [metric_cols].mean().reset_index())
    # Construye un score compuesto:
    #  + PRIMARY_METRIC alto es mejor
    #  + SECONDARY suma ponderada suave
    #  - PENALIZE resta (penaliza)
    g["_composite"] = 0.0
    g["_composite"] += g.get(PRIMARY_METRIC, 0.0) * 1.0
    for m in SECONDARY:
        if m in g:
            g["_composite"] += g[m] * 0.25
    for m in PENALIZE:
        if m in g:
            g["_composite"] -= g[m] * 0.50
    g = g.sort_values(["_composite", PRIMARY_METRIC, "relevance_score", "coherence_score"],
                      ascending=[False, False, False, False])
    return g

if __name__ == "__main__":
    df = load_runs_df("eval_")
    ranked = rank_configs(df)
    print("\n=== Promedios por configuración (ordenado por score compuesto) ===")
    print(ranked.fillna(0).to_string(index=False))

    # Sugerencia de “mejor” config
    best = ranked.iloc[0].to_dict()
    print("\n=== Configuración sugerida ===")
    print(f"- prompt_version: {best['prompt_version']}")
    print(f"- chunk_size    : {int(best['chunk_size'])}")
    # Muestra principales métricas si están presentes
    for k in ["correctness_score","relevance_score","coherence_score","toxicity_score","harmfulness_score","lc_is_correct","_composite"]:
        if k in best:
            print(f"  {k}: {best[k]:.3f}")
