# app/dashboard.py

import os
import io
import mlflow
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="📊 Dashboard General de Evaluación", layout="wide")
st.title("📈 Evaluación Completa del Chatbot por Pregunta")

# =========================
# Cargar experimentos eval_
# =========================
client = mlflow.tracking.MlflowClient()
experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

if not experiments:
    st.warning("No se encontraron experimentos de evaluación.")
    st.stop()

exp_names = [exp.name for exp in experiments]
selected_exp_name = st.selectbox("Selecciona un experimento para visualizar:", exp_names)

experiment = client.get_experiment_by_name(selected_exp_name)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"]
)

if not runs:
    st.warning("No hay ejecuciones registradas en este experimento.")
    st.stop()

# =========================
# Armar DataFrame de runs
# =========================
records = []
all_criteria_cols = set()

for idx, run in enumerate(runs, start=1):
    params = run.data.params
    metrics = run.data.metrics

    row = {
        "run_id": run.info.run_id,
        "idx": idx,  # posición ordenada por fecha (1 = más reciente)
        "pregunta": params.get("question"),
        "prompt_version": params.get("prompt_version"),
        "chunk_size": int(params.get("chunk_size", 0)),
        "chunk_overlap": int(params.get("chunk_overlap", 0)),
        "lc_is_correct": metrics.get("lc_is_correct", None),
        "start_time": run.info.start_time
    }

    # Capturar todas las métricas *_score (criterios)
    for k, v in metrics.items():
        if k.endswith("_score"):
            row[k] = v
            all_criteria_cols.add(k)

    records.append(row)

df = pd.DataFrame(records)

# =========================
# UI: Filtros básicos
# =========================
with st.sidebar:
    st.header("🔎 Filtros")
    sel_prompt = st.multiselect(
        "Prompt version:",
        sorted(df["prompt_version"].dropna().unique().tolist()),
        default=sorted(df["prompt_version"].dropna().unique().tolist())
    )
    sel_chunk = st.multiselect(
        "Chunk size:",
        sorted(df["chunk_size"].dropna().unique().tolist()),
        default=sorted(df["chunk_size"].dropna().unique().tolist())
    )

df_f = df[(df["prompt_version"].isin(sel_prompt)) & (df["chunk_size"].isin(sel_chunk))].copy()

# =========================
# Tabla base
# =========================
st.subheader("📋 Resultados individuales por pregunta")
base_cols = ["idx", "run_id", "pregunta", "prompt_version", "chunk_size", "chunk_overlap", "lc_is_correct"]
crit_cols_sorted = sorted(all_criteria_cols)  # e.g., correctness_score, relevance_score, ...
show_cols = base_cols + crit_cols_sorted
st.dataframe(df_f[show_cols], use_container_width=True, height=400)

# =========================
# Agrupaciones/Resumen
# =========================
st.subheader("📊 Desempeño agrupado por configuración")
grouped = (df_f
           .groupby(["prompt_version", "chunk_size"], dropna=False)
           .agg(
               promedio_correcto=("lc_is_correct", "mean"),
               preguntas=("pregunta", "count"),
               **{f"mean_{c}": (c, "mean") for c in crit_cols_sorted}
           )
           .reset_index())

st.dataframe(grouped, use_container_width=True)

# Barra para lc_is_correct promedio
grouped["config"] = grouped["prompt_version"] + " | " + grouped["chunk_size"].astype(str)
st.caption("🔹 Promedio de lc_is_correct por configuración")
st.bar_chart(grouped.set_index("config")["promedio_correcto"])

# =========================
# Comparador de criterios
# =========================
st.subheader("🧪 Comparador de criterios")

if crit_cols_sorted:
    sel_criteria = st.multiselect(
        "Criterios a comparar:",
        crit_cols_sorted,
        default=crit_cols_sorted[: min(3, len(crit_cols_sorted))]
    )

    if sel_criteria:
        # Long/melt para Altair
        long_df = df_f.melt(
            id_vars=["idx", "run_id", "prompt_version", "chunk_size"],
            value_vars=sel_criteria,
            var_name="criterion",
            value_name="score"
        ).dropna(subset=["score"])

        # Selector de eje X
        x_axis_choice = st.radio(
            "Eje X para comparar:",
            ["idx (orden temporal)", "chunk_size", "prompt_version"],
            horizontal=True
        )
        x_field = {"idx (orden temporal)": "idx", "chunk_size": "chunk_size", "prompt_version": "prompt_version"}[x_axis_choice]

        # Chart 1: líneas/columnas por criterio
        st.caption("🔹 Evolución/Comparación por criterio")
        if x_field in ("idx", "chunk_size"):
            chart = (alt.Chart(long_df)
                     .mark_line(point=True)
                     .encode(
                         x=alt.X(f"{x_field}:O", title=x_field),
                         y=alt.Y("mean(score):Q", title="score (promedio)"),
                         color=alt.Color("criterion:N", title="criterio"),
                         tooltip=["criterion", f"{x_field}", alt.Tooltip("mean(score):Q", format=".3f"),
                                  "prompt_version", "chunk_size"]
                     )
                     .properties(height=320))
        else:
            chart = (alt.Chart(long_df)
                     .mark_bar()
                     .encode(
                         x=alt.X("prompt_version:N", title="prompt_version"),
                         y=alt.Y("mean(score):Q", title="score (promedio)"),
                         color=alt.Color("criterion:N", title="criterio"),
                         column=alt.Column("chunk_size:N", title="chunk_size"),
                         tooltip=["criterion", "prompt_version", "chunk_size", alt.Tooltip("mean(score):Q", format=".3f")]
                     )
                     .properties(height=320))

        st.altair_chart(chart, use_container_width=True)

        # Chart 2: violín/boxplot por criterio (distribución)
        st.caption("🔹 Distribución por criterio (boxplot)")
        box = (alt.Chart(long_df)
               .mark_boxplot(extent="min-max")
               .encode(
                   x=alt.X("criterion:N", title="criterio"),
                   y=alt.Y("score:Q", title="score"),
                   color="criterion:N",
                   tooltip=["criterion", alt.Tooltip("score:Q", format=".3f")]
               )
               .properties(height=280))
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("Selecciona al menos un criterio para comparar.")
else:
    st.info("No se encontraron métricas por criterio ( *_score ) en este experimento.")

# =========================
# (Opcional) Razonamientos
# =========================
st.subheader("🧠 Razonamientos del evaluador (artefactos)")

# Elegir un run para inspección
run_id_options = df_f["run_id"].tolist()
sel_run = st.selectbox("Selecciona un run para ver artefactos de razonamiento:", run_id_options)

if sel_run:
    cols = st.columns(2)
    with cols[0]:
        st.caption("🗂️ Archivos de razonamiento por criterio (carpeta: reasoning/)")
    with cols[1]:
        show_samples = st.checkbox("Mostrar también question/prediction/reference (carpeta: sample/)", value=True)

    # Utilidad para leer artefactos como texto
    def read_artifact_text(_run_id: str, path: str) -> str:
        try:
            # Descarga a memoria usando un buffer (cuando es server local funciona),
            # si no, descarga a un tmp dir y léelo.
            tmpdir = st.session_state.get("tmp_artifacts_dir")
            if not tmpdir:
                tmpdir = os.path.join(os.getcwd(), ".mlflow_tmp")
                os.makedirs(tmpdir, exist_ok=True)
                st.session_state["tmp_artifacts_dir"] = tmpdir
            local_path = client.download_artifacts(_run_id, path, tmpdir)
            if os.path.isdir(local_path):
                # Si es un directorio, listamos
                return "\n".join(sorted(os.listdir(local_path)))
            else:
                with io.open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception as e:
            return f"(No se pudo leer el artefacto: {path}). Error: {e}"

    # Listar archivos en reasoning/
    try:
        items = client.list_artifacts(sel_run, path="reasoning")
    except Exception:
        items = []

    if not items:
        st.info("No se encontraron artefactos en reasoning/. ¿Registraste mlflow.log_text(..., artifact_file='reasoning/<criterio>.txt')?")
    else:
        for it in items:
            if not it.is_dir:
                with st.expander(f"📄 {it.path}"):
                    content = read_artifact_text(sel_run, it.path)
                    st.code(content, language="markdown")

    # (Opcional) sample/ question/prediction/reference
    if show_samples:
        try:
            sample_items = client.list_artifacts(sel_run, path="sample")
        except Exception:
            sample_items = []
        if sample_items:
            st.caption("📎 Artefactos de muestra (sample/)")
            for it in sample_items:
                if not it.is_dir:
                    with st.expander(f"📄 {it.path}"):
                        st.code(read_artifact_text(sel_run, it.path))
        else:
            st.caption("No se encontraron artefactos en sample/")

st.success("Listo. Ajusta filtros y compara criterios para tus ejecuciones ✅")
