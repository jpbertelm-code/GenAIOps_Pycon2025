# eval_run.py
import sys
import os
import json
import tempfile

import mlflow
from dotenv import load_dotenv

# Asegura imports relativos al proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.rag_pipeline import load_vectorstore_from_disk, build_chain  # noqa: E402

# LangChain
from langchain_openai import ChatOpenAI
from langchain.evaluation.qa import QAEvalChain
from langchain.evaluation.criteria import LabeledCriteriaEvalChain

# =========================
# Carga de entorno
# =========================
load_dotenv()

# =========================
# Configuración
# =========================
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1_asistente_rrhh")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATASET_PATH = os.getenv("DATASET_PATH", "tests/eval_dataset.json")

# Modelo LLM (puedes ajustar vía env: OPENAI_API_KEY, OPENAI_BASE_URL, etc.)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))

# =========================
# Criterios de evaluación
# =========================
# Built-in: correctness, relevance, coherence, harmfulness
# Custom: toxicity (definimos rúbrica)
CRITERIA = {
    "correctness": "¿La respuesta es factualmente correcta con respecto a la referencia?",
    "relevance": "¿La respuesta aborda la pregunta y usa información pertinente al input?",
    "coherence": "¿La respuesta es clara, bien estructurada y lógica?",
    "toxicity": "Evalúa presencia de lenguaje ofensivo, insultante, de odio o riesgoso.",
    "harmfulness": "¿La respuesta podría causar daño si se siguiera tal cual?",
}


# =========================
# Utilidades
# =========================
def eval_with_labeled_criteria(llm, input_text, prediction, reference, criteria_dict):
    """
    Ejecuta LabeledCriteriaEvalChain criterio por criterio para obtener
    métricas y razonamientos independientes.

    Retorna:
        dict { criterio: {"value": str, "score": float, "reasoning": str} }
    """
    results = {}
    for name, desc in criteria_dict.items():
        # Para built-in podemos pasar directamente el nombre (str).
        # Para custom (p. ej., "toxicity"), pasamos {name: descripcion}.
        crit_arg = name if name in {"correctness", "relevance", "coherence", "harmfulness"} else {name: desc}
        try:
            evaluator = LabeledCriteriaEvalChain.from_llm(llm=llm, criteria=crit_arg)
            graded = evaluator.evaluate_strings(
                input=input_text, prediction=prediction, reference=reference
            )
            results[name] = {
                "value": graded.get("value", "UNKNOWN"),
                "score": float(graded.get("score", 0) or 0),
                "reasoning": graded.get("reasoning", ""),
            }
        except Exception as e:
            # Si algo falla, registramos 0 y razón con el error para no romper el batch
            results[name] = {
                "value": "ERROR",
                "score": 0.0,
                "reasoning": f"Exception during evaluation: {repr(e)}",
            }
    return results


def log_reasoning_artifact(criterion: str, content: str):
    """
    Intenta usar mlflow.log_text; si no está disponible, usa archivo temporal con log_artifact.
    """
    try:
        # MLflow >= 2.9 tiene log_text
        mlflow.log_text(content, artifact_file=f"reasoning/{criterion}.txt")
    except Exception:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=f"_{criterion}.txt", encoding="utf-8") as tmpf:
            tmpf.write(content)
            tmp_path = tmpf.name
        mlflow.log_artifact(tmp_path, artifact_path="reasoning")


def main():
    # =========================
    # Dataset
    # =========================
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"No se encontró el dataset en {DATASET_PATH}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    # =========================
    # Vectorstore y cadena RAG
    # =========================
    vectordb = load_vectorstore_from_disk()
    chain = build_chain(vectordb, prompt_version=PROMPT_VERSION)

    # =========================
    # Evaluadores LLM
    # =========================
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=LLM_TEMPERATURE)
    qa_eval = QAEvalChain.from_llm(llm)

    # =========================
    # MLflow
    # =========================
    mlflow.set_experiment(f"eval_{PROMPT_VERSION}")
    print(f"📊 Experimento MLflow: eval_{PROMPT_VERSION}")

    # =========================
    # Loop de evaluación
    # =========================
    total = len(dataset)
    for i, pair in enumerate(dataset, start=1):
        pregunta = pair.get("question", "")
        respuesta_esperada = pair.get("answer", "")

        with mlflow.start_run(run_name=f"eval_q{i:03d}"):
            # 1) Inferencia del sistema
            result = chain.invoke({"question": pregunta, "chat_history": []})
            respuesta_generada = result.get("answer", "")

            # 2) Evaluación base (QAEvalChain)
            graded = qa_eval.evaluate_strings(
                input=pregunta,
                prediction=respuesta_generada,
                reference=respuesta_esperada
            )
            lc_verdict = graded.get("value", "UNKNOWN")
            lc_score = float(graded.get("score", 0) or 0)

            # 3) Evaluación por criterios (LabeledCriteriaEvalChain)
            criteria_results = eval_with_labeled_criteria(
                llm=llm,
                input_text=pregunta,
                prediction=respuesta_generada,
                reference=respuesta_esperada,
                criteria_dict=CRITERIA,
            )

            # ----- Logging -----
            # Params
            mlflow.log_param("prompt_version", PROMPT_VERSION)
            mlflow.log_param("chunk_size", CHUNK_SIZE)
            mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
            mlflow.log_param("openai_model", OPENAI_MODEL)

            # Guardar la pregunta y respuestas como artifacts para traza
            try:
                mlflow.log_text(pregunta, artifact_file="sample/question.txt")
                mlflow.log_text(respuesta_generada, artifact_file="sample/prediction.txt")
                mlflow.log_text(respuesta_esperada, artifact_file="sample/reference.txt")
            except Exception:
                # fallback
                pass

            # Métrica del eval clásico
            mlflow.log_metric("lc_is_correct", lc_score)

            # Métricas + reasoning por criterio
            for crit, data in criteria_results.items():
                mlflow.log_metric(f"{crit}_score", data["score"])

                reasoning_text = (
                    f"Pregunta:\n{pregunta}\n\n"
                    f"Predicción:\n{respuesta_generada}\n\n"
                    f"Referencia:\n{respuesta_esperada}\n\n"
                    f"Criterio: {crit}\n"
                    f"Value: {data['value']}\n"
                    f"Score: {data['score']}\n"
                    f"Reasoning:\n{data['reasoning']}\n"
                )
                log_reasoning_artifact(crit, reasoning_text)

            # Salida a consola
            print(f"\n[{i}/{total}] ✅ Pregunta:")
            print(pregunta)
            print(f"🧠 QAEvalChain → value={lc_verdict} · score={lc_score:.3f}")
            print("🧪 LabeledCriteriaEvalChain:")
            for crit, data in criteria_results.items():
                print(f"  - {crit:12s}: score={data['score']:.3f} · value={data['value']}")

    print("\n🏁 Evaluación finalizada.")


if __name__ == "__main__":
    main()
