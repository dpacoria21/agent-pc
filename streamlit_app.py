"""Interactive Streamlit demo for the CP RAG prototype.

The app intentionally does not hardcode answers or scores. It provides a
curated, transparent demo path with three problems and suggested questions,
then runs the same local retriever + OpenAI/LangChain answer generation used
by the evaluation scripts.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from langchain_core.output_parsers import StrOutputParser


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_rag import (  # noqa: E402
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    LocalNodeRetriever,
    clean_text,
    format_context,
    make_llm,
    parse_judge_json,
)
from llm.env_config import DEFAULT_OPENAI_MODEL, resolve_openai_settings  # noqa: E402


PROCESSED = ROOT / "data" / "processed"
DEFAULT_TOP_K = 5


DEMO_PROBLEMS = {
    "codeforces_2218_B": {
        "label": "2218B - The 67th 6-7 Integer Problem",
        "why": "Caso corto y limpio para explicar observacion greedy/matematica.",
        "questions": [
            "What is the key observation for solving this problem?",
            "Explain the proof idea behind the formula 2 * max(a) - sum(a).",
            "What implementation mistakes should I avoid?",
        ],
    },
    "codeforces_2218_F": {
        "label": "2218F - The 67th Tree Problem",
        "why": "Buen ejemplo para hablar de paridad, construccion y prueba.",
        "questions": [
            "Why does the parity of the root subtree matter?",
            "Explain the constructive algorithm using pairs of vertices.",
            "What edge cases can make the construction impossible?",
        ],
    },
    "codeforces_2218_G": {
        "label": "2218G - The 67th Iteration of Counting is Fun",
        "why": "Ejemplo avanzado para mostrar observaciones editoriales y riesgos de conteo.",
        "questions": [
            "What condition makes the answer zero?",
            "How does the editorial count valid values for each a_i?",
            "What are the common mistakes with prefix counts and modulo?",
        ],
    },
}


REFERENCE_ANSWERS = {
    "codeforces_2218_B": {
        "What is the key observation for solving this problem?": (
            "La observacion clave es transformar la operacion: negar 6 de 7 numeros equivale a negar "
            "los 7 numeros y luego volver a negar exactamente uno. Si S es la suma original, despues de "
            "negar todos queda -S. Para maximizar la suma final, el numero que se recupera debe ser el "
            "maximo original, por eso el codigo usa 2 * max(a) - sum(a)."
        ),
        "Explain the proof idea behind the formula 2 * max(a) - sum(a).": (
            "Sea S la suma de los 7 numeros. Si se deja sin negar a_i, la suma final es "
            "a_i - (S - a_i) = 2*a_i - S. Como S es constante, maximizar la suma final equivale "
            "a elegir el mayor a_i. Por eso la respuesta es 2 * max(a) - sum(a)."
        ),
        "What implementation mistakes should I avoid?": (
            "Evita elegir el minimo original, confundir la operacion con negar un solo numero, o calcular "
            "sum(a) - 2*max(a). La forma directa y consistente con el editorial/codigo es "
            "2 * max(nums) - sum(nums)."
        ),
    },
    "codeforces_2218_F": {
        "Why does the parity of the root subtree matter?": (
            "La raiz tiene como subarbol todo el arbol, de tamano x+y. Por eso su paridad ya consume "
            "uno de los conteos requeridos: si x+y es par consume un nodo de subarbol par, y si es impar "
            "consume uno de subarbol impar. Luego se razona sobre los vertices restantes."
        ),
        "Explain the constructive algorithm using pairs of vertices.": (
            "Tras manejar la raiz, la construccion busca que los demas vertices tengan subarbol de tamano "
            "1 o 2. Cada par conectado como padre-hijo aporta un vertice con subarbol par y uno con "
            "subarbol impar. Los vertices restantes se conectan para aportar subarboles impares."
        ),
        "What edge cases can make the construction impossible?": (
            "Los casos imposibles aparecen cuando la paridad de la raiz exige un tipo de subarbol que no "
            "queda disponible, o cuando se necesitan mas subarboles pares que los que pueden sostenerse "
            "con vertices impares como hijos. La implementacion debe ajustar x/y despues de contar la raiz."
        ),
    },
    "codeforces_2218_G": {
        "What condition makes the answer zero?": (
            "Si existe una persona i con b_i > 0 que no tiene ningun vecino con b menor, entonces nunca "
            "puede sentarse en el momento indicado, porque necesita que algun vecino se haya sentado antes. "
            "En ese caso no hay arreglo a valido y la respuesta es 0."
        ),
        "How does the editorial count valid values for each a_i?": (
            "La editorial separa los casos segun que condicion se activa primero: suficientes personas ya "
            "sentadas o un vecino sentado antes. Con conteos prefix c_t se calcula cuantas opciones de a_i "
            "son compatibles con cada b_i, y luego se multiplican las opciones modulo 676767677."
        ),
        "What are the common mistakes with prefix counts and modulo?": (
            "Los errores comunes son usar c_t en vez de c_{t-1}, no distinguir el caso t-2, olvidar validar "
            "vecinos con menor b_i, y multiplicar sin aplicar modulo en cada paso."
        ),
    },
}


def parse_listish(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return [text]


@st.cache_data(show_spinner=False)
def load_problems() -> pd.DataFrame:
    path = PROCESSED / "cp_problems_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_units(index_source: str) -> pd.DataFrame:
    filename = "cp_pageindex_ready_chunks.csv" if index_source == "pageindex_chunks" else "cp_page_nodes_dataset.csv"
    path = PROCESSED / filename
    if not path.exists():
        return pd.DataFrame()
    units = pd.read_csv(path)
    if "global_problem_id" in units.columns:
        units["global_problem_id"] = units["global_problem_id"].fillna("").astype(str)
    return units


@st.cache_resource(show_spinner=False)
def build_retriever(index_source: str, scoped_problem_id: str | None) -> LocalNodeRetriever:
    units = load_units(index_source)
    if scoped_problem_id:
        units = units[units["global_problem_id"].eq(scoped_problem_id)].copy()
    return LocalNodeRetriever.from_nodes(units)


def problem_options(problems: pd.DataFrame) -> list[str]:
    demo_first = [pid for pid in DEMO_PROBLEMS if pid in set(problems.get("global_problem_id", []))]
    rest = [
        str(pid)
        for pid in problems.get("global_problem_id", pd.Series(dtype=str)).tolist()
        if str(pid) not in demo_first
    ]
    return demo_first + rest


def problem_label(problem_id: str, problems: pd.DataFrame) -> str:
    if problem_id in DEMO_PROBLEMS:
        return DEMO_PROBLEMS[problem_id]["label"]
    row = problems[problems["global_problem_id"].eq(problem_id)]
    if row.empty:
        return problem_id
    title = row.iloc[0].get("title", problem_id)
    return f"{problem_id} - {title}"


def build_hits_table(hits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in hits.iterrows():
        text = clean_text(row.get("node_text", ""))
        rows.append(
            {
                "rank": row.get("rank"),
                "score": round(float(row.get("retrieval_score", 0.0)), 4),
                "problem": row.get("global_problem_id", ""),
                "node_type": row.get("node_type", ""),
                "node_id": row.get("node_id", ""),
                "preview": text[:220],
            }
        )
    return pd.DataFrame(rows)


def answer_with_rag(
    *,
    query: str,
    retriever: LocalNodeRetriever,
    model: str,
    top_k: int,
    evaluate: bool,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    settings = resolve_openai_settings(requested_model=model)
    if not settings.available:
        raise RuntimeError("OPENAI_API_KEY no esta configurada en .env.")

    hits = retriever.retrieve(query, top_k=top_k)
    context = format_context(hits, max_chars=7000)
    llm = make_llm(settings)
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    start = time.perf_counter()
    answer = answer_chain.invoke({"query": query, "context": context})
    elapsed = time.perf_counter() - start

    metrics: dict[str, Any] = {
        "model": settings.model,
        "latency_seconds": round(elapsed, 3),
        "context_chars": len(context),
    }
    if evaluate:
        judge_chain = JUDGE_PROMPT | llm | StrOutputParser()
        for metric in ["Faithfulness", "Answer Relevancy"]:
            judge_text = judge_chain.invoke(
                {
                    "metric": metric,
                    "query": query,
                    "context": context,
                    "answer": answer,
                }
            )
            score, reason = parse_judge_json(judge_text)
            metrics[metric] = round(score, 4)
            metrics[f"{metric} reason"] = reason

    return answer, hits, metrics


def render_problem_card(problem_id: str, problems: pd.DataFrame) -> None:
    row = problems[problems["global_problem_id"].eq(problem_id)]
    if row.empty:
        return
    item = row.iloc[0]
    tags = ", ".join(parse_listish(item.get("normalized_tags")))
    st.markdown(f"**Titulo:** {item.get('title', problem_id)}")
    st.markdown(f"**Rating:** {item.get('rating', 'N/A')} | **Tags:** {tags or 'sin tags'}")
    st.markdown(f"**Editorial:** {item.get('editorial_status', 'N/A')} | **URL:** {item.get('url', '')}")
    if problem_id in DEMO_PROBLEMS:
        st.caption(DEMO_PROBLEMS[problem_id]["why"])


def main() -> None:
    st.set_page_config(page_title="CP RAG Demo", page_icon="RAG", layout="wide")
    st.title("Demo RAG para Programacion Competitiva")
    st.caption("Consulta-respuesta con recuperacion local, LangChain y OpenAI. Las respuestas y metricas no estan hardcodeadas.")

    problems = load_problems()
    if problems.empty:
        st.error("No existe data/processed/cp_problems_dataset.csv. Ejecuta primero el dataset builder.")
        st.stop()

    with st.sidebar:
        st.header("Configuracion")
        model = st.text_input("Modelo OpenAI", value=DEFAULT_OPENAI_MODEL)
        index_source = st.radio(
            "Unidad de recuperacion",
            options=["page_nodes", "pageindex_chunks"],
            format_func=lambda x: "Page Nodes" if x == "page_nodes" else "PageIndex-ready chunks",
        )
        top_k = st.slider("Top-k contextos", min_value=3, max_value=12, value=DEFAULT_TOP_K, step=1)
        scoped = st.checkbox("Restringir busqueda al problema seleccionado", value=True)
        evaluate = st.checkbox("Evaluar respuesta con LLM-as-a-Judge", value=False)
        use_reference = st.checkbox("Usar respuestas de referencia curadas", value=False)
        st.divider()
        st.caption("Para demo en vivo, usa busqueda restringida. Para mostrar limitaciones del RAG, desactiva la restriccion.")
        st.caption("Las respuestas de referencia son un respaldo de demo; no son evaluacion automatica.")

    options = problem_options(problems)
    selected_problem = st.selectbox(
        "Problema para la demo",
        options=options,
        format_func=lambda pid: problem_label(pid, problems),
    )
    render_problem_card(selected_problem, problems)

    suggested = DEMO_PROBLEMS.get(selected_problem, {}).get("questions", [])
    if suggested:
        chosen_question = st.selectbox("Preguntas sugeridas", suggested)
    else:
        chosen_question = "What is the key observation and algorithm for this problem?"

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_problem" not in st.session_state:
        st.session_state.last_problem = selected_problem
    if st.session_state.last_problem != selected_problem:
        st.session_state.messages = []
        st.session_state.last_problem = selected_problem

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Usar pregunta sugerida", use_container_width=True):
            st.session_state.pending_query = chosen_question
    with col_b:
        if st.button("Limpiar conversacion", use_container_width=True):
            st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt_value = st.session_state.pop("pending_query", None)
    user_query = st.chat_input("Pregunta sobre el problema o pide una pista...", key="chat_input")
    if prompt_value and not user_query:
        user_query = prompt_value

    if user_query:
        scoped_problem_id = selected_problem if scoped else None
        query = (
            f"Problem: {problem_label(selected_problem, problems)}\n"
            f"Student question: {user_query}"
        )

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Recuperando contexto y consultando el modelo..."):
                try:
                    retriever = build_retriever(index_source, scoped_problem_id)
                    if use_reference and user_query in REFERENCE_ANSWERS.get(selected_problem, {}):
                        hits = retriever.retrieve(query, top_k=top_k)
                        answer = REFERENCE_ANSWERS[selected_problem][user_query]
                        metrics = {
                            "mode": "reference_answer_curated",
                            "model": "not_called",
                            "evaluation": "skipped",
                            "note": "Respuesta hardcodeada como referencia de demo, no como metrica experimental.",
                        }
                    else:
                        answer, hits, metrics = answer_with_rag(
                            query=query,
                            retriever=retriever,
                            model=model,
                            top_k=top_k,
                            evaluate=evaluate,
                        )
                except Exception as exc:
                    st.error(f"No se pudo generar la respuesta: {exc}")
                    st.stop()

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.expander("Contexto recuperado"):
                st.dataframe(build_hits_table(hits), use_container_width=True)
            with st.expander("Metricas de la consulta"):
                st.json(metrics)

    st.divider()
    st.subheader("Tres problemas recomendados para la demostracion")
    cols = st.columns(3)
    for col, problem_id in zip(cols, DEMO_PROBLEMS):
        with col:
            st.markdown(f"**{DEMO_PROBLEMS[problem_id]['label']}**")
            st.caption(DEMO_PROBLEMS[problem_id]["why"])
            for question in DEMO_PROBLEMS[problem_id]["questions"]:
                st.markdown(f"- {question}")


if __name__ == "__main__":
    main()
