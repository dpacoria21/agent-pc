# Manual de ejecucion: Div. 4 + LangChain + GPT-5.4 nano

Este manual deja el proyecto listo para hacer una demostracion local, sin Google Colab, usando problemas recientes de Codeforces Div. 4, LangChain y el modelo configurado en OpenAI.

## 1. Requisitos

- Tener Python instalado.
- Tener las dependencias instaladas.
- Tener un archivo `.env` en la raiz del proyecto.
- Tener conexion a internet para consultar Codeforces y OpenAI.

El archivo `.env` debe incluir la clave:

```env
OPENAI_API_KEY=tu_clave_aqui
OPENAI_MODEL=gpt-5.4-nano
```

El proyecto tambien acepta `OPENAPI_KEY` como alias, pero se recomienda usar `OPENAI_API_KEY`.

## 2. Instalar dependencias

Desde la raiz del proyecto:

```bash
python -m pip install -r requirements-local.txt
```

Dependencias importantes para esta fase:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `beautifulsoup4`
- `python-dotenv`
- `langchain-core`
- `langchain-openai`

## 3. Extraer 80 problemas recientes de Div. 4

Ejecuta:

```bash
python scripts/build_latest_div4_dataset.py --target-problems 80 --contest-lookback 30 --request-delay 1.0
```

Que hace:

- consulta la API publica de Codeforces;
- detecta las ultimas rondas `Div. 4`;
- selecciona aproximadamente 80 problemas;
- descarga statements;
- busca tutoriales/editoriales oficiales cuando estan disponibles;
- normaliza formulas de Codeforces desde `$$$...$$$` hacia Markdown LaTeX `$...$`;
- construye `cp_problems_dataset`;
- construye `cp_page_nodes_dataset`.

Archivos principales generados:

```text
data/processed/cp_problems_dataset.csv
data/processed/cp_page_nodes_dataset.csv
data/processed/latest_div4_contests.csv
data/processed/latest_div4_dataset_report.json
```

En la corrida actual se obtuvieron:

- 80 problemas;
- 1040 Page Nodes;
- 80 statements descargados;
- 77 editoriales descargadas;
- 3 editoriales no emparejadas.

## 4. Validar el dataset

Ejecuta:

```bash
python scripts/run_phase2_dataset_contract.py --fail-on-error
```

Que revisa:

- columnas obligatorias;
- ids de problemas;
- nodos por problema;
- placeholders de Codeforces;
- estados de statements/editorials;
- editoriales demasiado cortas.

Salida actual:

```text
contract_status: passed
problem_count: 80
page_node_count: 1040
warnings: 4
```

Las advertencias actuales son por editoriales descargadas pero inusualmente cortas.

## 5. Construir arboles semanticos con GPT

Ejecuta:

```bash
python scripts/run_phase3_llm_tree_builder.py --limit 5 --model gpt-5.4-nano
```

Que hace:

- toma problemas reales del dataset;
- envia statement + editorial al modelo;
- genera nodos pedagogicos estructurados;
- genera relaciones entre nodos.

Archivos generados:

```text
data/processed/cp_llm_tree_nodes_dataset.csv
data/processed/cp_llm_tree_edges_dataset.csv
data/processed/cp_llm_problem_analysis.json
data/processed/llm_tree_build_report.json
```

En la corrida actual:

- 5 problemas analizados por GPT;
- 60 nodos semanticos;
- 44 relaciones.

## 6. Construir indice tipo PageIndex-ready

Ejecuta:

```bash
python scripts/run_phase4_tree_index.py
```

Que hace:

- une problemas, Page Nodes y nodos semanticos generados por GPT;
- crea una estructura jerarquica;
- exporta nodos, aristas y chunks listos para comparar con PageIndex/hybrid tree search.

Archivos generados:

```text
data/processed/cp_pageindex_ready_nodes.csv
data/processed/cp_pageindex_ready_edges.csv
data/processed/cp_pageindex_ready_chunks.csv
```

En la corrida actual:

- 1190 nodos PageIndex-ready;
- 1233 aristas;
- 1464 chunks.

## 7. Ejecutar RAG con LangChain

El runner usa LangChain para:

- crear el `ChatOpenAI`;
- consultar el modelo `gpt-5.4-nano`;
- generar respuestas con contexto recuperado;
- evaluar con LLM-as-a-Judge.

La recuperacion local usa:

- TF-IDF;
- reduccion semantica con `TruncatedSVD`;
- normalizacion L2;
- producto punto sobre vectores normalizados, equivalente a similitud coseno.

### Modo A: Page Nodes

```bash
python scripts/run_langchain_openai_rag_eval.py --model gpt-5.4-nano --query-limit 8 --top-k 5 --index-source page_nodes --faithfulness-threshold 0.75 --answer-relevancy-threshold 0.75
```

Usa como unidades de recuperacion:

```text
data/processed/cp_page_nodes_dataset.csv
```

Archivos generados:

```text
data/processed/langchain_openai_rag_eval_results_page_nodes.csv
data/processed/langchain_openai_rag_eval_summary_page_nodes.csv
comparison_assets/langchain_openai_rag_eval_summary_page_nodes.png
```

### Modo B: PageIndex chunks

```bash
python scripts/run_langchain_openai_rag_eval.py --model gpt-5.4-nano --query-limit 8 --top-k 5 --index-source pageindex_chunks --faithfulness-threshold 0.75 --answer-relevancy-threshold 0.75
```

Usa como unidades de recuperacion:

```text
data/processed/cp_pageindex_ready_chunks.csv
```

Archivos generados:

```text
data/processed/langchain_openai_rag_eval_results_pageindex_chunks.csv
data/processed/langchain_openai_rag_eval_summary_pageindex_chunks.csv
comparison_assets/langchain_openai_rag_eval_summary_pageindex_chunks.png
```

## 8. Comparar metricas

Ejecuta:

```bash
python scripts/compare_langchain_eval_modes.py
```

Archivos generados:

```text
data/processed/langchain_openai_rag_eval_comparison.csv
data/processed/langchain_openai_rag_eval_comparison.json
comparison_assets/langchain_openai_rag_eval_comparison_table.png
comparison_assets/langchain_openai_rag_eval_comparison_bar.png
```

Resultados actuales:

| Modo de indice | Metrica | Umbral | Promedio | Desviacion estandar | % >= umbral |
|---|---:|---:|---:|---:|---:|
| Page Nodes | Faithfulness | 0.75 | 0.53 | 0.283 | 25 % |
| Page Nodes | Answer Relevancy | 0.75 | 0.54 | 0.290 | 25 % |
| PageIndex Chunks | Faithfulness | 0.75 | 0.43 | 0.282 | 12 % |
| PageIndex Chunks | Answer Relevancy | 0.75 | 0.54 | 0.289 | 38 % |

## 9. Como interpretar Faithfulness y Answer Relevancy

`Faithfulness` mide si la respuesta esta sustentada en el contexto recuperado. Penaliza respuestas que inventan informacion o que no se pueden verificar con los nodos recuperados.

`Answer Relevancy` mide si la respuesta contesta realmente la consulta del estudiante. Una respuesta puede ser fiel pero poco relevante si solo dice que falta informacion.

En esta corrida los valores no son altos. Eso es util para la tesis porque muestra una deficiencia real del prototipo local:

- algunos problemas tienen editoriales parciales;
- algunos tutoriales usan hints y no una solucion completa;
- `top_k=5` puede quedarse corto;
- la recuperacion TF-IDF + SVD no equivale a un vector DB semantico moderno;
- el judge LLM tiene variabilidad aun con temperatura cero.

## 10. Orden recomendado para una demo en vivo

```bash
python scripts/build_latest_div4_dataset.py --target-problems 80 --contest-lookback 30 --request-delay 1.0
python scripts/run_phase2_dataset_contract.py --fail-on-error
python scripts/run_phase3_llm_tree_builder.py --limit 5 --model gpt-5.4-nano
python scripts/run_phase4_tree_index.py
python scripts/run_langchain_openai_rag_eval.py --model gpt-5.4-nano --query-limit 8 --top-k 5 --index-source page_nodes --faithfulness-threshold 0.75 --answer-relevancy-threshold 0.75
python scripts/run_langchain_openai_rag_eval.py --model gpt-5.4-nano --query-limit 8 --top-k 5 --index-source pageindex_chunks --faithfulness-threshold 0.75 --answer-relevancy-threshold 0.75
python scripts/compare_langchain_eval_modes.py
```

## 11. Archivos para presentar

Usa estos archivos como evidencia visual:

```text
comparison_assets/langchain_openai_rag_eval_comparison_table.png
comparison_assets/langchain_openai_rag_eval_comparison_bar.png
comparison_assets/langchain_openai_rag_eval_summary_page_nodes.png
comparison_assets/langchain_openai_rag_eval_summary_pageindex_chunks.png
comparison_assets/demo_reference_answers_table.png
```

Y estos CSV para explicar trazabilidad:

```text
data/processed/langchain_openai_rag_eval_results_page_nodes.csv
data/processed/langchain_openai_rag_eval_results_pageindex_chunks.csv
data/processed/langchain_openai_rag_eval_comparison.csv
data/processed/demo_reference_answers.csv
```

## 12. Demo interactiva con Streamlit

La interfaz permite hacer una demostracion tipo consulta-respuesta:

- eliges un problema;
- eliges si la busqueda se restringe a ese problema o si usa todo el corpus;
- haces una pregunta;
- el sistema recupera contexto;
- GPT responde usando LangChain;
- puedes abrir la tabla de contexto recuperado;
- opcionalmente puedes evaluar la respuesta con `Faithfulness` y `Answer Relevancy`.

Ejecuta:

```bash
streamlit run streamlit_app.py
```

Si `streamlit` no esta en el PATH, usa:

```bash
python -m streamlit run streamlit_app.py
```

URL local esperada:

```text
http://localhost:8501
```

### Configuracion recomendada para demo

En la barra lateral:

- Modelo: `gpt-5.4-nano`
- Unidad de recuperacion: `Page Nodes`
- Top-k contextos: `5`
- Restringir busqueda al problema seleccionado: activado
- Evaluar respuesta con LLM-as-a-Judge: desactivado al inicio
- Usar respuestas de referencia curadas: activado solo si quieres una demo estable sin llamar al modelo

Para mostrar limitaciones del prototipo, desactiva la restriccion por problema y compara como cambia el contexto recuperado.

El modo de respuestas de referencia curadas esta hardcodeado para exposicion. Debe presentarse como material de demo o respuesta esperada, no como resultado experimental automatico.

### Tres problemas para demostrar

#### 1. 2218B - The 67th 6-7 Integer Problem

Uso recomendado: explicar una observacion corta, matematica y greedy.

Preguntas sugeridas:

```text
What is the key observation for solving this problem?
Explain the proof idea behind the formula 2 * max(a) - sum(a).
What implementation mistakes should I avoid?
```

#### 2. 2218F - The 67th Tree Problem

Uso recomendado: explicar construccion, paridad y prueba.

Preguntas sugeridas:

```text
Why does the parity of the root subtree matter?
Explain the constructive algorithm using pairs of vertices.
What edge cases can make the construction impossible?
```

#### 3. 2218G - The 67th Iteration of Counting is Fun

Uso recomendado: mostrar una editorial mas compleja, conteos y riesgos de implementacion.

Preguntas sugeridas:

```text
What condition makes the answer zero?
How does the editorial count valid values for each a_i?
What are the common mistakes with prefix counts and modulo?
```

### Nota sobre la demo

La interfaz no hardcodea respuestas ni metricas. Las preguntas sugeridas son curadas para que el flujo sea claro y reproducible, pero la respuesta se genera en tiempo real desde el contexto recuperado.

## 13. Nota importante sobre seguridad

No imprimas ni subas tu `.env`. El proyecto solo verifica que la clave existe y la usa localmente para llamar OpenAI mediante LangChain.
