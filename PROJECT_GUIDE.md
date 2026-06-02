# CP RAG local: guia unica de funcionamiento, comparacion y presentacion

Este proyecto es un prototipo local de RAG educativo para programacion competitiva. Actualmente tiene dos capas: un baseline local con Page Nodes planos y un prototipo local inspirado en PageIndex Hybrid Tree Search. La meta siguiente es convertirlo en una implementacion modular donde cada fase pueda desarrollarse, probarse y defenderse por separado antes de integrar PageIndex real:

https://docs.pageindex.ai/tutorials/tree-search/hybrid

Segun ese tutorial, Hybrid Tree Search combina busqueda basada en valor sobre nodos con busqueda guiada por LLM, usa chunks asociados a nodos, agrega scores por nodo, mantiene una cola de nodos unicos y permite que un agente decida si ya tiene suficiente informacion. Este repositorio ya simula esa arquitectura localmente sin API externa; todavia falta reemplazar la parte guiada por reglas por consultas reales al GPT/API y despues por PageIndex real.

## 1. Estructura limpia del proyecto

```text
agent-pc/
  PROJECT_GUIDE.md
  comparison_dashboard.html
  requirements-local.txt
  run_mingw64.sh

  src/
    dataset/
      schema.py
      quality_report.py
    llm/
      gpt_client.py
      prompts.py
      structured_outputs.py
    indexing/
      llm_tree_builder.py
    cp_dataset_scraper.py
    hybrid_tree_search.py
    rag_cp_student_profile.py
    search_helpers.py
    search_local.py
    search_faiss.py
    search_chromadb.py

  scripts/
    run_phase2_dataset_contract.py
    run_phase3_llm_tree_builder.py
    run_all_local.py
    run_hybrid_tree_prototype.py
    verify_codeforces_editorials.py
    add_math_binary_demo.py
    extract_rag_metrics.py
    compare_vector_backends.py
    generate_presentation_evidence.py
    generate_deck_diagrams.py

  data/
    cache/
    raw/
    processed/

  comparison_assets/
    hybrid_tree_architecture.png
    hybrid_tree_structure.png
    hybrid_tree_score_components.png
    hybrid_tree_recommendations.png
    page_index_comparison.png
    rag_retrieval_metrics.png
    retrieval_strategy_scores.png
    retrieval_method_overlap.png
    retrieval_average_scores.png
    vector_backend_metrics.png
    vector_backend_latency.png
    vector_backend_overlap.png
    strategy_prediction_by_model.png
    strategy_accuracy_by_model.png
    math_binary_strategy_classification.png
    math_binary_problem_strategy_map.png
```

La raiz queda con un solo `.md` y un solo `.html` de lectura. Los demas archivos son codigo, datos, dependencias o imagenes comparativas.

## 2. Como ejecutar

Desde PowerShell:

```powershell
cd C:\Users\Asus\Desktop\agent-pc
C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\run_all_local.py --skip-dataset
```

Desde MinGW64:

```bash
cd /c/Users/Asus/Desktop/agent-pc
./run_mingw64.sh --skip-dataset
```

Resultado esperado:

```text
PROJECT_GUIDE.md
comparison_dashboard.html
comparison_assets/
data/processed/
```

Abre `comparison_dashboard.html` para mostrar solo graficos comparativos.

Para ejecutar el nuevo prototipo local inspirado en PageIndex Hybrid Tree Search:

```powershell
cd C:\Users\Asus\Desktop\agent-pc
C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\run_hybrid_tree_prototype.py
```

Salida principal:

```text
data/processed/cp_tree_nodes_dataset.csv
data/processed/cp_tree_chunks_dataset.csv
data/processed/hybrid_tree_search_results.csv
data/processed/hybrid_tree_recommendations.csv
data/processed/hybrid_tree_demo_runs.csv
hybrid_tree_dashboard.html
comparison_assets/hybrid_tree_architecture.png
```

En el dashboard, las secciones clave para tu pregunta son:

- `Comparacion explicita de resultados recuperados`: tabla top-3 por backend para las queries binary/math.
- `Problemas demo: busqueda binaria vs formula directa`: lista los tres problemas usados como caso de estudio.
- `Ideas simuladas y respuesta del clasificador`: muestra la idea recibida, la respuesta actual y la estrategia fina esperada.
- `Clasificacion binary/formula por cada modelo`: muestra si `current_heuristic`, `local_matrix`, `FAISS` y `ChromaDB` predicen `BINARY_SEARCH` o `MATH_FORMULA`.

## 3. Que usa realmente el prototipo

La version local tiene tres capas de comparacion:

1. Baseline en memoria.
2. Ejemplo con FAISS.
3. Ejemplo con ChromaDB.

El baseline original usa:

- `pandas` para cargar y filtrar datasets.
- `scikit-learn` para TF-IDF y SVD.
- `numpy` para producto punto y similitud coseno.
- `matplotlib` para graficos.
- `sentence-transformers/all-MiniLM-L6-v2` solo si esta disponible.

Los ejemplos nuevos usan:

- `faiss-cpu` con `IndexFlatIP`.
- `chromadb` con una coleccion efimera y `hnsw:space = cosine`.

Para que la comparacion sea justa, los tres backends reciben exactamente los mismos embeddings:

```text
TF-IDF -> TruncatedSVD -> L2 normalization
```

Asi, si las metricas son iguales entre backends, eso es esperable: el objetivo no es cambiar el modelo semantico, sino mostrar distintas formas de servir la busqueda vectorial.

En la ejecucion local de presentacion se fuerza el fallback:

```python
# scripts/extract_rag_metrics.py
os.environ["RAG_FORCE_SEMANTIC_FALLBACK"] = "1"
rag = importlib.import_module("rag_cp_student_profile")
```

Eso activa:

```python
# src/rag_cp_student_profile.py
semantic_fallback_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
)
semantic_tfidf = semantic_fallback_vectorizer.fit_transform(node_texts)
semantic_svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
node_embeddings = semantic_svd.fit_transform(semantic_tfidf)
node_embeddings = normalize(node_embeddings)
SEMANTIC_BACKEND = "tfidf-svd-fallback"
```

Frase para presentacion:

> En el baseline uso matrices en memoria: TF-IDF, SVD, normalizacion y similitud coseno contra los Page Nodes. Luego agrego dos motores vectoriales, FAISS y ChromaDB, usando los mismos embeddings para comparar backend, latencia y consistencia de ranking.

## 3.2 Prototipo Hybrid Tree Search local

Archivo principal:

```text
src/hybrid_tree_search.py
```

Runner:

```text
scripts/run_hybrid_tree_prototype.py
```

Este modulo implementa una version local e interpretable inspirada en PageIndex Hybrid Tree Search. No llama al servicio de PageIndex ni a un LLM externo. La idea es dejar un prototipo funcional que puedas presentar y luego reemplazar por PageIndex real.

Flujo:

```text
1. cp_problems_dataset + cp_page_nodes_dataset
2. build_tree_nodes(...)
   root -> platform -> topic -> difficulty bucket -> problem -> section group -> content node
3. build_tree_chunks(...)
   divide node_text en chunks asociados al nodo
4. Value Search
   TF-IDF + SVD + L2 normalization + cosine similarity sobre chunks
5. Node scoring
   agrega scores de chunks al nodo y propaga score hacia ancestros
6. Guided Search
   analiza query: etapa, enfoque, riesgos y node_type esperado
7. Hybrid Queue
   combina nodos de value search y guided search, elimina duplicados y rerankea
8. Recommendation
   agrega scores por problema y genera razones interpretables
```

Componentes de score:

```text
hybrid_score = alpha * value_score
             + (1 - alpha) * guided_score
             + metadata_bonus
```

Donde:

```text
value_score       = similitud coseno de chunks agregada al nodo
guided_score      = coincidencia entre query intent, node_type, tags y resumen
metadata_bonus    = filtros por tags, dificultad y etapa pedagogica
```

Las queries demo actuales son:

```text
grid_l_proof
unique_values_formula
tree_mex_implementation
weird_chessboard_proof
```

En la ultima ejecucion:

```text
problems:   7
page_nodes: 91
tree_nodes: 133
chunks:     181
embedding:  TF-IDF + SVD + L2
```

Resultados esperados:

```text
grid_l_proof              -> codeforces_2219_A
unique_values_formula     -> codeforces_2219_B1
tree_mex_implementation   -> codeforces_2219_D
weird_chessboard_proof    -> codeforces_2219_E
```

Esta implementacion sigue la intuicion de PageIndex: recuperar nodos, no solo chunks. La diferencia es que aqui el "LLM tree search" esta simulado con reglas de intencion para poder ejecutar todo localmente sin API key.

## 3.3 Como obtiene editoriales de Codeforces

Antes las editoriales quedaban vacias porque el scraper buscaba texto directamente dentro del blog. En Codeforces, muchos blogs oficiales solo dejan un placeholder:

```text
Tutorial is loading...
```

La solucion real se carga despues con JavaScript desde un endpoint interno de la pagina. El flujo corregido en `src/cp_dataset_scraper.py` es:

```text
1. Tomar el problema desde la API de Codeforces.
2. Usar la URL de scraping:
   https://codeforces.com/contest/{contestId}/problem/{index}
3. Buscar en el bloque Contest materials el enlace Tutorial.
4. Abrir el blog oficial, por ejemplo:
   https://codeforces.com/blog/entry/152936
5. Detectar el problemCode del spoiler:
   2219A, 750A, 1915C, etc.
6. Leer el token CSRF publico de la pagina.
7. Hacer la misma llamada AJAX que hace Codeforces:
   POST /data/problemTutorial
   problemCode={contestId}{index}
8. Parsear el HTML devuelto y guardarlo en official_editorial.
9. Construir nodos EDITORIAL_FULL, EDITORIAL_PROOF,
   EDITORIAL_ALGORITHM y EDITORIAL_COMPLEXITY desde ese texto.
```

Funciones involucradas:

```python
find_codeforces_tutorial_url_from_problem_page(...)
find_codeforces_editorial_url(...)
fetch_codeforces_problem_tutorial(...)
scrape_codeforces_editorial(...)
```

Ejemplo del caso que viste en el navegador:

```text
Problema:  https://codeforces.com/problemset/problem/2219/A
Scraping:  https://codeforces.com/contest/2219/problem/A
Tutorial:  https://codeforces.com/blog/entry/152936
AJAX:      /data/problemTutorial
Code:      2219A
```

Esto no intenta evadir restricciones ni acceder a contenido privado. Solo reproduce la ruta publica que la pagina usa para mostrar el tutorial, con cache, delay y manejo de errores. Si Codeforces responde con challenge, redireccion, robots disallowed o bloqueo temporal, el dataset deja `editorial_status` como `access_challenge`, `unavailable`, `redirected` o estado equivalente, y el proceso continua sin romper el pipeline.

Para probar pocas editoriales sin hacer scraping grande:

```bash
./run_mingw64.sh --max-cf 3 --max-atcoder 0 --with-content --request-delay 1.5
```

Para verificar explicitamente que cada problema de Codeforces recibe su propia editorial, usa:

```powershell
C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\verify_codeforces_editorials.py --request-delay 1.5 --timeout 25
```

O desde MinGW64:

```bash
/c/Users/Asus/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe scripts/verify_codeforces_editorials.py --request-delay 1.5 --timeout 25
```

La prueba por defecto revisa:

```text
2219A, 2219B1, 2219C, 2219D, 2219E, 2220A, 2220B
```

Salida principal:

```text
data/processed/codeforces_editorial_verification.csv
data/processed/codeforces_editorial_verification.json
data/processed/codeforces_editorial_verification_summary.json
```

Campos importantes del reporte:

```text
problem_code
editorial_status
editorial_parse_method
editorial_text_chars
editorial_toggle_count
problem_code_ok
verified
```

En la ultima verificacion limpia, los 7 problemas fueron `verified=True`, todos con `editorial_parse_method=codeforces_ajax_problemTutorial`. Eso significa que el scraper no se quedo en el texto superficial del blog, sino que entro al contenido cargado por los toggles/spoilers de Codeforces.

## 4. Como genera los Page Nodes

Archivo:

```text
src/cp_dataset_scraper.py
```

Funcion principal:

```python
build_page_nodes_dataset(problems_dataset)
```

Cada problema se transforma en nodos pedagogicos:

```text
STATEMENT
INPUT
OUTPUT
CONSTRAINTS
EXAMPLES
NOTES
EDITORIAL_FULL
EDITORIAL_OBSERVATION
EDITORIAL_PROOF
EDITORIAL_ALGORITHM
EDITORIAL_COMPLEXITY
IMPLEMENTATION_HINTS
COMMON_MISTAKES
```

Cada nodo queda asociado al problema por:

```text
global_problem_id
node_id
parent_node_id
node_type
```

Ejemplo conceptual:

```text
codeforces_750_A::ROOT
codeforces_750_A::01_STATEMENT
codeforces_750_A::09_EDITORIAL_PROOF
codeforces_750_A::13_COMMON_MISTAKES
```

Importante:

> El prototipo actual descompone problemas en secciones utiles, pero todavia no construye un arbol de busqueda tipo PageIndex ni relaciones explicitas entre problemas similares.

## 5. Como busca informacion

Archivo:

```text
src/rag_cp_student_profile.py
```

### 5.1 Busqueda semantica

```python
def semantic_scores_for_query(query):
    query_embedding = encode_query_semantic(query)
    scores = np.dot(node_embeddings, query_embedding)
    return _clip_cosine(scores)
```

Como los vectores estan normalizados, `np.dot()` equivale a similitud coseno.

```text
semantic_score = cosine(query_embedding, node_embedding)
```

### 5.2 Busqueda keyword

```python
keyword_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
)
keyword_matrix = keyword_vectorizer.fit_transform(keyword_documents)
```

Favorece coincidencias literales como:

```text
dp, transition, greedy, proof, edge, binary search
```

### 5.3 Busqueda hibrida local

```python
hybrid_score =
  alpha * semantic_score
  + (1 - alpha) * keyword_score
  + metadata_bonus
```

Donde:

- `semantic_score`: similitud conceptual.
- `keyword_score`: coincidencia literal.
- `metadata_bonus`: tags, dificultad y tipo de nodo.

Esto es hibrido en sentido local, pero no es PageIndex Hybrid Tree Search. No hay cola de nodos, no hay LLM tree search y no hay score agregado chunk->node como en el tutorial de PageIndex.

## 6. Que comparan los graficos

El dashboard `comparison_dashboard.html` muestra solo graficos comparativos.

### 6.1 `page_index_comparison.png`

Compara visualmente:

- RAG plano por chunks.
- Page Index local por secciones.
- PageIndex Hybrid Tree Search como objetivo futuro.

### 6.2 `rag_retrieval_metrics.png`

Compara metricas sobre queries con ground truth simulado:

```python
ground_truth = {
    "greedy proof": [node_ids esperados],
    "dp state transition": [node_ids esperados],
    "edge cases implementation": [node_ids esperados],
}
```

Metricas:

- `precision_at_k`
- `recall_at_k`
- `reciprocal_rank`
- `average_similarity_score`
- `diversity_by_tags`
- `coverage_by_node_type`

Lo que mide:

> Si los nodos esperados aparecen dentro del top-k recuperado.

Lo que no mide:

> No ejecuta codigo del estudiante ni valida una solucion con tests.

### 6.3 `retrieval_strategy_scores.png`

Compara ranking y scores de:

```python
semantic_search(query)
keyword_search(query)
hybrid_search(query)
personalized_hybrid_search(query, user_id)
```

Query usada:

```text
I have problems proving greedy solutions and handling edge cases
```

### 6.4 `retrieval_method_overlap.png`

Compara cuantos `node_id` se repiten entre los top-k de cada metodo.

Sirve para explicar que los metodos no recuperan exactamente lo mismo.

Este es el grafico que compara resultados de busqueda entre metodos del RAG base:

```text
semantic_search
keyword_search
hybrid_search
personalized_hybrid_search
```

### 6.5 `retrieval_average_scores.png`

Compara el score promedio por metodo de recuperacion.

### 6.6 `vector_backend_metrics.png`

Compara tres backends sobre el mismo dataset y las mismas queries:

```text
local_matrix
faiss
chromadb
```

Los tres usan los mismos vectores `TF-IDF + SVD` normalizados. Por eso, en una corrida exacta, pueden recuperar resultados muy parecidos o identicos.

### 6.7 `vector_backend_latency.png`

Compara el tiempo promedio por query. En datasets pequenos, la matriz local puede verse mas rapida porque no paga costos de crear coleccion, agregar documentos o inicializar indices. En datasets grandes, FAISS y ChromaDB se vuelven mas relevantes.

### 6.8 `vector_backend_overlap.png`

Mide el overlap top-k entre backends con Jaccard promedio:

```text
overlap = |top_k_A interseccion top_k_B| / |top_k_A union top_k_B|
```

Si el overlap es alto, los backends estan devolviendo practicamente los mismos nodos para las mismas queries.

Este es el grafico que compara resultados de busqueda entre backends vectoriales:

```text
local_matrix
FAISS
ChromaDB
```

Ademas, el HTML incluye una tabla llamada `Comparacion explicita de resultados recuperados`, que muestra directamente:

```text
query_id
backend
rank
global_problem_id recuperado
node_type
score
is_relevant
```

### 6.9 `math_binary_strategy_classification.png`

Compara:

- clasificacion actual del agente;
- clasificacion fina propuesta para la demo.

### 6.10 `strategy_prediction_by_model.png`

Este es el grafico pedido para ver como clasifica cada modelo/backend las ideas de solucion:

```text
current_heuristic
local_matrix
FAISS
ChromaDB
```

Muestra cuantas ideas fueron clasificadas como:

```text
BINARY_SEARCH
MATH_FORMULA
UNKNOWN
MATH
```

### 6.11 `strategy_accuracy_by_model.png`

Muestra exactitud por modelo y estrategia esperada:

```text
BINARY_SEARCH
MATH_FORMULA
```

Sirve para ver si el modelo realmente esta diferenciando ambas rutas.

### 6.12 `math_binary_problem_strategy_map.png`

Muestra que los mismos problemas pueden admitir dos rutas:

- busqueda binaria;
- formula matematica.

La parte de los problemas de solucion por busqueda binaria y solucion directa por formula esta en tres lugares:

```text
data/processed/math_binary_demo_problems.csv
data/processed/math_binary_classification_report.csv
comparison_dashboard.html
```

En el HTML aparece como:

- `Problemas demo: busqueda binaria vs formula directa`;
- `Ideas simuladas y respuesta del clasificador`;
- `Clasificacion actual vs estrategia fina`;
- `Estrategias detectadas por problema demo`.
- `Clasificacion binary/formula por cada modelo`.

## 7. Flujo binary search vs math formula

Archivo:

```text
scripts/add_math_binary_demo.py
```

Problemas agregados:

| ID | Problema | Idea pedagogica |
|---|---|---|
| `codeforces_750_A` | New Year and Hurry | Inecuacion cuadratica o busqueda binaria del maximo x |
| `codeforces_1915_C` | Can I Square? | Raiz entera o busqueda binaria de la raiz |
| `codeforces_192_A` | Funky Numbers | Numeros triangulares, discriminante o busqueda/precomputo |

Cada problema declara estrategias aceptadas:

```python
"accepted_strategies": ["BINARY_SEARCH", "MATH_FORMULA"]
```

o:

```python
"accepted_strategies": ["BINARY_SEARCH", "MATH_FORMULA", "PRECOMPUTE_ENUMERATION"]
```

### 7.1 Ideas simuladas

Estudiante con ruta algoritmica:

```text
I will binary search the maximum x. The check is 5*x*(x+1)/2 <= 240-k and x <= n.
```

Estudiante con ruta matematica:

```text
I want to solve the quadratic inequality directly with the formula and floor the positive root.
```

### 7.2 Clasificador actual

El clasificador actual replica `analyze_student_idea()` y usa reglas amplias:

```python
if contains_any(text, ["dp", "state", "transition"]):
    approach = "DP"
elif contains_any(text, ["sort", "greedy", "choose", "always"]):
    approach = "GREEDY"
elif contains_any(text, ["math", "modulo", "parity", "formula", "gcd"]):
    approach = "MATH"
else:
    approach = "UNKNOWN"
```

Deficiencia:

```text
No existe BINARY_SEARCH como clase del clasificador actual.
```

Por eso:

| Idea | Respuesta actual |
|---|---|
| "I will binary search..." | `UNKNOWN` |
| "quadratic formula..." | `MATH` |

### 7.3 Clasificador fino de la demo

La demo agrega una capa diagnostica:

```python
if contains_any(text, ["binary search", "monotonic", "predicate"]):
    strategy = "BINARY_SEARCH"
elif contains_any(text, ["formula", "quadratic", "discriminant", "sqrt"]):
    strategy = "MATH_FORMULA"
elif contains_any(text, ["precompute", "set", "enumerate"]):
    strategy = "PRECOMPUTE_ENUMERATION"
else:
    strategy = "UNKNOWN"
```

Resultado local:

```text
current_agent_counts = {"UNKNOWN": 4, "MATH": 2}
fine_strategy_counts = {"BINARY_SEARCH": 3, "MATH_FORMULA": 3}
```

Lectura para defensa:

> El prototipo actual detecta algunas senales matematicas generales, pero no distingue todavia estrategia algoritmica por busqueda binaria frente a estrategia matematica por formula. Esa es una deficiencia concreta y medible.

## 8. Brechas frente a PageIndex Hybrid Tree Search real

El repositorio ahora tiene dos niveles: baseline plano y prototipo local con arbol. Aun asi, el prototipo local no reemplaza a PageIndex real porque la rama guiada por LLM esta simulada con reglas.

| Aspecto | Baseline plano | Prototipo tree local | PageIndex Hybrid Tree Search real |
|---|---|---|---|
| Estructura | Page Nodes planos | Arbol local `root -> platform -> topic -> difficulty -> problem -> section -> content` | Arbol PageIndex gestionado por su framework |
| Chunks | Nodos completos | Chunks asociados a nodos | Chunks asociados a nodos |
| Value search | Coseno sobre nodos | Coseno sobre chunks + propagacion a nodos | Value-based tree search |
| Guided search | Heuristica simple | Intent classifier por reglas | LLM tree search |
| Cola hibrida | No | Deduplicacion y reranking local | Cola hibrida de nodos unicos |
| Early stopping | No | `enough_information` heuristico | Decision del agente/LLM |
| Student model | Simulado | Metadata y perfiles locales | Integracion con memoria/agente |
| Evaluacion | Ground truth pequeno | Demos y scores locales | Evaluacion experimental completa |

Brechas que quedan:

1. Falta usar GPT para construir arboles semanticamente ricos desde las editoriales.
2. Falta usar GPT para analizar ideas del estudiante.
3. Falta reemplazar el guided search por LLM-guided tree search.
4. Falta adaptar el tree index al formato final de PageIndex.
5. Falta una evaluacion comparativa fuerte con ground truth manual.
6. Falta validar recomendaciones con perfiles reales o simulaciones mas rigurosas.

Nota sobre FAISS y ChromaDB:

> FAISS y ChromaDB solo cambian el motor vectorial. La mejora pedagogica no viene de usar otra base vectorial, sino de tener mejor estructura, mejor analisis de la idea del estudiante y mejor politica de recuperacion/recomendacion.

## 9. Implementacion comparativa con FAISS y ChromaDB

Archivo:

```text
scripts/compare_vector_backends.py
```

Este archivo ya no contiene la logica interna de cada backend. Solo orquesta la comparacion:

```text
src/search_helpers.py   -> carga nodos, crea documentos, embeddings, metricas y graficos
src/search_local.py     -> busqueda por matriz local
src/search_faiss.py     -> busqueda con FAISS IndexFlatIP
src/search_chromadb.py  -> busqueda con ChromaDB efimero
```

Entrada:

```text
data/processed/cp_page_nodes_dataset.csv
```

Solo se indexan nodos con `node_text` no vacio:

```python
nodes = nodes[nodes["node_text"].str.len() > 0].copy()
```

Esto explica por que el numero de nodos indexados puede ser menor que el total de Page Nodes. El total incluye nodos estructurales vacios o secciones sin contenido.

### 9.1 Helpers compartidos

Archivo:

```text
src/search_helpers.py
```

Responsabilidades:

- cargar `cp_page_nodes_dataset.csv`;
- eliminar nodos sin texto;
- construir documentos `node_title + node_type + node_text + tags`;
- crear embeddings `TF-IDF + SVD + L2 normalization`;
- calcular ground truth por query;
- calcular `precision_at_k`, `recall_at_k` y `reciprocal_rank`;
- generar graficos comparativos.

### 9.2 Embeddings compartidos

```python
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
)
tfidf = vectorizer.fit_transform(documents)
svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
node_embeddings = normalize(svd.fit_transform(tfidf)).astype("float32")
query_embeddings = normalize(
    svd.transform(vectorizer.transform(queries))
).astype("float32")
```

Esto crea un espacio vectorial unico. Luego ese mismo espacio se entrega a los tres backends.

### 9.3 Baseline local

Archivo:

```text
src/search_local.py
```

```python
def search_local_matrix(node_embeddings, query_embeddings, top_k):
    outputs = []
    for query_embedding in query_embeddings:
        scores = node_embeddings @ query_embedding
        order = np.argsort(-scores)[:top_k]
        outputs.append((order, scores[order]))
    return outputs
```

Es producto punto sobre vectores normalizados. Equivale a similitud coseno.

### 9.4 FAISS

Archivo:

```text
src/search_faiss.py
```

```python
def search_faiss_index_flat_ip(node_embeddings, query_embeddings, top_k):
    import faiss

    index = faiss.IndexFlatIP(node_embeddings.shape[1])
    index.add(node_embeddings)
    scores, indices = index.search(query_embeddings, top_k)
    return [(indices[i], scores[i]) for i in range(len(query_embeddings))]
```

`IndexFlatIP` usa inner product. Como los vectores estan normalizados, inner product equivale a coseno.

Ventaja:

- Es un ejemplo directo de indice vectorial.
- Escala mejor que una matriz manual cuando el dataset crece.

Limitacion:

- En este ejemplo se crea en memoria cada vez.
- No hay persistencia ni busqueda jerarquica.

### 9.5 ChromaDB

Archivo:

```text
src/search_chromadb.py
```

```python
collection = client.create_collection(
    name=f"cp_nodes_backend_comparison_{int(time.time() * 1000)}",
    metadata={"hnsw:space": "cosine"},
)
collection.add(
    ids=ids,
    embeddings=node_embeddings.tolist(),
    documents=documents,
    metadatas=metadatas,
)
result = collection.query(
    query_embeddings=query_embeddings.tolist(),
    n_results=top_k,
    include=["distances"],
)
```

ChromaDB recibe los embeddings ya calculados. En esta demo no le pedimos a Chroma que cree embeddings con un modelo externo.

Ventaja:

- Muestra una base vectorial mas cercana a una aplicacion RAG real.
- Permite guardar documentos, metadatos y embeddings juntos.

Limitacion:

- En esta demo se usa una coleccion efimera.
- Puede tener mas overhead inicial que FAISS o la matriz local.

### 9.6 Queries usadas para comparar backends

```text
binary search maximum x quadratic inequality contest time
integer square root perfect square sum
triangular numbers discriminant formula binary search
dp state transition editorial algorithm
greedy proof edge cases common mistakes
```

Las tres primeras apuntan a la demo binary/math. Las dos ultimas prueban temas generales del RAG.

### 9.7 Archivos generados

```text
comparison_assets/vector_backend_metrics.csv
comparison_assets/vector_backend_results.csv
comparison_assets/vector_backend_summary.json
comparison_assets/vector_backend_metrics.png
comparison_assets/vector_backend_latency.png
comparison_assets/vector_backend_overlap.png
```

### 9.8 Por que dan casi los mismos resultados

No es un bug. En esta comparacion es esperable.

Los tres motores reciben:

```text
mismos nodos
mismos documentos
mismos embeddings TF-IDF + SVD
misma normalizacion L2
mismas queries
mismo top_k
misma nocion de similitud coseno
```

Entonces:

- `local_matrix` calcula `node_embeddings @ query_embedding`.
- `FAISS IndexFlatIP` calcula inner product exacto.
- ChromaDB esta configurado con `hnsw:space = cosine` y recibe los mismos embeddings.

Como los vectores estan normalizados:

```text
inner_product = cosine_similarity
```

Por eso `local_matrix` y `FAISS IndexFlatIP` deberian ser identicos o casi identicos. ChromaDB tambien puede coincidir porque el dataset es pequeno y los vecinos son faciles de recuperar.

Cuando si deberian variar:

- si se usa un embedding diferente por backend;
- si FAISS usa un indice aproximado como IVF/HNSW en lugar de `IndexFlatIP`;
- si ChromaDB usa configuraciones HNSW mas agresivas;
- si el dataset crece mucho;
- si se agregan filtros por metadata;
- si se persisten colecciones con politicas distintas;
- si se compara busqueda plana contra tree-search real;
- si se agrega reranking con LLM.

Conclusion para defensa:

> Los resultados iguales muestran que los backends estan recibiendo la misma representacion vectorial. La diferencia actual es operacional: interfaz, latencia, persistencia y escalabilidad. La diferencia de calidad pedagogica deberia aparecer al cambiar la estrategia de retrieval, por ejemplo con PageIndex Hybrid Tree Search, no solo al cambiar de motor vectorial.

### 9.9 Interpretacion correcta

Si FAISS, ChromaDB y la matriz local dan metricas parecidas, no significa que los tres sistemas sean equivalentes en produccion. Significa que, con el mismo dataset pequeno y los mismos embeddings, recuperan nodos similares.

La comparacion sirve para explicar:

- el baseline local;
- un indice vectorial FAISS;
- una base vectorial ChromaDB;
- la diferencia entre cambiar backend y cambiar estrategia de retrieval.

## 10. Prompt maestro para analizar este prototipo y compararlo con PageIndex

Usa este prompt cuando quieras pedir a otra IA que analice el proyecto sin implementar todavia PageIndex Hybrid Tree Search:

```text
Actua como ingeniero senior de RAG, investigador en IA educativa y evaluador tecnico de sistemas de retrieval.

Tengo un prototipo local de RAG educativo para programacion competitiva ubicado en una carpeta con esta estructura:

- src/cp_dataset_scraper.py
- src/rag_cp_student_profile.py
- src/search_helpers.py
- src/search_local.py
- src/search_faiss.py
- src/search_chromadb.py
- scripts/run_all_local.py
- scripts/add_math_binary_demo.py
- scripts/extract_rag_metrics.py
- scripts/compare_vector_backends.py
- scripts/generate_presentation_evidence.py
- scripts/generate_deck_diagrams.py
- data/processed/
- comparison_assets/
- PROJECT_GUIDE.md
- comparison_dashboard.html

No quiero que implementes PageIndex Hybrid Tree Search todavia. Quiero que analices el prototipo actual como baseline para contrastarlo despues con:

https://docs.pageindex.ai/tutorials/tree-search/hybrid

Objetivos del analisis:

1. Explicar que usa el baseline local y como se agregaron FAISS/ChromaDB:
   - pandas;
   - TF-IDF;
   - TruncatedSVD;
   - numpy.dot;
   - similitud coseno;
   - matplotlib.

   Explicar tambien los ejemplos con:
   - FAISS IndexFlatIP;
   - ChromaDB con hnsw:space = cosine;
   - mismos embeddings TF-IDF + SVD para comparar de forma justa.

2. Explicar como se generan los Page Nodes:
   - origen en cp_problems_dataset;
   - salida en cp_page_nodes_dataset;
   - campos node_id, global_problem_id, parent_node_id, node_type, node_text;
   - tipos STATEMENT, EDITORIAL_PROOF, EDITORIAL_ALGORITHM, COMMON_MISTAKES, etc.

3. Explicar como funciona el retrieval:
   - semantic_search;
   - keyword_search;
   - hybrid_search;
   - personalized_hybrid_search;
   - formula hybrid_score = alpha * semantic_score + (1-alpha) * keyword_score + metadata_bonus.

4. Explicar que comparan los graficos del HTML:
   - precision@k, recall@k y reciprocal_rank;
   - score por ranking;
   - overlap top-k entre metodos;
   - score promedio;
   - clasificacion binary search vs math formula.
   - comparacion local_matrix vs FAISS vs ChromaDB.

5. Analizar la demo binary/math:
   - codeforces_750_A New Year and Hurry;
   - codeforces_1915_C Can I Square?;
   - codeforces_192_A Funky Numbers;
   - estudiante que propone busqueda binaria;
   - estudiante que propone formula matematica;
   - salida actual: UNKNOWN/MATH;
   - salida fina esperada: BINARY_SEARCH/MATH_FORMULA.

6. Identificar brechas frente a PageIndex Hybrid Tree Search:
   - el baseline plano no tiene arbol;
   - el prototipo tree local si agrega chunks a nodos, pero no usa PageIndex real;
   - el guided search local aun usa reglas, no LLM;
   - el early stopping es heuristico;
   - no hay evaluacion real con usuarios;
   - no hay juez online ni ejecucion de codigo.

7. Proponer una matriz de evaluacion futura, sin implementar:
   - baseline local actual vs PageIndex Hybrid Tree Search;
   - queries iguales;
   - ground truth manual;
   - precision@k;
   - recall@k;
   - MRR;
   - latencia;
   - cobertura por node_type;
   - calidad pedagogica de la respuesta.

Entrega:

- Explicacion tecnica clara.
- Tabla de limitaciones.
- Tabla de metricas sugeridas.
- Ruta de codigo con archivos y funciones.
- Recomendaciones para la futura implementacion real, sin escribir codigo de PageIndex aun.
```

## 11. Como defenderlo en una frase

> Este prototipo ya tiene un baseline plano y una simulacion local tipo Hybrid Tree Search: extrae problemas y editoriales reales, construye Page Nodes, genera un arbol de recuperacion, compara backends vectoriales y produce recomendaciones interpretables. La siguiente fase es reemplazar las reglas por GPT para construir arboles semanticos y analizar ideas del estudiante, antes de integrar PageIndex real.

## 12. Roadmap modular del proyecto

La fase 1 ya esta lista: extraccion responsable de problemas Codeforces y editoriales oficiales. A partir de ahora el proyecto debe crecer por modulos, no como un unico notebook/script. Cada modulo debe tener entrada, salida, validacion y demo.

### 12.1 Estado actual

| Fase | Modulo | Estado | Resultado |
|---|---|---|---|
| 1 | Extraccion Codeforces + editoriales | Listo | `cp_problems_dataset`, `cp_page_nodes_dataset`, verificacion 7/7 editoriales |
| 2 | Contrato de dataset y calidad | Implementado | `dataset_contract_report.json`, `dataset_quality_report.json`, `dataset_contract_issues.csv` |
| 3 | Estructuracion semantica con GPT | Implementado base | `cp_llm_problem_analysis.json`, `cp_llm_tree_nodes_dataset`, `cp_llm_tree_edges_dataset` |
| 4 | Tree Index local/PageIndex-ready | Parcial | Existe `hybrid_tree_search.py`, falta formato compatible final |
| 5 | Retrieval hibrido real | Parcial | Existe simulacion local; falta GPT-guided search y PageIndex real |
| 6 | Analisis de ideas del estudiante | Pendiente critico | Falta usar GPT para clasificar enfoque, etapa, riesgos y feedback |
| 7 | Recomendador adaptativo | Parcial | Existe scoring local, falta personalizacion con analisis LLM |
| 8 | Evaluacion experimental | Parcial | Hay metricas basicas, falta ground truth fuerte y ablation study |
| 9 | Demo/presentacion | Parcial | Hay dashboards e imagenes, falta demo narrativa final |

### 12.2 Estructura modular recomendada

No es necesario mover todo de golpe. La migracion puede ser gradual desde los archivos actuales hacia esta estructura:

```text
src/
  ingestion/
    codeforces_scraper.py
    atcoder_scraper.py
    editorial_scraper.py

  dataset/
    schema.py
    quality_report.py
    normalization.py

  llm/
    gpt_client.py
    prompts.py
    structured_outputs.py

  indexing/
    page_nodes.py
    llm_tree_builder.py
    tree_schema.py
    pageindex_adapter.py

  retrieval/
    local_tree_search.py
    pageindex_hybrid_search.py
    vector_backends.py
    reranker.py

  student_model/
    idea_analyzer.py
    profile_builder.py
    recommendation_engine.py

  evaluation/
    retrieval_metrics.py
    recommendation_metrics.py
    student_analysis_metrics.py
    ablation.py

  reporting/
    dashboards.py
    architecture_diagrams.py
```

Mapeo desde lo que existe hoy:

| Archivo actual | Modulo futuro |
|---|---|
| `src/cp_dataset_scraper.py` | `ingestion/`, `dataset/`, `indexing/page_nodes.py` |
| `src/hybrid_tree_search.py` | `indexing/tree_schema.py`, `retrieval/local_tree_search.py` |
| `src/search_local.py`, `search_faiss.py`, `search_chromadb.py` | `retrieval/vector_backends.py` |
| `src/rag_cp_student_profile.py` | `student_model/`, `retrieval/`, `evaluation/` |
| `scripts/verify_codeforces_editorials.py` | `evaluation/dataset_quality.py` |
| `scripts/run_hybrid_tree_prototype.py` | `scripts/run_phase_4_tree_search.py` |

### 12.3 Fase 2: contrato del dataset

Objetivo:

```text
Congelar la forma minima que todo modulo posterior puede asumir.
```

Entradas:

```text
data/processed/cp_problems_dataset.csv
data/processed/cp_page_nodes_dataset.csv
```

Salidas:

```text
data/processed/dataset_contract_report.json
data/processed/dataset_quality_report.json
```

Validaciones:

```text
- cada problema tiene global_problem_id unico;
- cada problema Codeforces tiene editorial_status;
- cada editorial descargada tiene editorial_problem_code correcto;
- cada Page Node tiene parent/global_problem_id;
- cada nodo editorial importante no esta vacio;
- no aparece "Tutorial is loading..." en official_editorial.
```

Esta fase evita que el resto del sistema falle por columnas faltantes o parseos incompletos.

Implementacion actual:

```text
src/dataset/schema.py
src/dataset/quality_report.py
scripts/run_phase2_dataset_contract.py
```

Comando:

```powershell
C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\run_phase2_dataset_contract.py --fail-on-error
```

Ultima ejecucion:

```text
contract_status: passed
problem_count: 7
page_node_count: 91
severity_counts: {}
```

### 12.4 Fase 3: estructuracion semantica con GPT

Esta sera la primera fase que usa tu API de GPT.

Objetivo:

```text
Convertir cada problema + editorial en una estructura pedagogica tipo arbol.
```

Entrada por problema:

```text
title
statement
constraints
samples
official_editorial
rating
tags
```

Salida esperada:

```text
data/processed/cp_llm_problem_analysis.json
data/processed/cp_llm_tree_nodes_dataset.json
```

El GPT no debe inventar soluciones. Debe estructurar lo que ya existe:

```json
{
  "problem_id": "codeforces_2219_A",
  "main_topic": "math_constructive",
  "strategies": ["direct_formula", "bounded_search"],
  "student_skills": ["proof", "mathematical_modeling", "implementation"],
  "tree": {
    "title": "Grid L",
    "children": [
      {
        "type": "MATHEMATICAL_MODEL",
        "title": "Edge count equation",
        "evidence": "p+2q = m(n+1)+n(m+1)"
      },
      {
        "type": "PROOF",
        "title": "Necessity and sufficiency",
        "children": [
          {"type": "PROOF_STEP", "title": "Necessity"},
          {"type": "PROOF_STEP", "title": "Sufficiency"}
        ]
      },
      {
        "type": "ALGORITHM",
        "title": "Iterate bounded n and compute m"
      }
    ]
  }
}
```

Prompt base para esta fase:

```text
Actua como experto en programacion competitiva e IA educativa.
Recibiras el statement y editorial oficial de un problema.
Tu tarea es estructurar el contenido en un arbol pedagogico JSON.

Reglas:
- No inventes contenido que no este en el statement o editorial.
- Separa observacion, modelo matematico, prueba, algoritmo, complejidad, implementacion y errores comunes.
- Incluye evidence_text corto para cada nodo.
- Marca confidence entre 0 y 1.
- Si una seccion no aparece, usa null o lista vacia.
- Devuelve solo JSON valido.
```

Esta fase es importante porque PageIndex Hybrid Tree Search funciona mejor cuando el arbol tiene nodos semanticamente utiles, no solo secciones fijas.

Implementacion actual:

```text
src/llm/gpt_client.py
src/llm/prompts.py
src/llm/structured_outputs.py
src/indexing/llm_tree_builder.py
scripts/run_phase3_llm_tree_builder.py
```

Comando:

```powershell
C:\Users\Asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\run_phase3_llm_tree_builder.py --limit 3
```

Para usar GPT real, define antes:

```powershell
$env:OPENAI_API_KEY="tu_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Si no hay `OPENAI_API_KEY`, el modulo usa fallback heuristico con el mismo formato de salida. Esto permite probar el pipeline completo sin bloquear la demo.

Ultima ejecucion local:

```text
selected_problem_count: 3
tree_node_count: 16
edge_count: 5
client_available: false
generation_status: fallback_after_missing_api_key
```

Esto significa que la infraestructura de Fase 3 ya esta lista, pero esa corrida no uso GPT real porque la key no estaba disponible en el entorno de terminal.

### 12.5 Fase 4: Tree Index compatible con PageIndex

Objetivo:

```text
Transformar el JSON estructurado por GPT en nodos listos para busqueda jerarquica.
```

Salida:

```text
data/processed/cp_tree_nodes_dataset.csv
data/processed/cp_tree_chunks_dataset.csv
data/processed/cp_tree_edges_dataset.csv
```

Campos minimos:

```text
tree_node_id
parent_tree_node_id
global_problem_id
node_type
title
summary
node_text
evidence_text
depth
order
metadata
```

Aqui tambien se agregan relaciones tipo grafo:

```text
SAME_STRATEGY
PREREQUISITE_OF
ALTERNATIVE_APPROACH
SAME_PROOF_PATTERN
SAME_COMMON_MISTAKE
```

Aunque PageIndex trabaja con arbol, estas relaciones cruzadas pueden guardarse como metadata adicional para reranking y analisis, sin romper la estructura principal.

### 12.6 Fase 5: retrieval hibrido

Objetivo:

```text
Comparar tres niveles de retrieval.
```

Niveles:

| Nivel | Metodo | Uso |
|---|---|---|
| Baseline | Page Nodes planos + cosine/TF-IDF | Control experimental |
| Tree local | `hybrid_tree_search.py` | Simulacion interpretable sin API |
| PageIndex real | Hybrid Tree Search oficial | Propuesta final |

El flujo recomendado:

```text
query del estudiante
-> metadata filter
-> candidate problem selection
-> tree search
-> reranking pedagogico
-> contexto recuperado
```

Metricas:

```text
precision@k
recall@k
MRR
node_type_coverage
problem_hit_rate
strategy_match_rate
latency
```

### 12.7 Fase 6: analisis de ideas del estudiante con GPT

Esta es la fase mas importante para tu tesis.

Objetivo:

```text
Analizar una idea escrita por el estudiante y convertirla en senales pedagogicas.
```

Entrada:

```text
problem_id
problem_statement
retrieved_context
student_idea
attempt_history opcional
```

Salida:

```json
{
  "detected_approach": "BINARY_SEARCH",
  "reasoning_stage": "HYPOTHESIS",
  "idea_quality": "PARTIAL",
  "risk_type": ["WRONG_PROOF", "EDGE_CASES"],
  "missing_concepts": ["sufficiency proof"],
  "next_hint_level": 2,
  "feedback": "Tu idea de acotar la busqueda va bien, pero aun falta justificar..."
}
```

Este modulo debe comparar:

```text
heuristica actual vs GPT analyzer vs ground truth manual
```

Metricas:

```text
accuracy de approach
accuracy de reasoning_stage
F1 de risk_type
agreement con evaluador humano
calidad de hint
```

### 12.8 Fase 7: recomendador adaptativo

Objetivo:

```text
Recomendar el siguiente problema o el siguiente nodo de ayuda.
```

Usa:

```text
perfil del estudiante
debilidades detectadas
historial de intentos
analisis GPT de ideas
retrieval tree search
dificultad objetivo
```

Salida:

```text
problem_id
recommendation_score
reason
target_skill
expected_difficulty
suggested_hint_policy
```

La razon debe ser explicable:

```text
Recomendado porque el estudiante falla en pruebas de suficiencia y este problema contiene una prueba constructiva en dificultad cercana.
```

### 12.9 Fase 8: evaluacion experimental

Objetivo:

```text
Probar que la propuesta mejora algo medible.
```

Comparaciones:

```text
Flat Page Nodes
Flat Page Nodes + metadata
Tree local
Tree local + GPT idea analysis
PageIndex Hybrid Tree Search
```

Ground truth inicial:

```text
10 queries de prueba
10 ideas de estudiante
10 recomendaciones esperadas
```

Tablas necesarias:

```text
retrieval_results_by_method.csv
student_idea_eval.csv
recommendation_eval.csv
ablation_summary.csv
```

### 12.10 Fase 9: demo final

Objetivo:

```text
Tener una ejecucion unica para exposicion en vivo.
```

Comando esperado:

```powershell
python scripts/run_full_thesis_demo.py
```

Debe mostrar:

```text
1. dataset cargado;
2. problema seleccionado;
3. idea del estudiante;
4. analisis GPT;
5. nodos recuperados;
6. recomendacion;
7. graficos;
8. explicacion final.
```

### 12.11 Orden recomendado de desarrollo

No implementes todo al mismo tiempo. El orden correcto es:

```text
1. Fase 2: contrato y calidad del dataset.
2. Fase 3: GPT estructura 3 problemas en arbol JSON.
3. Fase 4: convertir esos JSON en tree_nodes/tree_chunks.
4. Fase 6: GPT analiza ideas de estudiante para esos mismos 3 problemas.
5. Fase 5: comparar retrieval plano vs tree local.
6. Fase 7: recomendacion adaptativa usando el analisis GPT.
7. Fase 8: metricas y ablation.
8. Fase 9: demo final.
```

El minimo producto funcional para la siguiente iteracion es:

```text
3 problemas Codeforces con editorial real
3 arboles generados por GPT
6 ideas simuladas de estudiantes
comparacion heuristica vs GPT
retrieval tree local
dashboard con resultados
```

Ese alcance es pequeno, defendible y suficiente para validar si el enfoque de arbol + analisis de ideas tiene sentido antes de escalar a cientos de problemas.
