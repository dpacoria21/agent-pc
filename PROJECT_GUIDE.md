# CP RAG local: guia unica de funcionamiento, comparacion y presentacion

Este proyecto es un prototipo local de RAG educativo para programacion competitiva. Su objetivo actual no es implementar PageIndex Hybrid Tree Search, sino servir como contraste tecnico antes de construir una version real basada en el tutorial de PageIndex:

https://docs.pageindex.ai/tutorials/tree-search/hybrid

Segun ese tutorial, Hybrid Tree Search combina busqueda basada en valor sobre nodos con busqueda guiada por LLM, usa chunks asociados a nodos, agrega scores por nodo, mantiene una cola de nodos unicos y permite que un agente decida si ya tiene suficiente informacion. Este prototipo local todavia no hace eso. Aqui se documenta exactamente que hace, que compara y donde estan sus limitaciones.

## 1. Estructura limpia del proyecto

```text
agent-pc/
  PROJECT_GUIDE.md
  comparison_dashboard.html
  requirements-local.txt
  run_mingw64.sh

  src/
    cp_dataset_scraper.py
    rag_cp_student_profile.py
    search_helpers.py
    search_local.py
    search_faiss.py
    search_chromadb.py

  scripts/
    run_all_local.py
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

## 8. Deficiencias frente a PageIndex Hybrid Tree Search

| Aspecto | Prototipo local actual | PageIndex Hybrid Tree Search esperado |
|---|---|---|
| Estructura | DataFrame plano de Page Nodes | Arbol de nodos |
| Retrieval | Ranking global sobre todos los nodos | Recorrido/seleccion de nodos |
| Vector store | Tiene ejemplos con FAISS y ChromaDB, pero como ranking vectorial plano | Puede usar valor por chunks asociados a nodos |
| Agregacion chunk->node | No existe | Score de nodo agregado desde chunks |
| LLM tree search | No existe | Rama complementaria guiada por LLM |
| Cola de nodos unicos | No existe | Si existe en el enfoque hibrido |
| Early stopping | No existe | El agente puede terminar al reunir suficiente informacion |
| Evaluacion | Ground truth simulado | Requiere evaluacion real por queries y tareas |
| Estudiante | Sesiones simuladas | Debe integrarse con memoria real |

Deficiencias principales del prototipo:

1. La busqueda es plana, no jerarquica.
2. El Page Index local es una segmentacion por secciones, no un tree-search.
3. No hay value prediction por nodo.
4. No hay LLM que explore el arbol.
5. No hay mecanismo de cola ni terminacion temprana.
6. La clasificacion de ideas es heuristica y fragil.
7. Las metricas usan ground truth simulado.
8. No valida soluciones como juez online.

Nota despues de agregar FAISS y ChromaDB:

> Aunque ahora existen ejemplos con FAISS y ChromaDB, eso no convierte el sistema en PageIndex Hybrid Tree Search. FAISS y ChromaDB solo cambian el motor de busqueda vectorial; todavia no hay arbol real, value-based tree search, LLM tree search, cola de nodos ni early stopping.

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

6. Identificar deficiencias frente a PageIndex Hybrid Tree Search:
   - no hay arbol real;
   - no hay value-based tree search;
   - no hay agregacion chunk->node;
   - no hay LLM tree search;
   - no hay cola de nodos unicos;
   - no hay early stopping;
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

> Este prototipo no pretende ser todavia PageIndex Hybrid Tree Search. Es un baseline local interpretable: convierte problemas en Page Nodes, compara retrieval semantico, keyword, hibrido, personalizado, FAISS y ChromaDB, y muestra una deficiencia clara en el analisis de ideas cuando debe distinguir busqueda binaria de formula matematica. Esa brecha justifica la siguiente implementacion con tree-search hibrido real.
