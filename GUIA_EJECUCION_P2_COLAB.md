# Guía de Ejecución — Práctica 2 KG-RAG en Google Colab
**Ingeniería Ontológica 3010090 — Universidad Nacional de Colombia**

---

## Requisitos previos

| Requisito | Dónde obtenerlo |
|---|---|
| Cuenta Google (con Drive) | — |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) → API Keys |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
| `LANGCHAIN_API_KEY` | [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) → API Keys |
| GraphDB Free | [graphdb.ontotext.com](https://graphdb.ontotext.com) → Download |

---

## PASO 1 · Preparar Google Drive

### 1.1 Crear la estructura de carpetas

En tu Google Drive crea exactamente esta estructura:

```
Mi unidad/
└── RAG_P2/
    ├── corpus/          ← PDFs del corpus (50 artículos de arXiv)
    ├── indices_p2/      ← Se crea automáticamente al ejecutar
    └── biomedical_ontology.ttl
```

### 1.2 Subir los archivos

1. Copia la carpeta `notebooks/` de tu proyecto local
2. Sube `biomedical_ontology.ttl` a `Mi unidad/RAG_P2/`
3. Sube los 50 PDFs del corpus a `Mi unidad/RAG_P2/corpus/`
   - Si ya los tienes del Trabajo 1, simplemente cópialos a esta nueva ruta

---

## PASO 2 · Configurar secretos en Google Colab

> Los secretos permiten usar las API keys sin escribirlas en el código.

1. Abre **Google Colab** → cualquier notebook
2. En el panel izquierdo, haz clic en el icono 🔑 (**Secrets**)
3. Agrega cada secreto con el botón **"Add new secret"**:

| Nombre del secreto | Valor |
|---|---|
| `GOOGLE_API_KEY` | Tu clave de Google AI Studio |
| `GROQ_API_KEY` | Tu clave de Groq |
| `LANGCHAIN_API_KEY` | Tu clave de LangSmith |
| `TAVILY_API_KEY` | Tu clave de Tavily |

4. Activa el toggle **"Notebook access"** para cada secreto

---

## PASO 3 · Configurar GraphDB

GraphDB es la base de datos del Knowledge Graph. Hay dos opciones:

### Opción A — GraphDB en tu máquina local + ngrok (recomendado)

**En tu PC:**

```bash
# 1. Descarga e instala GraphDB Free desde graphdb.ontotext.com
# 2. Inicia GraphDB (por defecto en localhost:7200)

# 3. Instala ngrok (para exponer GraphDB a internet)
# Descarga desde: https://ngrok.com/download

# 4. Expón el puerto 7200 de GraphDB
ngrok http 7200
```

ngrok te dará una URL pública tipo: `https://abc123.ngrok.io`

**En Colab**, cambia esta línea en el notebook:
```python
# Antes (localhost):
GRAPHDB_BASE = 'http://localhost:7200'

# Después (con ngrok):
GRAPHDB_BASE = 'https://abc123.ngrok.io'  # URL que te dio ngrok
```

### Opción B — Omitir GraphDB (modo degradado)

Si no puedes instalar GraphDB, el sistema funciona sin el Knowledge Graph.
Las tools `knowledge_graph_query` devolverán un mensaje de error controlado
y el agente usará solo la búsqueda vectorial.

**Modificación necesaria en `RAG_KnowledgeGraph_Completo.ipynb`:**
```python
# Comentar la subida de la ontología:
# upload_ontology(TTL_PATH, GRAPHDB_URL)
print('⚠️  Modo sin KG — solo búsqueda vectorial activa')
```

---

## PASO 4 · Orden de ejecución

### Opción A — Notebook único (recomendado para entrega)

Abre directamente `RAG_KnowledgeGraph_Completo.ipynb` y ejecuta **todas las celdas en orden**.
Este notebook incluye todos los componentes integrados.

```
Colab → Archivo → Subir notebook → RAG_KnowledgeGraph_Completo.ipynb
Entorno de ejecución → Ejecutar todo (Ctrl+F9)
```

### Opción B — Notebooks individuales (desarrollo modular)

Ejecuta en este orden:

```
1. P2_01_semantic_indexing.ipynb    → Construye el índice FAISS semántico
2. P2_02_query_transformation.ipynb → Prueba HyDE y Query Decomposition
3. P2_03_advanced_retrieval.ipynb   → Prueba recuperación MMR
4. P2_04_react_reflecting.ipynb     → Prueba el agente ReAct + Reflecting
5. P2_05_knowledge_graph.ipynb      → Configura GraphDB y SPARQL
6. P2_06_langsmith_metrics.ipynb    → Evalúa métricas y trazabilidad
```

> **Nota:** El notebook 01 construye el índice FAISS que usan todos los demás.
> Solo es necesario ejecutarlo UNA vez — guarda el índice en Drive.

---

## PASO 5 · Verificar LangSmith

Después de ejecutar cualquier consulta, verifica las trazas:

1. Ve a [smith.langchain.com](https://smith.langchain.com)
2. Selecciona el proyecto **`RAG-KnowledgeGraph-P2`**
3. Verás cada ejecución con:
   - Los nodos del grafo (react_agent → tools → generate → reflect)
   - Las tool calls del agente
   - Tokens usados y latencia por nodo
   - El score de la reflexión

---

## PASO 6 · Verificar GraphDB (si lo tienes activo)

1. Abre `http://localhost:7200` (o la URL de ngrok)
2. Ve a **Repositories** → selecciona `biomed`
3. Ejecuta esta consulta SPARQL de prueba:

```sparql
PREFIX bio: <http://www.unal.edu.co/biomed#>
SELECT ?clase (COUNT(?ind) AS ?total)
WHERE {
    ?ind a ?clase .
    FILTER(STRSTARTS(STR(?clase), 'http://www.unal.edu.co/biomed#'))
}
GROUP BY ?clase
ORDER BY DESC(?total)
```

Deberías ver las 12 clases con sus individuos.

---

## Solución de problemas frecuentes

### Error: `ModuleNotFoundError: No module named 'langchain_experimental'`
```python
!pip install -q langchain-experimental==0.3.4
```

### Error: `Rate limit exceeded` (Google API)
El `SemanticChunker` hace muchas llamadas de embeddings.
Agrega `time.sleep(2)` entre lotes o usa una cuenta con mayor cuota.

### Error: `Connection refused` en GraphDB
- Verifica que GraphDB esté corriendo en tu PC
- Verifica que ngrok esté activo y la URL sea correcta
- Comprueba que el repositorio `biomed` existe en GraphDB

### El índice FAISS ya existe pero da error al cargar
```python
# Forzar reconstrucción borrando el índice:
import shutil
shutil.rmtree('/content/drive/MyDrive/RAG_P2/indices_p2/faiss_semantic', ignore_errors=True)
```

### Colab se desconecta durante la indexación
El índice se guarda en Drive progresivamente. Al reconectar,
el notebook detecta el índice existente y lo carga sin reindexar.

---

## Checklist de entrega

- [ ] `biomedical_ontology.ttl` — Ontología OWL con 12 clases, 14 propiedades, individuos
- [ ] `RAG_KnowledgeGraph_Completo.ipynb` — Notebook con 3 casos de uso ejecutados
- [ ] Captura de LangSmith mostrando trazas de los 3 casos de uso
- [ ] Consultas SPARQL documentadas (SELECT, FILTER, ORDER BY, LIMIT, UPDATE)
- [ ] 5 casos de inferencia documentados en el informe
- [ ] Informe técnico PDF con métricas (Recall@k, MRR, nDCG, Relevance, Faithfulness)

---

## Resumen de rutas en Drive

```
/content/drive/MyDrive/RAG_P2/
├── corpus/                          ← 50 PDFs
├── biomedical_ontology.ttl          ← Ontología OWL
└── indices_p2/
    ├── faiss_semantic/              ← Índice FAISS (auto-generado)
    │   ├── index.faiss
    │   └── index.pkl
    └── semantic_chunks.pkl          ← Chunks semánticos (auto-generado)
```
