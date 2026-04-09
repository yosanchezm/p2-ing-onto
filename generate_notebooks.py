"""
Generador del notebook unificado del sistema RAG.
Ejecutar: py -3 generate_notebooks.py
Produce: notebooks/RAG_BioMed_Completo.ipynb
"""

import json, os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'notebooks')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def notebook(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": [], "toc_visible": True}
        },
        "cells": cells
    }

def md(src):
    return {"cell_type": "markdown", "id": f"md{abs(hash(src))%99999:05d}",
            "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None,
            "id": f"cd{abs(hash(src))%99999:05d}",
            "metadata": {}, "outputs": [], "source": src}

# =============================================================================
# CÉLULAS DEL NOTEBOOK UNIFICADO
# =============================================================================

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# PORTADA Y TABLA DE CONTENIDOS
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("""# Sistema RAG Biomédico — Trabajo Práctico 1
**Ingeniería Ontológica – 3010090 · Universidad Nacional de Colombia, Medellín**
Profesor: Jaime Alberto Guzmán Luna

---

## Descripción
Sistema de Recuperación Aumentada por Generación (RAG) sobre un corpus de
**50 artículos científicos de arXiv** en el dominio de **Bioinformática y Medicina**.

## Tabla de contenidos
| Sección | Componente |
|---|---|
| [0 · Setup](#setup) | Dependencias, Drive, API keys, modelos |
| [1 · Corpus](#corpus) | Descarga de 50 papers de arXiv |
| [2 · Indexación](#indexacion) | Chunking, embeddings, FAISS |
| [3 · Clasificación](#clasificacion) | Intención del usuario (Gemini) |
| [4 · Recuperación](#recuperacion) | Búsqueda semántica + k dinámico (Groq) |
| [5 · Generación](#generacion) | Respuesta con citas (Gemini) |
| [6 · Verificación](#verificacion) | Grounding / coherencia / completitud (Gemini) |
| [7 · RAG Completo](#rag) | LangGraph + 6 Tools + 10 casos de uso |

## Asignación de LLMs
| Tarea | LLM | Justificación |
|---|---|---|
| Clasificación de consultas | **Gemini 2.0 Flash** | Comprensión contextual profunda; superior en seguimiento de instrucciones complejas y razonamiento semántico. |
| k dinámico en recuperación | **Groq LLaMA 3.3 70B** | Decisión simple y directa; Groq ofrece latencia <500 ms, ideal para micro-decisiones. |
| Generación de respuesta | **Gemini 2.0 Flash** | Síntesis de texto largo y coherente con citas; maneja mejor contextos extensos. |
| Verificación / Crítica | **Gemini 2.0 Flash** | Razonamiento crítico multi-criterio; detecta alucinaciones sutiles con mayor precisión. |
| Consultas generales (sin RAG) | **Groq LLaMA 3.3 70B** | Velocidad en respuestas directas sin necesidad de procesar contexto. |
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 0 · SETUP
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("## 0 · Setup <a id='setup'></a>\nInstala dependencias, monta Drive y configura todas las credenciales y modelos que usarán el resto de las secciones."),

code("""# ── Instalar todas las dependencias ──────────────────────────────────────
!pip install -q \\
    arxiv==2.1.3 \\
    langchain==0.3.14 \\
    langchain-community==0.3.14 \\
    langchain-google-genai==2.0.8 \\
    langchain-groq==0.2.3 \\
    langgraph==0.2.70 \\
    faiss-cpu==1.9.0 \\
    pymupdf==1.25.2 \\
    pydantic==2.10.4
print('Dependencias instaladas')
"""),

code("""# ── Montar Google Drive ────────────────────────────────────────────────────
from google.colab import drive, userdata
drive.mount('/content/drive')

import os, json, time
from typing import TypedDict, Literal

# Rutas del proyecto (todas las secciones las comparten)
BASE_DIR   = '/content/drive/MyDrive/RAG_BioMed'
CORPUS_DIR = f'{BASE_DIR}/corpus'
INDEX_DIR  = f'{BASE_DIR}/faiss_index'
MANIFEST   = f'{BASE_DIR}/manifest.json'

os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR,  exist_ok=True)
print(f'Drive montado. Base: {BASE_DIR}')
"""),

code("""# ── API keys (configurar en Colab: panel izq. > icono llave > Secrets) ────
GEMINI_KEY = userdata.get('GEMINI_API_KEY')
GROQ_KEY   = userdata.get('GROQ_API_KEY')

# Modelos
GEMINI_MODEL = 'gemini-2.0-flash'
GROQ_MODEL   = 'llama-3.3-70b-versatile'
EMBED_MODEL  = 'models/text-embedding-004'

# Hiperparámetros de indexación
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
MAX_REINTENTOS = 3
"""),

code("""# ── Inicializar LLMs y embeddings (se reutilizan en todo el notebook) ────
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# Gemini — generación y razonamiento complejo
gemini     = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_KEY, temperature=0.2)
gemini_det = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_KEY, temperature=0.0)

# Groq — velocidad en decisiones simples
groq_llm = ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_KEY, temperature=0.0)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GEMINI_KEY)

print('Modelos inicializados:')
print(f'  Gemini  : {GEMINI_MODEL}')
print(f'  Groq    : {GROQ_MODEL}')
print(f'  Embed   : {EMBED_MODEL}')
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 · CORPUS
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("""## 1 · Construcción del Corpus <a id='corpus'></a>
Descarga **50 artículos científicos** de arXiv sobre bioinformática, genómica
y diagnóstico médico con IA. Los PDFs se guardan en Drive junto con un
manifiesto JSON de metadatos.

**Fuente:** arXiv.org (acceso abierto)
**Formato de archivo:** `doc_NN_<arxiv_id>.pdf`
"""),

code("""import arxiv

def descargar_corpus(consulta: str, n: int = 50,
                     directorio: str = CORPUS_DIR) -> list[dict]:
    \"\"\"
    Descarga hasta `n` artículos de arXiv y guarda PDFs + metadatos.

    Args:
        consulta   : Cadena de búsqueda para arXiv.
        n          : Número máximo de artículos.
        directorio : Carpeta de destino en Drive.

    Returns:
        Lista de dicts con metadatos de cada artículo.
    \"\"\"
    client = arxiv.Client(page_size=10, delay_seconds=3.0)
    search = arxiv.Search(
        query=consulta,
        max_results=n,
        sort_by=arxiv.SortCriterion.Relevance
    )
    metadatos = []
    for i, paper in enumerate(client.results(search)):
        arxiv_id = paper.entry_id.split('/')[-1].replace('/', '_')
        filename = f'doc_{i+1:02d}_{arxiv_id}.pdf'
        filepath = os.path.join(directorio, filename)

        if not os.path.exists(filepath):
            try:
                paper.download_pdf(dirpath=directorio, filename=filename)
                estado = 'descargado'
            except Exception as e:
                print(f'  [{i+1}] ERROR: {e}')
                continue
        else:
            estado = 'ya existe'

        print(f'[{i+1:02d}/{n}] {estado:10s}  {paper.title[:65]}')
        metadatos.append({
            'doc_id'    : f'doc_{i+1:02d}',
            'filename'  : filename,
            'filepath'  : filepath,
            'titulo'    : paper.title,
            'autores'   : [a.name for a in paper.authors[:5]],
            'anio'      : paper.published.year,
            'arxiv_id'  : arxiv_id,
            'categorias': paper.categories,
            'abstract'  : paper.summary,
        })
        time.sleep(0.5)
    return metadatos
"""),

code("""# ── Ejecutar descarga ──────────────────────────────────────────────────────
CONSULTA_ARXIV = (
    'bioinformatics machine learning genomics medical diagnosis '
    'deep learning protein structure drug discovery'
)

print(f'Descargando 50 articulos: {CONSULTA_ARXIV[:70]}...\\n')
corpus_meta = descargar_corpus(CONSULTA_ARXIV, n=50)

# Guardar manifiesto
with open(MANIFEST, 'w', encoding='utf-8') as f:
    json.dump(corpus_meta, f, ensure_ascii=False, indent=2)

print(f'\\nTotal descargados : {len(corpus_meta)}')
print(f'Manifiesto        : {MANIFEST}')
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 · INDEXACIÓN
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("""## 2 · Consumo e Indexación <a id='indexacion'></a>
Carga los PDFs, los divide en *chunks* con solapamiento y genera el índice
vectorial FAISS. Cada chunk conserva metadatos completos para las citas posteriores.

| Parámetro | Valor |
|---|---|
| `chunk_size` | 1 000 tokens |
| `chunk_overlap` | 200 tokens |
| Modelo de embeddings | `text-embedding-004` (Google) |
| Base vectorial | FAISS (local, guardada en Drive) |
"""),

code("""from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def cargar_documentos(manifest_path: str, corpus_dir: str) -> list[Document]:
    \"\"\"
    Carga todos los PDFs del corpus enriqueciendo sus metadatos con:
    doc_id, titulo, autores, anio, arxiv_id.
    Estos campos se preservan en cada chunk para permitir citas precisas.
    \"\"\"
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    todos = []
    for entrada in manifest:
        filepath = os.path.join(corpus_dir, entrada['filename'])
        if not os.path.exists(filepath):
            print(f'  No encontrado: {entrada[\"filename\"]}')
            continue
        try:
            paginas = PyMuPDFLoader(filepath).load()
            for pag in paginas:
                pag.metadata.update({
                    'doc_id'  : entrada['doc_id'],
                    'titulo'  : entrada['titulo'],
                    'autores' : ', '.join(entrada['autores'][:3]),
                    'anio'    : entrada['anio'],
                    'arxiv_id': entrada['arxiv_id'],
                    'filename': entrada['filename'],
                })
            todos.extend(paginas)
            print(f'  {entrada[\"doc_id\"]}  {len(paginas):3d} pags.  {entrada[\"titulo\"][:55]}')
        except Exception as e:
            print(f'  ERROR en {entrada[\"doc_id\"]}: {e}')
    print(f'\\nTotal paginas cargadas: {len(todos)}')
    return todos

def segmentar(docs: list[Document]) -> list[Document]:
    \"\"\"
    Divide las páginas en chunks y asigna chunk_id único (doc_id_chunk_NNN).
    El chunk_id es esencial para la trazabilidad de citas en la respuesta.
    \"\"\"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=['\\n\\n', '\\n', '. ', ' ', '']
    )
    chunks  = splitter.split_documents(docs)
    conteo: dict[str, int] = {}
    for ch in chunks:
        did = ch.metadata.get('doc_id', 'unk')
        conteo[did] = conteo.get(did, 0) + 1
        ch.metadata['chunk_id'] = f'{did}_chunk_{conteo[did]:03d}'
    print(f'{len(docs)} paginas -> {len(chunks)} chunks  '
          f'(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})')
    return chunks

def crear_indice(chunks: list[Document]) -> FAISS:
    \"\"\"
    Genera embeddings con Google text-embedding-004 y construye el índice FAISS.
    El índice se persiste en Drive para evitar re-indexar en cada sesión.
    \"\"\"
    print(f'Generando embeddings para {len(chunks)} chunks...')
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_DIR)
    print(f'Indice FAISS guardado: {vs.index.ntotal} vectores -> {INDEX_DIR}')
    return vs
"""),

code("""# ── Pipeline de indexación ─────────────────────────────────────────────────
print('=== INDEXACION ===\\n')
docs_raw = cargar_documentos(MANIFEST, CORPUS_DIR)
chunks   = segmentar(docs_raw)
vs       = crear_indice(chunks)

# Verificación rápida
test_docs = vs.similarity_search('deep learning protein structure', k=2)
for d in test_docs:
    m = d.metadata
    print(f'  [{m[\"doc_id\"]}] p.{m.get(\"page\",0)+1}  {d.page_content[:80]}...')
"""),

code("""# ── (Re)cargar índice desde Drive en sesiones posteriores ─────────────────
# Ejecutar esta celda si el índice ya existe y se quiere evitar re-indexar.
def cargar_indice() -> FAISS:
    \"\"\"Carga el índice FAISS previamente guardado en Drive.\"\"\"
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    print(f'Indice cargado: {vs.index.ntotal} vectores')
    return vs

# vs = cargar_indice()   # <-- descomentar si el índice ya existe
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 · CLASIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("""## 3 · Clasificación de Consultas <a id='clasificacion'></a>
Identifica la **intención** del usuario en una de cuatro categorías y decide
si se requiere acceso al corpus.

**LLM:** Gemini 2.0 Flash — elegido por su capacidad de comprensión contextual
profunda y razonamiento semántico para distinguir matices entre categorías.
"""),

code("""from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class ClasificacionConsulta(BaseModel):
    \"\"\"Salida estructurada del clasificador de intención.\"\"\"
    categoria: Literal['busqueda', 'resumen', 'comparacion', 'general'] = Field(
        description='Tipo de consulta: busqueda | resumen | comparacion | general'
    )
    requiere_rag: bool = Field(
        description='True si necesita búsqueda en la base vectorial'
    )
    doc_ids_mencionados: list[str] = Field(
        default=[],
        description='IDs de documentos mencionados explícitamente (e.g. doc_01)'
    )
    razonamiento: str = Field(description='Justificación de la clasificación')

# Cadena de clasificación (Prompt | Gemini | PydanticParser)
_clf_parser = PydanticOutputParser(pydantic_object=ClasificacionConsulta)
_clf_prompt = ChatPromptTemplate.from_messages([
    ('system', \"\"\"Clasifica consultas de un sistema RAG con corpus de 50 papers
bioinformaticos/medicos de arXiv.

Categorias:
- busqueda   : el usuario pide datos o hechos especificos de los papers
- resumen    : pide resumir uno o varios papers del corpus
- comparacion: quiere contrastar metodos, papers o enfoques
- general    : pregunta de conocimiento general (no requiere corpus)

{format_instructions}\"\"\"),
    ('human', 'Consulta: {query}')
])
_clf_chain = _clf_prompt | gemini_det | _clf_parser

def clasificar_consulta(query: str) -> ClasificacionConsulta:
    \"\"\"
    Clasifica la consulta del usuario con Gemini.

    Args:
        query : Pregunta del usuario en lenguaje natural.

    Returns:
        ClasificacionConsulta con categoría, flag RAG y razonamiento.
    \"\"\"
    return _clf_chain.invoke({
        'query': query,
        'format_instructions': _clf_parser.get_format_instructions()
    })
"""),

code("""# ── Pruebas de clasificación ───────────────────────────────────────────────
consultas_test = [
    ('¿Qué redes neuronales predicen la estructura de proteinas?',      'busqueda'),
    ('Resume el paper doc_01',                                           'resumen'),
    ('Compara doc_02 y doc_05 en terminos de metodologia',               'comparacion'),
    ('¿Que es el ADN?',                                                  'general'),
    ('Lista los papers sobre drug discovery del corpus',                 'busqueda'),
]

print(f'  {\"CONSULTA\":<55} {\"CATEGORIA\":<14} {\"RAG\"}')
print('-' * 80)
for query, esperado in consultas_test:
    r = clasificar_consulta(query)
    ok = 'OK' if r.categoria == esperado else 'DIFF'
    print(f'  {query:<55} {r.categoria:<14} {str(r.requiere_rag):<6} {ok}')
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 · RECUPERACIÓN
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("""## 4 · Búsqueda Semántica (Recuperación) <a id='recuperacion'></a>
Recupera los fragmentos más relevantes con **k dinámico**: el número de
documentos a recuperar se decide por un LLM según la complejidad de la consulta.

**LLM para k:** Groq LLaMA 3.3 70B — elegido por su latencia ultra-baja (<500 ms),
suficiente para esta micro-decisión sin necesidad de razonamiento profundo.
"""),

code("""def determinar_k(query: str, tipo: str) -> int:
    \"\"\"
    Usa Groq para decidir cuántos fragmentos (k) recuperar según la
    complejidad de la consulta. Rango: [3, 15].

    Groq se elige por velocidad: es una decisión simple que no requiere
    razonamiento profundo, y la latencia baja mejora la experiencia del usuario.

    Args:
        query : Consulta del usuario.
        tipo  : Categoría de la consulta (busqueda, resumen, comparacion).

    Returns:
        Entero k en [3, 15].
    \"\"\"
    resp = groq_llm.invoke(
        f'Tipo de consulta: \"{tipo}\". Consulta: \"{query[:100]}\"\\n'
        f'¿Cuantos fragmentos (3-15) necesito recuperar?\\n'
        f'Guia: busqueda simple=3-5, resumen=5-8, comparacion=6-10, amplia=10-15.\\n'
        f'Responde SOLO el numero entero.'
    )
    try:
        return max(3, min(int(resp.content.strip()), 15))
    except ValueError:
        return 5

def recuperar(query: str, tipo: str) -> list[Document]:
    \"\"\"
    Recupera fragmentos relevantes con k dinámico y añade similarity_score
    a los metadatos de cada documento para la trazabilidad.

    Args:
        query : Consulta del usuario.
        tipo  : Tipo de consulta para determinar k.

    Returns:
        Lista de Documents con metadatos enriquecidos.
    \"\"\"
    k = determinar_k(query, tipo)
    print(f'  k dinamico: {k}')
    docs_scores = vs.similarity_search_with_score(query, k=k)
    resultados  = []
    for doc, score in docs_scores:
        doc.metadata['similarity_score'] = round(float(score), 4)
        resultados.append(doc)
        m = doc.metadata
        print(f'  [{m[\"doc_id\"]}] p.{m.get(\"page\",0)+1}  '
              f'score={score:.3f}  {doc.page_content[:70]}...')
    return resultados
"""),

code("""# ── Prueba de recuperación ─────────────────────────────────────────────────
q = '¿Que metodos de deep learning se aplican al diagnostico de cancer?'
print(f'Query: {q}\\n')
docs_rec = recuperar(q, 'busqueda')
print(f'\\n{len(docs_rec)} fragmentos recuperados')
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 · GENERACIÓN
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("""## 5 · Generación de Respuesta <a id='generacion'></a>
Construye la respuesta citando los fragmentos recuperados con referencias
numéricas `[1]`, `[2]`, etc. Los metadatos (`doc_id`, `página`, `chunk_id`)
se preservan para la trazabilidad completa.

**LLM:** Gemini 2.0 Flash — elegido por su capacidad de síntesis de texto largo
y coherente, manejo de contextos extensos y seguimiento de instrucciones de formato.
"""),

code("""def formatear_contexto(docs: list[Document]) -> tuple[str, list[dict]]:
    \"\"\"
    Construye el bloque de contexto etiquetado [N] para el prompt de generación
    y la lista de citas con metadatos completos.

    Args:
        docs : Documentos recuperados.

    Returns:
        (contexto_formateado, lista_de_citas)
    \"\"\"
    partes, citas = [], []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        cabecera = (f'[{i}] {m.get(\"titulo\",\"\")[:60]}  |  '
                    f'doc_id={m.get(\"doc_id\")}  '
                    f'p.{m.get(\"page\",0)+1}  '
                    f'chunk={m.get(\"chunk_id\",\"\")}')
        partes.append(f'--- Fragmento {i} ---\\n{cabecera}\\n{doc.page_content}')
        citas.append({
            'num'    : i,
            'doc_id' : m.get('doc_id'),
            'titulo' : m.get('titulo', '')[:70],
            'autores': m.get('autores', ''),
            'pagina' : m.get('page', 0) + 1,
            'chunk_id': m.get('chunk_id', ''),
            'score'  : m.get('similarity_score', 0.0),
        })
    return '\\n\\n'.join(partes), citas

_gen_prompt = ChatPromptTemplate.from_messages([
    ('system', \"\"\"Eres un asistente experto en bioinformatica y medicina basado en
50 papers cientificos de arXiv.

INSTRUCCIONES:
1. Responde UNICAMENTE con informacion del contexto proporcionado.
2. Incluye citas numericas [1], [2], etc. al referenciar cada fragmento.
3. Si la informacion no esta en el contexto, escribe: \"No encontrado en el corpus\".
4. Mantén tono academico y preciso.

CONTEXTO:
{contexto}\"\"\"),
    ('human', '{query}')
])
_gen_chain = _gen_prompt | gemini

def generar_respuesta(query: str, docs: list[Document],
                      sugerencias: str = '') -> dict:
    \"\"\"
    Genera la respuesta con citas a partir del contexto recuperado.

    Args:
        query       : Pregunta del usuario.
        docs        : Documentos recuperados por la búsqueda semántica.
        sugerencias : Feedback del verificador para mejorar (opcional).

    Returns:
        Dict con 'respuesta', 'citas' y 'contexto_completo'.
    \"\"\"
    contexto, citas = formatear_contexto(docs)
    extra = f'\\n\\nMEJORA SOLICITADA POR VERIFICADOR: {sugerencias}' if sugerencias else ''
    resp  = _gen_chain.invoke({'query': query + extra, 'contexto': contexto[:6000]})
    return {'respuesta': resp.content, 'citas': citas, 'contexto_completo': contexto}
"""),

code("""# ── Demo de generación ─────────────────────────────────────────────────────
q_gen  = '¿Como se usa el deep learning para diagnosticar enfermedades con imagenes medicas?'
d_gen  = vs.similarity_search(q_gen, k=5)
r_gen  = generar_respuesta(q_gen, d_gen)

print('=== RESPUESTA ===')
print(r_gen['respuesta'])
print('\\n=== CITAS ===')
for c in r_gen['citas']:
    print(f'  [{c[\"num\"]}] {c[\"doc_id\"]} | p.{c[\"pagina\"]} | {c[\"titulo\"][:55]}')
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6 · VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("""## 6 · Verificación / Crítica <a id='verificacion'></a>
Evalúa automáticamente la respuesta según tres criterios (escala 0–1).
Si alguno no supera **0.7**, se solicita regeneración (máximo 3 intentos).

| Criterio | Descripción |
|---|---|
| **Grounding** | ¿Cada afirmación está respaldada por el contexto? |
| **Coherencia** | ¿Es lógica y libre de alucinaciones? |
| **Completitud** | ¿Responde completamente la pregunta? |

**LLM:** Gemini 2.0 Flash — elegido por su razonamiento crítico multi-criterio
y mayor capacidad para detectar alucinaciones sutiles frente a Groq.
"""),

code("""class ResultadoVerificacion(BaseModel):
    \"\"\"Salida estructurada del agente verificador.\"\"\"
    aprobado            : bool
    puntuacion_grounding   : float = Field(ge=0.0, le=1.0)
    puntuacion_coherencia  : float = Field(ge=0.0, le=1.0)
    puntuacion_completitud : float = Field(ge=0.0, le=1.0)
    problemas   : list[str] = []
    sugerencias : str = ''

_ver_parser = PydanticOutputParser(pydantic_object=ResultadoVerificacion)
_ver_prompt = ChatPromptTemplate.from_messages([
    ('system', \"\"\"Verifica la calidad de la respuesta RAG biomedica.

CRITERIOS (0.0 - 1.0):
1. GROUNDING   : fraccion de afirmaciones respaldadas por el contexto
2. COHERENCIA  : ausencia de contradicciones y alucinaciones
3. COMPLETITUD : que tan completa es la respuesta a la pregunta

REGLA: aprobado = True si y solo si los tres criterios >= 0.7

{format_instructions}

CONTEXTO RECUPERADO:
{contexto}\"\"\"),
    ('human', 'PREGUNTA:\\n{query}\\n\\nRESPUESTA A EVALUAR:\\n{respuesta}')
])
_ver_chain = _ver_prompt | gemini_det | _ver_parser

def verificar_respuesta(query: str, respuesta: str,
                        contexto: str) -> ResultadoVerificacion:
    \"\"\"
    Evalúa grounding, coherencia y completitud de la respuesta generada.

    Args:
        query     : Pregunta original del usuario.
        respuesta : Respuesta generada por el módulo de generación.
        contexto  : Contexto de fragmentos usado para generar la respuesta.

    Returns:
        ResultadoVerificacion con puntuaciones, veredicto y sugerencias.
    \"\"\"
    return _ver_chain.invoke({
        'query'   : query,
        'respuesta': respuesta,
        'contexto': contexto[:3500],
        'format_instructions': _ver_parser.get_format_instructions()
    })
"""),

code("""# ── Demo de verificación ───────────────────────────────────────────────────
ver = verificar_respuesta(q_gen, r_gen['respuesta'], r_gen['contexto_completo'])
print(f'Aprobado    : {ver.aprobado}')
print(f'Grounding   : {ver.puntuacion_grounding:.2f}')
print(f'Coherencia  : {ver.puntuacion_coherencia:.2f}')
print(f'Completitud : {ver.puntuacion_completitud:.2f}')
if ver.problemas:
    print(f'Problemas   : {ver.problemas}')
if ver.sugerencias:
    print(f'Sugerencias : {ver.sugerencias[:150]}')
"""),
]

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7 · RAG COMPLETO CON LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────
cells += [
md("""## 7 · Sistema RAG Completo con LangGraph <a id='rag'></a>
Integra todos los componentes anteriores en un **grafo de ejecución** con
transiciones predefinidas (no autónomas).

```
START → [clasificar]
            │ requiere_rag=True          │ requiere_rag=False
            ▼                             ▼
        [recuperar]              [respuesta_directa] → END
            │
        [generar]
            │
        [verificar]
           / \\
    pass /   \\ fail (intento < 3)
        ▼     ▼
       END  [generar]  ← loop controlado
```
"""),

code("""from langchain_core.tools import tool

# ═══════════════════════════════════════════════════════════════════════════
# TOOLS  (se registran en el grafo y pueden usarse de forma autónoma)
# ═══════════════════════════════════════════════════════════════════════════

# Cargar manifiesto para las tools que lo necesitan
with open(MANIFEST, 'r', encoding='utf-8') as f:
    _manifest_list = json.load(f)
_manifest_idx = {d['doc_id']: d for d in _manifest_list}

@tool
def buscar_documentos(query: str, k: int = 5) -> str:
    \"\"\"
    Busqueda semantica en el corpus biomedico (50 papers de arXiv).
    Retorna los fragmentos mas relevantes con metadatos completos.
    \"\"\"
    resultados = []
    for doc, score in vs.similarity_search_with_score(query, k=k):
        m = doc.metadata
        resultados.append({
            'doc_id'  : m.get('doc_id'),
            'titulo'  : m.get('titulo', '')[:70],
            'pagina'  : m.get('page', 0) + 1,
            'chunk_id': m.get('chunk_id'),
            'score'   : round(float(score), 4),
            'texto'   : doc.page_content[:400],
        })
    return json.dumps(resultados, ensure_ascii=False, indent=2)

@tool
def resumir_documento(doc_id: str) -> str:
    \"\"\"Resume el contenido de un paper especifico del corpus dado su doc_id.\"\"\"
    info = _manifest_idx.get(doc_id, {})
    if not info:
        return f'Documento {doc_id} no encontrado.'
    docs = vs.similarity_search(info.get('titulo', doc_id), k=6,
                                filter={'doc_id': doc_id})
    if not docs:
        return f'Sin fragmentos para {doc_id}.'
    ctx  = '\\n---\\n'.join(d.page_content for d in docs[:5])
    resp = gemini.invoke(
        f'Resume este articulo cientifico en maximo 250 palabras.\\n'
        f'Titulo: {info.get(\"titulo\",\"\")}\\n\\n{ctx[:4000]}'
    )
    return resp.content

@tool
def comparar_documentos(doc_ids: str, aspecto: str = 'metodologia') -> str:
    \"\"\"
    Compara dos o mas papers del corpus segun un aspecto.
    doc_ids debe ser una cadena con IDs separados por coma (e.g. 'doc_01,doc_03').
    \"\"\"
    ids = [i.strip() for i in doc_ids.split(',')]
    if len(ids) < 2:
        return 'Se necesitan al menos 2 doc_ids.'
    textos = {}
    for did in ids:
        docs = vs.similarity_search(aspecto, k=4, filter={'doc_id': did})
        if docs:
            titulo = _manifest_idx.get(did, {}).get('titulo', did)
            textos[did] = f'TITULO: {titulo}\\n' + '\\n'.join(d.page_content for d in docs[:3])
    if len(textos) < 2:
        return 'No se recuperaron suficientes fragmentos.'
    ctx  = '\\n\\n===\\n\\n'.join(f'[{k}]:\\n{v}' for k, v in textos.items())
    resp = gemini.invoke(f'Compara estos papers respecto a \"{aspecto}\":\\n\\n{ctx[:5000]}')
    return resp.content

@tool
def listar_documentos(anio: int = 0) -> str:
    \"\"\"Lista todos los papers del corpus. Filtra por anio si anio > 0.\"\"\"
    docs = _manifest_list if anio == 0 else [d for d in _manifest_list if d.get('anio') == anio]
    if not docs:
        return f'Sin documentos del anio {anio}.'
    return '\\n'.join(
        f'{d[\"doc_id\"]}: {d[\"titulo\"][:65]}  ({d[\"anio\"]})  '
        f'- {d[\"autores\"][0] if d[\"autores\"] else \"N/A\"}'
        for d in docs
    )

@tool
def buscar_por_autor(nombre: str) -> str:
    \"\"\"Busca papers del corpus cuyos autores contengan `nombre` (no sensible a mayusculas).\"\"\"
    hits = [d for d in _manifest_list
            if any(nombre.lower() in a.lower() for a in d.get('autores', []))]
    if not hits:
        return f'Sin papers de \"{nombre}\" en el corpus.'
    return '\\n'.join(
        f'{d[\"doc_id\"]}: {d[\"titulo\"][:65]}  - {\", \".join(d[\"autores\"][:3])}'
        for d in hits
    )

@tool
def obtener_metadata(doc_id: str) -> str:
    \"\"\"Retorna metadatos completos de un paper: titulo, autores, anio, arxiv_id, abstract.\"\"\"
    info = _manifest_idx.get(doc_id)
    if not info:
        return f'{doc_id} no encontrado.'
    return json.dumps({
        'doc_id'  : info['doc_id'],
        'titulo'  : info['titulo'],
        'autores' : info['autores'],
        'anio'    : info['anio'],
        'arxiv_id': info['arxiv_id'],
        'abstract': info.get('abstract', '')[:500],
    }, ensure_ascii=False, indent=2)

TOOLS = [buscar_documentos, resumir_documento, comparar_documentos,
         listar_documentos, buscar_por_autor, obtener_metadata]
print(f'{len(TOOLS)} tools registradas: {[t.name for t in TOOLS]}')
"""),

code("""# ── Estado del grafo ─────────────────────────────────────────────────────
class RAGState(TypedDict):
    query            : str
    tipo_consulta    : str   # busqueda | resumen | comparacion | general
    requiere_rag     : bool
    docs_recuperados : list  # fragmentos serializados para trazabilidad
    contexto         : str   # texto formateado para el prompt
    respuesta        : str
    verificacion     : dict
    intento          : int
    respuesta_final  : str
    traza            : dict  # log completo de cada nodo
"""),

code("""# ── Nodos del grafo ───────────────────────────────────────────────────────
def nodo_clasificar(state: RAGState) -> RAGState:
    \"\"\"Nodo 1: Clasifica la intención con Gemini (comprensión contextual profunda).\"\"\"
    r = clasificar_consulta(state['query'])
    return {**state,
            'tipo_consulta': r.categoria,
            'requiere_rag' : r.requiere_rag,
            'traza': {**state.get('traza', {}),
                      'clasificacion': {'categoria': r.categoria,
                                        'requiere_rag': r.requiere_rag,
                                        'razonamiento': r.razonamiento,
                                        'llm': GEMINI_MODEL}}}

def nodo_recuperar(state: RAGState) -> RAGState:
    \"\"\"Nodo 2: k dinámico con Groq + búsqueda semántica FAISS.\"\"\"
    docs = recuperar(state['query'], state['tipo_consulta'])
    docs_ser, partes = [], []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        docs_ser.append({'contenido': doc.page_content, 'metadata': {
            'doc_id': m.get('doc_id'), 'titulo': m.get('titulo','')[:70],
            'autores': m.get('autores',''), 'pagina': m.get('page',0)+1,
            'chunk_id': m.get('chunk_id',''),
            'similarity_score': m.get('similarity_score', 0.0)}})
        cabecera = (f'[{i}] {m.get(\"titulo\",\"\")[:55]} | '
                    f'{m.get(\"doc_id\")} p.{m.get(\"page\",0)+1} | '
                    f'chunk={m.get(\"chunk_id\",\"\")}')
        partes.append(f'--- Fragmento {i} ---\\n{cabecera}\\n{doc.page_content}')
    return {**state,
            'docs_recuperados': docs_ser,
            'contexto': '\\n\\n'.join(partes),
            'traza': {**state.get('traza', {}),
                      'recuperacion': {
                          'num_docs': len(docs_ser),
                          'docs': [f'{d[\"metadata\"][\"doc_id\"]} (score={d[\"metadata\"][\"similarity_score\"]})'
                                   for d in docs_ser],
                          'llm_k': GROQ_MODEL}}}

def nodo_generar(state: RAGState) -> RAGState:
    \"\"\"Nodo 3: Genera respuesta con citas usando Gemini (síntesis larga y coherente).\"\"\"
    sug = state.get('verificacion', {}).get('sugerencias', '')
    docs_obj = [Document(page_content=d['contenido'], metadata=d['metadata'])
                for d in state['docs_recuperados']]
    result = generar_respuesta(state['query'], docs_obj, sugerencias=sug)
    return {**state,
            'respuesta': result['respuesta'],
            'traza': {**state.get('traza', {}),
                      'generacion': {'intento': state.get('intento', 1),
                                     'llm': GEMINI_MODEL,
                                     'longitud': len(result['respuesta'])}}}

def nodo_verificar(state: RAGState) -> RAGState:
    \"\"\"Nodo 4: Verifica grounding/coherencia/completitud con Gemini (razonamiento crítico).\"\"\"
    ver = verificar_respuesta(state['query'], state['respuesta'], state['contexto'])
    ver_dict = {'aprobado': ver.aprobado,
                'grounding': ver.puntuacion_grounding,
                'coherencia': ver.puntuacion_coherencia,
                'completitud': ver.puntuacion_completitud,
                'problemas': ver.problemas,
                'sugerencias': ver.sugerencias}
    return {**state,
            'verificacion': ver_dict,
            'respuesta_final': state['respuesta'] if ver.aprobado
                               else state.get('respuesta_final', ''),
            'traza': {**state.get('traza', {}),
                      'verificacion': {**ver_dict, 'llm': GEMINI_MODEL}}}

def nodo_respuesta_directa(state: RAGState) -> RAGState:
    \"\"\"Nodo 5: Responde sin corpus con Groq (velocidad para preguntas generales).\"\"\"
    resp = groq_llm.invoke(state['query'])
    return {**state,
            'respuesta': resp.content,
            'respuesta_final': resp.content,
            'traza': {**state.get('traza', {}),
                      'respuesta_directa': {'llm': GROQ_MODEL, 'sin_rag': True}}}
"""),

code("""from langgraph.graph import StateGraph, START, END

# ── Enrutadores condicionales ──────────────────────────────────────────────
def enrutar_clasificacion(state: RAGState) -> str:
    \"\"\"Dirige a recuperacion (RAG) o respuesta directa.\"\"\"
    return 'recuperar' if state['requiere_rag'] else 'respuesta_directa'

def enrutar_verificacion(state: RAGState) -> str:
    \"\"\"Aprueba la respuesta o solicita regeneracion (max MAX_REINTENTOS).\"\"\"
    if state['verificacion']['aprobado']:
        return 'aprobado'
    if state.get('intento', 1) >= MAX_REINTENTOS:
        state['respuesta_final'] = state['respuesta']   # forzar salida
        return 'aprobado'
    state['intento'] = state.get('intento', 1) + 1
    return 'regenerar'

# ── Construir grafo ────────────────────────────────────────────────────────
def construir_grafo():
    \"\"\"
    Ensambla el grafo LangGraph del sistema RAG con transiciones predefinidas.

    Nodos: clasificar, recuperar, generar, verificar, respuesta_directa
    Aristas fijas: recuperar->generar->verificar, respuesta_directa->END
    Aristas condicionales:
        clasificar  -> [recuperar | respuesta_directa]
        verificar   -> [END | generar]  (loop controlado)
    \"\"\"
    wf = StateGraph(RAGState)
    wf.add_node('clasificar',        nodo_clasificar)
    wf.add_node('recuperar',         nodo_recuperar)
    wf.add_node('generar',           nodo_generar)
    wf.add_node('verificar',         nodo_verificar)
    wf.add_node('respuesta_directa', nodo_respuesta_directa)

    wf.add_edge(START,       'clasificar')
    wf.add_edge('recuperar', 'generar')
    wf.add_edge('generar',   'verificar')
    wf.add_edge('respuesta_directa', END)

    wf.add_conditional_edges('clasificar', enrutar_clasificacion,
        {'recuperar': 'recuperar', 'respuesta_directa': 'respuesta_directa'})
    wf.add_conditional_edges('verificar', enrutar_verificacion,
        {'aprobado': END, 'regenerar': 'generar'})

    return wf.compile()

app = construir_grafo()
print('Grafo LangGraph compilado')
"""),

code("""# ── Visualizar el grafo ───────────────────────────────────────────────────
try:
    from IPython.display import Image, display
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print(f'Visualizacion no disponible en este entorno: {e}')
"""),

code("""def ejecutar_rag(query: str, verbose: bool = True) -> dict:
    \"\"\"
    Punto de entrada principal del sistema RAG.

    Ejecuta el grafo completo desde la clasificacion hasta la verificacion
    y retorna el estado final con respuesta, citas y traza completa.

    Args:
        query   : Pregunta del usuario en lenguaje natural.
        verbose : Si True, imprime trazabilidad y respuesta final.

    Returns:
        Estado final RAGState con todos los campos poblados.
    \"\"\"
    estado: RAGState = {
        'query': query, 'tipo_consulta': '', 'requiere_rag': True,
        'docs_recuperados': [], 'contexto': '', 'respuesta': '',
        'verificacion': {}, 'intento': 1, 'respuesta_final': '', 'traza': {}
    }
    resultado = app.invoke(estado)

    if verbose:
        sep = '=' * 65
        print(f'\\n{sep}')
        print('TRAZABILIDAD')
        print(sep)
        for nodo, info in resultado['traza'].items():
            print(f'\\n  [{nodo.upper()}]')
            for k, v in info.items():
                s = str(v)
                print(f'    {k:<22}: {s[:110]}{\"...\" if len(s)>110 else \"\"}')

        if resultado.get('docs_recuperados'):
            print(f'\\n{sep}')
            print('DOCUMENTOS RECUPERADOS')
            print(sep)
            for d in resultado['docs_recuperados']:
                m = d['metadata']
                print(f'  {m[\"doc_id\"]} | p.{m[\"pagina\"]} | '
                      f'score={m[\"similarity_score\"]}  {m[\"titulo\"][:50]}')

        ver = resultado.get('verificacion', {})
        if ver:
            print(f'\\n{sep}')
            print(f'VERIFICACION: aprobado={ver[\"aprobado\"]}  '
                  f'grounding={ver.get(\"grounding\",0):.2f}  '
                  f'coherencia={ver.get(\"coherencia\",0):.2f}  '
                  f'completitud={ver.get(\"completitud\",0):.2f}')

        print(f'\\n{sep}')
        print('RESPUESTA FINAL')
        print(sep)
        print(resultado['respuesta_final'])

    return resultado
"""),

md("""### Casos de Uso (10 consultas de demostración)
Ejecuta cada celda para mostrar el sistema funcionando.
"""),

code("""# CU-01: Búsqueda de método específico
resultado_01 = ejecutar_rag(
    '¿Que metodos de deep learning se usan para predecir la estructura de proteinas?'
)
"""),

code("""# CU-02: Búsqueda de datasets
resultado_02 = ejecutar_rag(
    '¿Cuales son los principales datasets usados para diagnostico de cancer con ML?'
)
"""),

code("""# CU-03: Redes convolucionales en imagenes medicas
resultado_03 = ejecutar_rag(
    '¿Como se aplican las CNN al analisis de imagenes medicas?'
)
"""),

code("""# CU-04: NLP en registros clinicos
resultado_04 = ejecutar_rag(
    '¿Que tecnicas de NLP se usan para extraer informacion de registros clinicos?'
)
"""),

code("""# CU-05: Resumen de un paper
resultado_05 = ejecutar_rag('Resume el paper doc_01')
"""),

code("""# CU-06: Resumen tematico
resultado_06 = ejecutar_rag(
    'Dame un resumen de los papers del corpus sobre drug discovery'
)
"""),

code("""# CU-07: Comparación de documentos
resultado_07 = ejecutar_rag(
    'Compara los enfoques de doc_02 y doc_04 en terminos de metodologia y resultados'
)
"""),

code("""# CU-08: Consulta general (sin RAG)
resultado_08 = ejecutar_rag('¿Que es el aprendizaje por transferencia (transfer learning)?')
"""),

code("""# CU-09: LLMs en genomica
resultado_09 = ejecutar_rag(
    '¿Que modelos de lenguaje se han aplicado a secuencias genomicas?'
)
"""),

code("""# CU-10: Uso directo de tool listar_documentos
print('=== TOOL: listar_documentos ===')
print(listar_documentos.invoke({'anio': 0})[:1000])
print('\\n=== TOOL: buscar_por_autor ===')
print(buscar_por_autor.invoke({'nombre': 'Li'}))
print('\\n=== TOOL: obtener_metadata ===')
print(obtener_metadata.invoke({'doc_id': 'doc_01'}))
"""),
]

# =============================================================================
# GUARDAR NOTEBOOK UNIFICADO
# =============================================================================
output_path = os.path.join(OUTPUT_DIR, 'RAG_BioMed_Completo.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook(cells), f, ensure_ascii=False, indent=1)

print(f'Notebook generado: {output_path}')
print(f'Celdas totales   : {len(cells)}')
print(f'Secciones        : 0-Setup | 1-Corpus | 2-Indexacion | 3-Clasificacion')
print(f'                   4-Recuperacion | 5-Generacion | 6-Verificacion | 7-RAG')
