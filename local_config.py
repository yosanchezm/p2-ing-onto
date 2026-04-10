"""
local_config.py — Configuración compartida para ejecución local.
Reemplaza las celdas de google.colab en todos los notebooks P2.

Las rutas se resuelven automáticamente relativas a este archivo,
por lo que funciona sin importar dónde esté clonado el proyecto.

PRIMER USO (una sola vez por colaborador):
    1. cp env.example .env   y rellena tus API keys
    2. python3 -m venv venv
    3. venv/bin/pip install -r requirements.txt
    4. venv/bin/python -m ipykernel install --user --name rag-p2 --display-name "Python (RAG P2)"

Uso en notebooks (ya inyectado por patch_notebooks_local.py):
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    from local_config import *
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Rutas relativas al repo (funcionan para cualquier colaborador) ─────────────
BASE_DIR   = Path(__file__).parent.resolve()
CORPUS_DIR = BASE_DIR / 'corpus'
INDEX_DIR  = BASE_DIR / 'indices_p2'
TTL_PATH   = BASE_DIR / 'notebooks' / 'biomedical_ontology.ttl'

# Crear directorios si no existen
CORPUS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# ── Cargar .env (cada colaborador tiene el suyo, no se versiona) ──────────────
env_file = BASE_DIR / '.env'
if env_file.exists():
    load_dotenv(env_file)
else:
    print(f'⚠️  No se encontró .env en {BASE_DIR}')
    print('   Crea el archivo: cp env.example .env  y rellena tus claves')

# ── Variables de entorno ───────────────────────────────────────────────────────
GOOGLE_API_KEY    = os.getenv('GOOGLE_API_KEY', '')
GROQ_API_KEY      = os.getenv('GROQ_API_KEY', '')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY', '')
TAVILY_API_KEY    = os.getenv('TAVILY_API_KEY', '')

os.environ['GOOGLE_API_KEY']       = GOOGLE_API_KEY
os.environ['GROQ_API_KEY']         = GROQ_API_KEY
os.environ['LANGCHAIN_API_KEY']    = LANGCHAIN_API_KEY
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2', 'true')
os.environ['LANGCHAIN_PROJECT']    = os.getenv('LANGCHAIN_PROJECT', 'RAG-KnowledgeGraph-P2')
os.environ['TAVILY_API_KEY']       = TAVILY_API_KEY

# ── GraphDB ───────────────────────────────────────────────────────────────────
GRAPHDB_BASE = os.getenv('GRAPHDB_BASE', 'http://localhost:7200')
REPO_NAME    = os.getenv('GRAPHDB_REPO', 'biomed-kg')
GRAPHDB_URL  = f'{GRAPHDB_BASE}/repositories/{REPO_NAME}'


def check_setup():
    """Verifica que todas las claves y dependencias estén configuradas."""
    import requests
    ok = True
    print('🔍 Verificando configuración local...\n')

    keys = {
        'GOOGLE_API_KEY':    GOOGLE_API_KEY,
        'GROQ_API_KEY':      GROQ_API_KEY,
        'LANGCHAIN_API_KEY': LANGCHAIN_API_KEY,
        'TAVILY_API_KEY':    TAVILY_API_KEY,
    }
    for name, val in keys.items():
        if val and val not in ('', 'tu_clave_aqui'):
            print(f'  ✅ {name}')
        else:
            print(f'  ❌ {name} — falta en .env')
            ok = False

    print()
    pdfs = list(CORPUS_DIR.glob('*.pdf'))
    print(f'  {"✅" if len(pdfs) >= 50 else "⚠️ "} Corpus: {len(pdfs)} PDFs en corpus/')
    if len(pdfs) < 50:
        print(f'     → Ejecuta: python3 download_corpus.py')

    faiss_ok = (INDEX_DIR / 'faiss_semantic').exists()
    print(f'  {"✅" if faiss_ok else "⚠️ "} Índice FAISS: '
          f'{"encontrado" if faiss_ok else "no construido — ejecuta P2_01_semantic_indexing.ipynb"}')

    try:
        r = requests.get(f'{GRAPHDB_BASE}/rest/repositories', timeout=3)
        repos = [rep['id'] for rep in r.json()]
        if REPO_NAME in repos:
            print(f'  ✅ GraphDB: repo "{REPO_NAME}" activo')
        else:
            print(f'  ⚠️  GraphDB: corriendo pero repo "{REPO_NAME}" no existe')
            print(f'     Repos disponibles: {repos}')
            print(f'     Ejecuta: python3 setup_graphdb.py')
            ok = False
    except Exception as e:
        print(f'  ❌ GraphDB: no responde en {GRAPHDB_BASE}')
        ok = False

    print()
    if ok:
        print('✅ Todo listo para ejecutar los notebooks')
    else:
        print('⚠️  Revisa los ítems marcados antes de continuar')
    return ok


if __name__ == '__main__':
    check_setup()
