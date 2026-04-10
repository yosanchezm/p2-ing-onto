"""
patch_notebooks_local.py
Convierte todos los notebooks P2 de Colab a ejecución local.

Cambios que aplica:
  1. Elimina celdas de !pip install (ya en requirements.txt)
  2. Reemplaza celdas de google.colab (userdata + drive) por local_config
  3. Actualiza rutas /content/drive/MyDrive/RAG_P2 → ruta local
  4. Corrige nombre de repo GraphDB: 'biomed' → 'biomed-kg'

Uso:
    python3 patch_notebooks_local.py
"""

import json
import re
import shutil
from pathlib import Path

BASE_DIR   = Path(__file__).parent.resolve()
NOTEBOOKS  = BASE_DIR / 'notebooks'
LOCAL_PATH = str(BASE_DIR)

# Notebooks P2 a parchear
TARGET_NOTEBOOKS = [
    'P2_01_semantic_indexing.ipynb',
    'P2_02_query_transformation.ipynb',
    'P2_03_advanced_retrieval.ipynb',
    'P2_04_react_reflecting.ipynb',
    'P2_05_knowledge_graph.ipynb',
    'P2_06_langsmith_metrics.ipynb',
    'RAG_KnowledgeGraph_Completo.ipynb',
]

# ── Celda de configuración local que reemplaza google.colab ──────────────────
LOCAL_SETUP_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ── Configuración local (reemplaza google.colab) ──────────────────────────\n",
        "import sys\n",
        "from pathlib import Path\n",
        "# Ruta al repo resuelta desde este notebook (funciona para cualquier colaborador)\n",
        "_REPO_ROOT = Path('__file__').parent.parent if '__file__' in dir() else Path().resolve()\n",
        "# Buscar local_config.py subiendo directorios si hace falta\n",
        "for _p in [_REPO_ROOT, _REPO_ROOT.parent, Path().resolve(), Path().resolve().parent]:\n",
        "    if (_p / 'local_config.py').exists():\n",
        "        _REPO_ROOT = _p\n",
        "        break\n",
        "if str(_REPO_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(_REPO_ROOT))\n",
        "from local_config import (\n",
        "    BASE_DIR, CORPUS_DIR, INDEX_DIR, TTL_PATH,\n",
        "    GRAPHDB_BASE, REPO_NAME, GRAPHDB_URL,\n",
        "    GOOGLE_API_KEY, GROQ_API_KEY, LANGCHAIN_API_KEY, TAVILY_API_KEY\n",
        ")\n",
        "print('✅ Configuración local cargada')\n",
        "print(f'   BASE_DIR:    {BASE_DIR}')\n",
        "print(f'   CORPUS_DIR:  {CORPUS_DIR}')\n",
        "print(f'   GRAPHDB_URL: {GRAPHDB_URL}')\n",
    ]
}


def is_pip_install_cell(source: list[str]) -> bool:
    """Detecta celdas que solo instalan paquetes con pip."""
    joined = ''.join(source)
    return '!pip install' in joined and 'import' not in joined.replace('!pip install', '')


def is_colab_setup_cell(source: list[str]) -> bool:
    """Detecta celdas que usan google.colab o montan Drive."""
    joined = ''.join(source)
    return (
        'google.colab' in joined or
        'drive.mount' in joined or
        "userdata.get" in joined
    )


def has_only_drive_mount(source: list[str]) -> bool:
    """Detecta celdas que SOLO montan Drive."""
    joined = ''.join(source).strip()
    return 'drive.mount' in joined and len(joined) < 200


def patch_source(source: list[str]) -> list[str]:
    """Aplica reemplazos de texto en el source de una celda."""
    joined = ''.join(source)

    # 1. Rutas de Drive → local
    joined = joined.replace(
        '/content/drive/MyDrive/RAG_P2',
        str(BASE_DIR)
    )
    # Variantes sin guión bajo
    joined = joined.replace(
        "Path('/content/drive/MyDrive/RAG_P2')",
        f"Path('{BASE_DIR}')"
    )

    # 2. GraphDB repo name
    joined = re.sub(r"'biomed'(?!\s*-)", "'biomed-kg'", joined)
    joined = joined.replace('"biomed"', '"biomed-kg"')
    # En la URL directa
    joined = joined.replace(
        '/repositories/biomed\n',
        '/repositories/biomed-kg\n'
    )
    joined = joined.replace(
        "/repositories/biomed'",
        "/repositories/biomed-kg'"
    )
    joined = joined.replace(
        '/repositories/biomed"',
        '/repositories/biomed-kg"'
    )

    # 3. Eliminar imports de google.colab (líneas individuales)
    lines = joined.split('\n')
    clean_lines = []
    for line in lines:
        if 'from google.colab import' in line:
            continue  # eliminar
        if 'drive.mount(' in line:
            continue  # eliminar
        # Reemplazar userdata.get('KEY') con os.getenv('KEY')
        line = re.sub(r"userdata\.get\('(\w+)'\)", r"os.getenv('\1', '')", line)
        clean_lines.append(line)
    joined = '\n'.join(clean_lines)

    # Reconvertir a lista de líneas con \n
    result_lines = []
    for line in joined.split('\n'):
        result_lines.append(line + '\n')
    # Quitar el \n final extra
    if result_lines and result_lines[-1] == '\n':
        result_lines[-1] = ''
    return result_lines


def extract_imports_from_colab_cell(source: list[str]) -> list[str] | None:
    """
    Si una celda mezcla google.colab con imports de langchain,
    extrae solo las líneas de import (sin colab, sin rutas, sin os.environ).
    Retorna lista de líneas o None si no hay imports útiles.
    """
    joined = ''.join(source)
    if 'google.colab' not in joined:
        return None
    if 'from langchain' not in joined and 'from langgraph' not in joined and 'from SPARQLWrapper' not in joined:
        return None

    skip_patterns = [
        'google.colab', 'drive.mount', 'userdata',
        'BASE_DIR', 'CORPUS_DIR', 'INDEX_DIR', 'GRAPHDB_BASE',
        'GRAPHDB_URL', 'REPO_NAME', 'os.environ', 'embeddings =',
        'gemini =', 'groq =', 'ls_client', 'BIO =', 'TTL_PATH',
        "print('✅", 'HuggingFaceEmbeddings',
    ]
    lines = joined.split('\n')
    kept = [ln for ln in lines if not any(p in ln for p in skip_patterns)]
    clean = '\n'.join(kept).strip()
    if not clean:
        return None
    return [ln + '\n' for ln in clean.split('\n') if ln.strip()]


MODELS_CELL_SUFFIX = [
    "\n",
    "# ── Modelos ──────────────────────────────────────────────────────────────\n",
    "embeddings = GoogleGenerativeAIEmbeddings(model='models/text-embedding-004', task_type='retrieval_document')\n",
    "gemini = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.2, max_tokens=4096)\n",
    "groq   = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.0, max_tokens=512)\n",
    "BIO    = 'http://www.unal.edu.co/biomed#'\n",
    "\n",
    "print('✅ Imports y modelos listos')\n",
]


def patch_notebook(nb_path: Path) -> bool:
    """Parchea un notebook en su lugar. Devuelve True si se modificó."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    new_cells = []
    inserted_config = False
    modified = False

    for i, cell in enumerate(cells):
        source = cell.get('source', [])
        ctype  = cell.get('cell_type', '')

        if ctype == 'code':
            # Eliminar celdas de solo pip install
            if is_pip_install_cell(source):
                modified = True
                print(f'    [REMOVED] pip install cell')
                continue

            # Reemplazar celdas de google.colab por local_config (solo 1 vez)
            if is_colab_setup_cell(source):
                if not inserted_config:
                    new_cells.append(LOCAL_SETUP_CELL)
                    inserted_config = True
                    modified = True
                    print(f'    [REPLACED] google.colab setup → local_config')
                    # Si la celda también tenía imports de langchain, preservarlos
                    extra_imports = extract_imports_from_colab_cell(source)
                    if extra_imports:
                        imports_cell = {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": [],
                            "source": extra_imports + MODELS_CELL_SUFFIX
                        }
                        new_cells.append(imports_cell)
                        print(f'    [ADDED] imports cell (recuperada del colab cell)')
                else:
                    # Si hay una segunda celda de colab (ej. solo drive.mount), eliminarla
                    modified = True
                    print(f'    [REMOVED] extra colab cell')
                continue

            # Parchear source de la celda (rutas, repo name, etc.)
            new_source = patch_source(source)
            if new_source != source:
                cell['source'] = new_source
                modified = True
            new_cells.append(cell)
        else:
            new_cells.append(cell)

    # Si no hubo celda de colab (raro), insertar config al inicio de todas formas
    if not inserted_config:
        new_cells.insert(1, LOCAL_SETUP_CELL)  # después del título markdown
        modified = True
        print(f'    [INSERTED] local_config at top')

    nb['cells'] = new_cells

    if modified:
        # Backup
        backup = nb_path.with_suffix('.ipynb.bak')
        shutil.copy2(nb_path, backup)

        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f'    ✅ Guardado (backup: {backup.name})')
    else:
        print(f'    ℹ️  Sin cambios necesarios')

    return modified


def main():
    print('🔧 Parchando notebooks P2 para ejecución local...\n')
    patched = 0

    for nb_name in TARGET_NOTEBOOKS:
        nb_path = NOTEBOOKS / nb_name
        if not nb_path.exists():
            print(f'⚠️  No encontrado: {nb_name}')
            continue

        print(f'📓 {nb_name}')
        if patch_notebook(nb_path):
            patched += 1
        print()

    print(f'✅ {patched}/{len(TARGET_NOTEBOOKS)} notebooks parchados')
    print('\nPróximo paso:')
    print('  1. Copia env.example → .env y rellena tus API keys')
    print('  2. pip install -r requirements.txt')
    print('  3. python3 local_config.py   (verifica setup)')
    print('  4. Ejecuta P2_01_semantic_indexing.ipynb para construir el índice')


if __name__ == '__main__':
    main()
