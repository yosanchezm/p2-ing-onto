"""
setup_graphdb.py — Configura el repositorio GraphDB y carga la ontología.

Requiere GraphDB corriendo en localhost:7200.
El repositorio detectado actualmente es 'biomed-kg'.

Uso:
    python3 setup_graphdb.py
"""

import requests
from pathlib import Path

BASE_DIR    = Path(__file__).parent.resolve()
TTL_PATH    = BASE_DIR / 'notebooks' / 'biomedical_ontology.ttl'
GRAPHDB_BASE = 'http://localhost:7200'
REPO_NAME   = 'biomed-kg'
GRAPHDB_URL = f'{GRAPHDB_BASE}/repositories/{REPO_NAME}'


def check_graphdb() -> bool:
    """Verifica que GraphDB esté corriendo."""
    try:
        r = requests.get(f'{GRAPHDB_BASE}/rest/repositories', timeout=5)
        repos = [rep['id'] for rep in r.json()]
        print(f'✅ GraphDB corriendo. Repositorios: {repos}')
        return True
    except Exception as e:
        print(f'❌ GraphDB no responde en {GRAPHDB_BASE}: {e}')
        return False


def repo_exists() -> bool:
    try:
        r = requests.get(f'{GRAPHDB_BASE}/rest/repositories', timeout=5)
        repos = [rep['id'] for rep in r.json()]
        return REPO_NAME in repos
    except Exception:
        return False


def create_repo() -> bool:
    """Crea el repositorio biomed-kg si no existe."""
    config = f"""
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rep:  <http://www.openrdf.org/config/repository#> .
@prefix sr:   <http://www.openrdf.org/config/repository/sail#> .
@prefix sail: <http://www.openrdf.org/config/sail#> .
@prefix owlim: <http://www.ontotext.com/trree/owlim#> .

[] a rep:Repository ;
   rep:repositoryID "{REPO_NAME}" ;
   rdfs:label "Biomedical Knowledge Graph" ;
   rep:repositoryImpl [
      rep:repositoryType "graphdb:FreeSailRepository" ;
      sr:sailImpl [
         sail:sailType "graphdb:FreeSail" ;
         owlim:ruleset "owl-horst-optimized" ;
         owlim:storage-folder "storage" ;
      ]
   ] .
"""
    resp = requests.post(
        f'{GRAPHDB_BASE}/rest/repositories',
        files={'config': ('config.ttl', config, 'text/turtle')},
        timeout=15
    )
    if resp.status_code in (200, 201):
        print(f'✅ Repositorio "{REPO_NAME}" creado')
        return True
    elif resp.status_code == 409:
        print(f'ℹ️  Repositorio "{REPO_NAME}" ya existe')
        return True
    else:
        print(f'❌ Error creando repositorio: {resp.status_code} — {resp.text[:300]}')
        return False


def count_triples() -> int:
    try:
        sparql_url = f'{GRAPHDB_URL}?query=SELECT+%28COUNT%28*%29+AS+%3Fcount%29+WHERE+%7B+%3Fs+%3Fp+%3Fo+%7D'
        r = requests.get(sparql_url, headers={'Accept': 'application/sparql-results+json'}, timeout=10)
        data = r.json()
        return int(data['results']['bindings'][0]['count']['value'])
    except Exception:
        return -1


def upload_ontology() -> bool:
    """Sube la ontología TTL a GraphDB."""
    if not TTL_PATH.exists():
        print(f'❌ No se encontró: {TTL_PATH}')
        return False

    with open(TTL_PATH, 'rb') as f:
        content = f.read()

    resp = requests.post(
        f'{GRAPHDB_URL}/statements',
        data=content,
        headers={'Content-Type': 'text/turtle; charset=UTF-8'},
        timeout=30
    )

    if resp.status_code in (200, 204):
        triples = count_triples()
        print(f'✅ Ontología cargada. Triples en repositorio: {triples}')
        return True
    else:
        print(f'❌ Error subiendo ontología: {resp.status_code} — {resp.text[:300]}')
        return False


def main():
    print('🔧 Configurando GraphDB para ejecución local...\n')

    if not check_graphdb():
        print('\n💡 Para iniciar GraphDB:')
        print('   graphdb -d   (daemon mode, si está en PATH)')
        print('   O usa la interfaz gráfica de GraphDB Desktop')
        return

    print()

    if not repo_exists():
        print(f'📦 Creando repositorio "{REPO_NAME}"...')
        if not create_repo():
            return
    else:
        print(f'✅ Repositorio "{REPO_NAME}" ya existe')

    print()
    triples = count_triples()
    if triples > 0:
        print(f'ℹ️  El repositorio ya tiene {triples} triples.')
        resp = input('¿Recargar la ontología de todas formas? [s/N]: ').strip().lower()
        if resp != 's':
            print('✅ Configuración GraphDB completa (sin recarga)')
            return

    print(f'📤 Subiendo ontología: {TTL_PATH.name}...')
    upload_ontology()

    print('\n✅ GraphDB configurado. Prueba en:')
    print(f'   http://localhost:7200/sparql?repository={REPO_NAME}')


if __name__ == '__main__':
    main()
