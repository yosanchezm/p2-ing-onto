"""
download_corpus.py — Descarga 50 artículos biomédicos de arXiv como corpus P2.

Usa los mismos términos del dominio biomédico de la P1.
Los PDFs se guardan en corpus/ (reutilizable de P1 si ya existen).

Uso:
    python3 download_corpus.py
"""

import os
import sys
import time
from pathlib import Path

BASE_DIR   = Path(__file__).parent.resolve()
CORPUS_DIR = BASE_DIR / 'corpus'
CORPUS_DIR.mkdir(exist_ok=True)

# ── Verificar si ya existe corpus de P1 ──────────────────────────────────────
existing = list(CORPUS_DIR.glob('*.pdf'))
if len(existing) >= 50:
    print(f'✅ Ya tienes {len(existing)} PDFs en corpus/ — no es necesario descargar')
    sys.exit(0)

print(f'📥 Descargando corpus biomédico ({50 - len(existing)} PDFs necesarios)...')
print(f'   Destino: {CORPUS_DIR}\n')

try:
    import arxiv
except ImportError:
    print('Instalando arxiv...')
    os.system(f'{sys.executable} -m pip install -q arxiv')
    import arxiv

# ── Queries de búsqueda en arXiv ─────────────────────────────────────────────
SEARCH_QUERIES = [
    'BRCA1 BRCA2 breast cancer gene mutation',
    'EGFR lung cancer targeted therapy',
    'COVID-19 SARS-CoV-2 antiviral mechanism',
    'CRISPR gene editing therapeutic applications',
    'immunotherapy checkpoint inhibitor cancer',
    'Alzheimer disease amyloid tau biomarker',
    'diabetes insulin resistance type 2',
    'genomics precision medicine cancer treatment',
    'RNA sequencing single cell transcriptomics',
    'protein structure prediction AlphaFold',
]

client = arxiv.Client(num_retries=3, delay_seconds=2)
downloaded = len(existing)
seen_ids   = {p.stem for p in existing}

for query in SEARCH_QUERIES:
    if downloaded >= 50:
        break

    print(f'🔍 Query: "{query}"')
    search = arxiv.Search(
        query=query,
        max_results=8,
        sort_by=arxiv.SortCriterion.Relevance
    )

    try:
        for result in client.results(search):
            if downloaded >= 50:
                break

            paper_id = result.entry_id.split('/')[-1].replace('.', '_')
            if paper_id in seen_ids:
                continue

            # Nombre del archivo: arxiv_ID.pdf
            filename = CORPUS_DIR / f'arxiv_{paper_id}.pdf'

            try:
                result.download_pdf(dirpath=str(CORPUS_DIR), filename=filename.name)
                seen_ids.add(paper_id)
                downloaded += 1
                print(f'  [{downloaded:02d}/50] {result.title[:70]}')
                time.sleep(1.5)  # respetar rate limit
            except Exception as e:
                print(f'  ⚠️  Error descargando {paper_id}: {e}')
                continue

    except Exception as e:
        print(f'  ⚠️  Error en query "{query}": {e}')
        continue

final_count = len(list(CORPUS_DIR.glob('*.pdf')))
print(f'\n✅ Corpus descargado: {final_count} PDFs en {CORPUS_DIR}')

if final_count < 50:
    print(f'⚠️  Solo se descargaron {final_count}/50 PDFs.')
    print('   Puedes ejecutar este script nuevamente o agregar PDFs manualmente.')
