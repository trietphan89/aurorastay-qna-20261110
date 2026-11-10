"""Page-cited hotel-policy retrieval with a native FAISS index and JSON docstore.

Only the four source PDFs enter the index. CSV answers and reference passages
are evaluation evidence and are never loaded by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import math
from typing import Any

# The POC corpus is tiny. Bound native thread pools before native libraries are
# imported: this also avoids the macOS ARM OpenMP crash observed during audit.
for _thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_thread_variable, '1')

EMBED_MODEL = 'HarishMaths/Hotel-Policy-Embedding'
EMBED_REVISION = '3c9f67da8057fb1e9f742c9e6680d5a4b33b4dcd'
DEFAULT_CHUNK_TOKENS = 110
DEFAULT_OVERLAP_TOKENS = 16
POLICIES = {
    'aurorastay_booking_terms_and_cancellation_policy.pdf': 'AS-01',
    'aurorastay_room_and_property_information.pdf': 'AS-02',
    'aurorastay_refunds_and_additional_charges_guide.pdf': 'AS-03',
    'aurorastay_guest_conduct_accessibility_and_resolution_guidelines.pdf': 'AS-04',
}
SCHEMA_VERSION = 'aurorastay-retrieval-v1'
RETRIEVAL_METHOD = 'dense-bm25-rrf-property-dedup-v1'
HYBRID_CONFIG = {'dense_candidates': 32, 'lexical_candidates': 32,
                 'rrf_constant': 60, 'dense_weight': 1.0, 'lexical_weight': 1.5,
                 'bm25_k1': 1.5, 'bm25_b': 0.75, 'lexical_relative_floor': 0.30,
                 'lexical_anchor_min_score': 4.0,
                 'exact_property_boost': 0.035, 'default_k': 4}
_STOP_WORDS = set('a an the i me my we our you your he she they it this that '
                  'is are am was were be been being have has had do does did '
                  'what which when where who how can could would should will '
                  'to of and or for in on at by from with as about there here '
                  'aurorastay aurora stay hotel hotels'.split())
_SYNONYMS = {'dog': ('pet', 'animal'), 'noisy': ('noise', 'quiet'),
             'loud': ('noise', 'quiet'), 'free': ('included',),
             'smoke': ('smoking',), 'vape': ('vaping',),
             'fee': ('charge',), 'budget': ('charge', 'fee')}


def _lexical_tokens(text: str, *, expand: bool = False) -> list[str]:
    words = re.findall(r'[a-z0-9]+', text.lower())
    normalized = []
    for word in words:
        if word in _STOP_WORDS:
            continue
        if word.startswith('cancel'):
            word = 'cancel'
        elif word.startswith('refund'):
            word = 'refund'
        elif word.startswith('park'):
            word = 'park'
        elif word.startswith('charg'):
            word = 'charge'
        elif word.startswith('smok'):
            word = 'smoke'
        elif word.startswith('vap'):
            word = 'vape'
        elif word == 'policies':
            word = 'policy'
        elif word in {'pets', 'fees'} or (word.endswith('s') and len(word) > 4):
            word = word[:-1]
        normalized.append(word)
        if expand:
            normalized.extend(_SYNONYMS.get(word, ()))
    return normalized


def _model_location() -> tuple[str, dict[str, str]]:
    snapshot = os.getenv('AURORASTAY_EMBED_MODEL_PATH')
    if snapshot:
        if Path(snapshot).name != EMBED_REVISION or not Path(snapshot).is_dir():
            raise ValueError('Local embedding snapshot must match the pinned checkpoint revision')
        return snapshot, {}
    return EMBED_MODEL, {'revision': EMBED_REVISION}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    location, kwargs = _model_location()
    return AutoTokenizer.from_pretrained(location, **kwargs)


def token_count(text: str, *, special_tokens: bool = True) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=special_tokens,
                                      truncation=False))


def tokenizer_limit() -> int:
    # The checkpoint encoder has 512 positions, but its published tokenizer uses
    # a 128-token operating limit. Respect the smaller documented limit.
    return min(int(get_tokenizer().model_max_length), 128)


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    location, kwargs = _model_location()
    return HuggingFaceEmbeddings(
        model_name=location,
        model_kwargs={'device': 'cpu', **kwargs},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32},
    )


def _clean_text(text: str) -> str:
    return re.sub(r'[ \t]+', ' ', text).strip()


def _docling_converter():
    # Parsing libraries and layout models are intentionally not imported in the
    # inference path. The backend only needs the saved native index + JSON.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    options = PdfPipelineOptions()
    options.do_ocr = False  # The supplied PDFs contain extractable digital text.
    options.do_table_structure = True
    options.accelerator_options = AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU)
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
    })


def parse_policies(data_dir: str | Path, cache_dir: str | Path) -> dict[str, Any]:
    """Parse PDF layout once, retain Docling JSON, page-level Markdown and tables."""
    from docling_core.types.doc import DoclingDocument
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    converter = None
    documents = {}
    for name, code in POLICIES.items():
        source = data_dir / name
        if not source.is_file():
            raise FileNotFoundError(f'Missing policy PDF: {name}')
        digest = sha256(source)
        cached = cache_dir / f'{source.stem}.docling.json'
        meta = cache_dir / f'{source.stem}.source.json'
        valid_cache = cached.exists() and meta.exists()
        if valid_cache:
            valid_cache = json.loads(meta.read_text()).get('sha256') == digest
        if valid_cache:
            doc = DoclingDocument.model_validate_json(cached.read_text())
        else:
            if converter is None:
                converter = _docling_converter()
            result = converter.convert(source)
            if str(getattr(result.status, 'value', result.status)) != 'success':
                raise RuntimeError(f'Docling conversion did not succeed: {name}')
            doc = result.document
            cached.write_text(doc.model_dump_json())
            _write_json(meta, {'filename': name, 'policy_code': code, 'sha256': digest})
        (cache_dir / f'{source.stem}.md').write_text(doc.export_to_markdown())
        documents[name] = doc
    return documents


def _page_numbers(item) -> list[int]:
    return sorted({int(p.page_no) for p in getattr(item, 'prov', [])})


def _table_rows(markdown: str) -> list[str]:
    """Keep each complete table row with its column names, without pipe padding."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    pipe_lines = [line for line in lines if line.startswith('|')]
    if len(pipe_lines) < 2:
        return [markdown.strip()]
    cells = lambda line: [re.sub(r'\s+', ' ', c).strip() for c in line.strip('|').split('|')]
    header = cells(pipe_lines[0])
    result = []
    for line in pipe_lines[1:]:
        row = cells(line)
        if all(re.fullmatch(r'[:\- ]*', c) for c in row):
            continue
        if not any(row):
            continue
        result.append('; '.join(f'{key}: {value}' for key, value in zip(header, row)
                                if value and value != 'None'))
    return result or [markdown.strip()]


def _section_name(item, current: str) -> str:
    label = str(getattr(getattr(item, 'label', ''), 'value', getattr(item, 'label', '')))
    text = _clean_text(getattr(item, 'text', ''))
    if label in {'section_header', 'title'} and text:
        return text
    if re.match(r'^\d+\. [A-Z]', text) and len(text) < 100:
        return text
    return current


def create_chunks(documents: dict[str, Any], *, chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                  overlap_tokens: int = DEFAULT_OVERLAP_TOKENS):
    """Make token-bounded, page-local chunks without splitting table rows.

    Table rows retain all columns. If the checkpoint budget cannot hold a whole
    row, construction stops instead of silently truncating policy information.
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    limit = tokenizer_limit()
    if not (16 <= overlap_tokens < chunk_tokens <= limit - 2):
        raise ValueError('Require 16 <= overlap < chunk size <= tokenizer limit - 2')
    chunks = []
    report = []
    for name, doc in documents.items():
        section = 'Policy details'
        prose = []
        prose_pages: list[int] = []
        pending_section = section
        last_text = ''
        last_text_pages = []
        sequence = 0

        def append_chunk(body: str, pages: list[int], section_name: str,
                         kind: str, context_note: str = '') -> None:
            nonlocal sequence
            body = body.strip()
            if not body:
                return
            # The heading is useful embedding context. Drop it only when a
            # complete table row otherwise crosses the checkpoint token limit.
            prefix = section_name + '\n'
            candidate = prefix + body
            if token_count(candidate) > limit and kind == 'table_row':
                candidate = body
            count = token_count(candidate)
            if count > limit:
                raise ValueError(f'Unsafe truncation avoided: {name} p{pages} '
                                 f'{kind} has {count} tokens, limit {limit}')
            sequence += 1
            chunk_id = f'{POLICIES[name]}-p{pages[0]:02}-{sequence:03}'
            metadata = {'source': name, 'policy_code': POLICIES[name],
                        'page': pages[0], 'pages': pages, 'section': section_name,
                        'chunk_id': chunk_id, 'kind': kind, 'token_count': count,
                        'effective_date': '2026-01-01', 'review_date': '2027-01-01'}
            if context_note:
                metadata['context_note'] = context_note
            chunks.append(Document(page_content=candidate, metadata=metadata))

        def flush_prose() -> None:
            nonlocal prose, prose_pages
            if not prose:
                return
            heading_budget = token_count(pending_section + '\n', special_tokens=False)
            budget = min(chunk_tokens, limit - 2 - heading_budget)
            if budget <= overlap_tokens:
                raise ValueError(f'Heading is too long for tokenizer: {pending_section}')
            splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                get_tokenizer(), chunk_size=budget,
                chunk_overlap=min(overlap_tokens, budget // 4),
                separators=['\n\n', '\n', '. ', '; ', ', ', ' ', ''],
            )
            for piece in splitter.split_text('\n\n'.join(prose)):
                append_chunk(piece, sorted(set(prose_pages)), pending_section, 'prose')
            prose, prose_pages = [], []

        for item, _level in doc.iterate_items():
            label = str(getattr(getattr(item, 'label', ''), 'value', getattr(item, 'label', '')))
            pages = _page_numbers(item)
            text = _clean_text(getattr(item, 'text', ''))
            if not pages:
                continue
            if max(pages) <= 1:
                # Keep the shared operational controls once, from AS-01 p1.
                # Omit duplicated titles, metadata and document-map navigation.
                if POLICIES[name] == 'AS-01' and text.startswith((
                        'Use the property code,', 'Operating note:')):
                    append_chunk(text, pages, 'How to use this policy', 'prose')
                continue
            if label in {'page_header', 'page_footer', 'picture', 'caption'}:
                continue
            if re.match(r'^AuroraStay Hotels\s*\|', text) or re.fullmatch(r'Page \d+', text):
                continue
            new_section = _section_name(item, section)
            if new_section != section:
                flush_prose()
                section = new_section
                pending_section = section
                last_text = ''
                last_text_pages = []
                continue
            if label == 'table':
                flush_prose()
                rows = _table_rows(item.export_to_markdown(doc=doc))
                for row in rows:
                    append_chunk(row, pages, section, 'table_row',
                                 last_text if last_text_pages == pages else '')
                report.append({'source': name, 'pages': pages, 'section': section,
                               'table_rows': len(rows)})
            elif text:
                if prose_pages and pages != sorted(set(prose_pages)):
                    flush_prose()
                pending_section = section
                prose.append(text)
                prose_pages.extend(pages)
                last_text = text
                last_text_pages = pages
        flush_prose()
    ids = [c.metadata['chunk_id'] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError('Chunk IDs must be unique')
    return chunks, report


def save_index(vectorstore, index_dir: str | Path, *, chunks=None,
               manifest_extra: dict[str, Any] | None = None) -> None:
    """Serialize native FAISS and JSON only; no pickle deserialization required."""
    import faiss
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    docs = []
    for i in range(vectorstore.index.ntotal):
        doc_id = vectorstore.index_to_docstore_id[i]
        doc = vectorstore.docstore.search(doc_id)
        docs.append({'id': doc_id, 'page_content': doc.page_content, 'metadata': doc.metadata})
    faiss.write_index(vectorstore.index, str(index_dir / 'index.faiss'))
    _write_json(index_dir / 'docstore.json', docs)
    manifest = {'schema_version': SCHEMA_VERSION, 'embedding_model': EMBED_MODEL,
                'embedding_revision': EMBED_REVISION, 'tokenizer_limit': tokenizer_limit(),
                'tokenizer_vocab_sha256': hashlib.sha256(json.dumps(
                    get_tokenizer().get_vocab(), sort_keys=True).encode()).hexdigest(),
                'normalization': 'unit L2; inner product equals cosine similarity',
                'faiss_metric': 'inner_product', 'dimension': vectorstore.index.d,
                'chunk_count': vectorstore.index.ntotal,
                'retrieval_method': RETRIEVAL_METHOD, 'hybrid_config': HYBRID_CONFIG,
                'lexical_query_synonyms': _SYNONYMS,
                'files': {n: sha256(index_dir / n) for n in ['index.faiss', 'docstore.json']}}
    if manifest_extra:
        manifest.update(manifest_extra)
    _write_json(index_dir / 'manifest.json', manifest)


def build_corpus(data_dir: str | Path, index_dir: str | Path, *,
                 chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
                 overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
                 cache_dir: str | Path | None = None):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    data_dir, index_dir = Path(data_dir), Path(index_dir)
    cache_dir = Path(cache_dir) if cache_dir else index_dir.parent / 'parsed_policies'
    started = time.perf_counter()
    documents = parse_policies(data_dir, cache_dir)
    chunks, tables = create_chunks(documents, chunk_tokens=chunk_tokens,
                                   overlap_tokens=overlap_tokens)
    vectorstore = FAISS.from_documents(chunks, get_embeddings(),
                                      ids=[c.metadata['chunk_id'] for c in chunks],
                                      distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    manifest_extra = {'chunk_tokens': chunk_tokens, 'overlap_tokens': overlap_tokens,
                      'source_files': {n: sha256(data_dir / n) for n in POLICIES},
                      'source_count': len(POLICIES), 'page_count': sum(len(d.pages) for d in documents.values()),
                      'parser': 'Docling, OCR off, table structure on',
                      'build_elapsed_seconds': round(time.perf_counter() - started, 3)}
    save_index(vectorstore, index_dir, manifest_extra=manifest_extra)
    _write_json(cache_dir / 'chunking_report.json', {
        **manifest_extra, 'chunks_by_source': dict(Counter(c.metadata['source'] for c in chunks)),
        'chunks_by_kind': dict(Counter(c.metadata['kind'] for c in chunks)),
        'max_chunk_tokens_including_special': max(c.metadata['token_count'] for c in chunks),
        'table_extraction': tables,
    })
    return vectorstore


def load_index(index_dir: str | Path):
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.documents import Document
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / 'manifest.json').read_text())
    if (manifest['schema_version'] != SCHEMA_VERSION or
            manifest['embedding_model'] != EMBED_MODEL or
            manifest['embedding_revision'] != EMBED_REVISION):
        raise ValueError('Index embedding model/revision does not match this runtime')
    if (manifest.get('retrieval_method') != RETRIEVAL_METHOD or
            manifest.get('hybrid_config') != HYBRID_CONFIG):
        raise ValueError('Saved retrieval settings do not match the frozen implementation')
    for name, digest in manifest['files'].items():
        if sha256(index_dir / name) != digest:
            raise ValueError(f'Index integrity failure: {name}')
    raw = json.loads((index_dir / 'docstore.json').read_text())
    documents = {item['id']: Document(page_content=item['page_content'], metadata=item['metadata'])
                 for item in raw}
    index = faiss.read_index(str(index_dir / 'index.faiss'))
    if (index.ntotal != len(raw) or index.d != manifest['dimension'] or
            index.metric_type != faiss.METRIC_INNER_PRODUCT):
        raise ValueError('Index/docstore dimensions differ')
    return FAISS(embedding_function=get_embeddings(), index=index,
                 docstore=InMemoryDocstore(documents),
                 index_to_docstore_id={i: item['id'] for i, item in enumerate(raw)},
                 distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)


@lru_cache(maxsize=4)
def _hybrid_data(vectorstore):
    documents = [vectorstore.docstore.search(vectorstore.index_to_docstore_id[i])
                 for i in range(vectorstore.index.ntotal)]
    term_counts = [Counter(_lexical_tokens(d.page_content)) for d in documents]
    document_frequency = Counter(t for count in term_counts for t in count)
    lengths = [sum(count.values()) for count in term_counts]
    average_length = sum(lengths) / len(lengths)
    aliases = {}
    groups = defaultdict(list)
    for doc in documents:
        content = doc.page_content
        row = content.split('\n', 1)[-1]
        codes = set(re.findall(r'ASH-[A-Z]{3}', content))
        if (doc.metadata['kind'] == 'table_row' and
                'property-specific' in doc.metadata['section'].lower() and len(codes) == 1):
            key = 'property:' + next(iter(codes))
            for match in re.finditer(r'ASH-([A-Z]{3})[^;\n]*,\s*([A-Za-z ]+)\s+[A-Z]{2}', row):
                aliases[match.group(2).strip().lower()] = 'ASH-' + match.group(1)
        else:
            key = re.sub(r'\s+', ' ', row).strip().lower()
        groups[key].append(doc)
    representative = {}
    for key, group in groups.items():
        # Prefer the complete row when one PDF repeats only a subset of its
        # columns. Exact facts still come from the cited representative PDF.
        best = sorted(group, key=lambda d: (-len(d.page_content.split('\n', 1)[-1]),
                                            d.metadata['chunk_id']))[0]
        for doc in group:
            representative[doc.metadata['chunk_id']] = (key, best)
    return documents, term_counts, document_frequency, lengths, average_length, aliases, representative


def _validate_query(query: str, k: int, vectorstore) -> None:
    if vectorstore is None:
        raise ValueError('A built or loaded vectorstore is required')
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise ValueError('Query must contain 1-4000 characters')
    if token_count(query.strip()) > tokenizer_limit():
        raise ValueError('Please shorten the question to the 128-token embedding limit')
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError('k must be an integer from 1 to 10')


def retrieve_documents_dense(query: str, k: int = 4, vectorstore=None):
    """Unchanged dense-only baseline for inspecting the hybrid improvement."""
    _validate_query(query, k, vectorstore)
    return vectorstore.similarity_search(query.strip(), k=k)


def retrieve_documents(query: str, k: int = 4, vectorstore=None):
    """Fuse dense/BM25 ranks, keep property identity, and remove duplicate rows.

    This ranking uses source text and the input question only. It never reads
    benchmark answers, expected contexts, or evaluation/test labels. Scores are
    retrieval rankings, not answer-confidence estimates.
    """
    from langchain_core.documents import Document
    _validate_query(query, k, vectorstore)
    (documents, counts, frequencies, lengths, mean_length,
     aliases, representatives) = _hybrid_data(vectorstore)
    dense = vectorstore.similarity_search(query.strip(), k=min(
        HYBRID_CONFIG['dense_candidates'], len(documents)))
    explicit_codes = set(re.findall(r'ASH[- ]([A-Z]{3})\b', query.upper()))
    property_codes = {'ASH-' + code for code in explicit_codes}
    for alias, code in aliases.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', query.lower()):
            property_codes.add(code)
    query_terms = set(_lexical_tokens(query, expand=True))
    lexical = []
    n = len(documents)
    for doc, tf, length in zip(documents, counts, lengths):
        score = 0.0
        for term in query_terms:
            if not tf.get(term):
                continue
            idf = math.log(1 + (n - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            k1, b = HYBRID_CONFIG['bm25_k1'], HYBRID_CONFIG['bm25_b']
            score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * length / mean_length))
        if score > 0:
            lexical.append((score, doc))
    lexical.sort(key=lambda pair: (-pair[0], pair[1].metadata['chunk_id']))
    lexical_scores = {doc.metadata['chunk_id']: score for score, doc in lexical}
    anchor_floor = None
    if lexical:
        floor = lexical[0][0] * HYBRID_CONFIG['lexical_relative_floor']
        if lexical[0][0] >= HYBRID_CONFIG['lexical_anchor_min_score']:
            anchor_floor = floor
        lexical = [pair for pair in lexical if pair[0] >= floor][:HYBRID_CONFIG['lexical_candidates']]
    fused = defaultdict(float)
    rank_metadata = defaultdict(dict)
    candidates = {}
    for method, ranked, weight in [
        ('dense', dense, HYBRID_CONFIG['dense_weight']),
        ('bm25', [pair[1] for pair in lexical], HYBRID_CONFIG['lexical_weight']),
    ]:
        for rank, doc in enumerate(ranked, 1):
            key = doc.metadata['chunk_id']
            fused[key] += weight / (HYBRID_CONFIG['rrf_constant'] + rank)
            rank_metadata[key][method + '_rank'] = rank
            candidates[key] = doc
    # An exact property lookup can be relevant even when a multi-part question
    # spends more words on a different issue such as cancellation or refunds.
    if property_codes:
        for doc in documents:
            if property_codes.intersection(re.findall(r'ASH-[A-Z]{3}', doc.page_content)):
                candidates[doc.metadata['chunk_id']] = doc
                fused[doc.metadata['chunk_id']] += HYBRID_CONFIG['exact_property_boost']
    ranked = sorted(candidates, key=lambda key: (-fused[key], key))
    result, seen = [], set()
    for key in ranked:
        doc = candidates[key]
        doc_codes = set(re.findall(r'ASH-[A-Z]{3}', doc.page_content))
        if property_codes and doc_codes and not doc_codes.intersection(property_codes):
            continue
        # A rare lexical anchor such as a property code or policy term provides
        # a stronger constraint than this checkpoint's high generic cosines.
        # k is a maximum: do not fill spare positions with unmatched dense hits.
        property_match = bool(property_codes.intersection(doc_codes))
        if anchor_floor is not None and lexical_scores.get(key, 0) < anchor_floor and not property_match:
            continue
        group_key, representative = representatives[key]
        if group_key in seen:
            continue
        seen.add(group_key)
        metadata = {**representative.metadata, 'retrieval_method': RETRIEVAL_METHOD,
                    'ranking_score': round(fused[key], 8), **rank_metadata[key]}
        result.append(Document(page_content=representative.page_content, metadata=metadata))
        if len(result) == k:
            break
    return result


def format_chunk(document) -> str:
    m = document.metadata
    citation = f"[{m['policy_code']} p{m['page']} {m['chunk_id']}]"
    note = m.get('context_note', '')
    text = document.page_content
    if note and note not in text:
        text = note + '\n' + text
    return f"{citation}\nSource: {m['source']}\n{text}"


def retrieve(query: str, k: int = 4, vectorstore=None) -> list[str]:
    return [format_chunk(d) for d in retrieve_documents(query, k, vectorstore)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--index-dir', default='artifacts/faiss_index')
    parser.add_argument('--cache-dir', default='build/retrieval/parsed_policies')
    parser.add_argument('--chunk-tokens', type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument('--overlap-tokens', type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()
    result = build_corpus(args.data_dir, args.index_dir, cache_dir=args.cache_dir,
                          chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens)
    print(f'Built {result.index.ntotal} page-cited chunks, dimension {result.index.d}')
