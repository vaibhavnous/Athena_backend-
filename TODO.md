# Comprehensive Fix Plan

## Issue 1: IntegrityError — NULL fingerprint in ai_store
**Root cause:** `utilis/db.py` `ai_store_db_writer()` does NOT insert the `fingerprint` column, but the actual DB table has `fingerprint` as NOT NULL.
**Affected callers:** `nodes/kpi_extraction.py`, `nodes/req_extraction.py`, `nodes/table_nomination.py`, `nodes/hitl.py`

### Fixes:
- [ ] `utilis/db.py`: Add `fingerprint: Optional[str] = None` param to `ai_store_db_writer`, include in INSERT
- [ ] `nodes/kpi_extraction.py`: Pass `fingerprint=fingerprint` to both `ai_store_db_writer` calls
- [ ] `nodes/req_extraction.py`: Pass `fingerprint=fingerprint` to all 3 `ai_store_db_writer` calls
- [ ] `nodes/table_nomination.py`: Pass `fingerprint=fingerprint` to `ai_store_db_writer` call
- [ ] `nodes/hitl.py`: Pass `fingerprint=fingerprint` to both `ai_store_db_writer` calls
- [ ] `add_missing_columns.py`: Add `fingerprint` column to `ai_store` schema definition

## Issue 2: KeyError 'pinecone' / KeyError 'index_name'
**Root cause:** `config` has no `"pinecone"` section. `memory_lookup.py` does `pinecone_conf["index_name"]` which crashes.

### Fixes:
- [ ] `nodes/memory_lookup.py`: Change `pinecone_conf["index_name"]` → `pinecone_conf.get("index_name", "ai-store-index")` (2 occurrences)

## Issue 3: ingestion.py NULL fingerprint (already fixed partially)
- [x] `pinecone_conf = config["pinecone"]` → `config.get("pinecone", {})` (done)
- [x] `pinecone_conf["index_name"]` → `.get()` (done)
- [x] `pinecone_conf["api_key"]` → `.get()` (done)
- [x] `_store_and_register`: `fingerprint = new_state.get("fingerprint") or ""` (done)

