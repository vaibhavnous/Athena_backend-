"""
Silver Code Generation (POC MODE)

Generates standalone Databricks/Spark scripts from generated bronze metadata and
semantic enrichment. In demo mode, generated bronze scripts are treated as proof
that bronze tables exist.
"""

from __future__ import annotations

import ast
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, TypedDict

from state import Stage01State
from utilis.logger import logger


SILVER_MAX_WORKERS = int(os.environ.get("SILVER_MAX_WORKERS", "4"))


class SilverTableRef(TypedDict):
    database_name: str
    schema_name: str
    table_name: str
    bronze_table: str
    silver_table: str


def _silver_output_dir() -> str:
    return os.path.join(os.getcwd(), "generated_code", "silver")


def _bronze_bundle_path() -> str:
    return os.path.join(os.getcwd(), "generated_code", "bronze", "bronze_scripts.json")


def _silver_readme_path() -> str:
    return os.path.join(_silver_output_dir(), "README.md")


def _silver_ui_path() -> str:
    return os.path.join(_silver_output_dir(), "index.html")


def _validate_python(code: str) -> None:
    compile(code, "<silver_generated>", "exec")
    ast.parse(code)


def _load_bronze_bundle() -> Dict[str, Any]:
    path = _bronze_bundle_path()
    if not os.path.exists(path):
        return {"scripts": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _table_name_from_ref(item: Dict[str, Any]) -> str:
    return str(item.get("table") or item.get("table_name") or "").strip()


def _resolve_tables_for_silver(state: Stage01State) -> List[SilverTableRef]:
    bronze_results = state.get("bronze_generation_results") or _load_bronze_bundle().get("scripts", [])
    bronze_schema = str(state.get("bronze_schema") or "bronze")
    silver_schema = str(state.get("silver_schema") or "silver")
    resolved: List[SilverTableRef] = []

    for item in bronze_results:
        if not isinstance(item, dict):
            continue
        table_name = _table_name_from_ref(item)
        if not table_name:
            continue
        resolved.append(
            {
                "database_name": str(item.get("database_name") or "insurance"),
                "schema_name": str(item.get("schema_name") or "dbo"),
                "table_name": table_name,
                "bronze_table": f"{bronze_schema}.bronze_{table_name}",
                "silver_table": f"{silver_schema}.silver_{table_name}",
            }
        )

    return resolved


def _columns_for_table(enriched_metadata: Dict[str, Any], table_name: str) -> List[Dict[str, Any]]:
    columns = enriched_metadata.get("columns", []) if isinstance(enriched_metadata, dict) else []
    return [
        column
        for column in columns
        if str(column.get("table_name") or "").strip().lower() == table_name.lower()
    ]


def _safe_python_list(values: List[str]) -> str:
    return repr([value for value in values if value])


def _datatype_cast(data_type: str) -> str | None:
    normalized = data_type.lower().strip()
    if normalized in {"int", "integer", "smallint", "tinyint"}:
        return "int"
    if normalized in {"bigint"}:
        return "bigint"
    if normalized in {"float", "real"}:
        return "double"
    if normalized in {"decimal", "numeric", "money", "smallmoney"}:
        return "decimal(38,10)"
    if normalized in {"date"}:
        return "date"
    if normalized in {"datetime", "datetime2", "smalldatetime", "timestamp"}:
        return "timestamp"
    if normalized in {"bit", "boolean"}:
        return "boolean"
    return None


COLUMN_NAME_CORRECTIONS = {
    "rererence_id": "reference_id",
}


def _normalized_column_name(column: Dict[str, Any]) -> str:
    normalized = str(column.get("column_name") or "").strip().lower()
    return COLUMN_NAME_CORRECTIONS.get(normalized, normalized)


def generate_silver_script(
    *,
    table_ref: SilverTableRef,
    enriched_columns: List[Dict[str, Any]],
    run_id: str,
    silver_catalog: str = "main",
    silver_schema: str = "silver",
) -> str:
    table_name = table_ref["table_name"]
    bronze_table = table_ref["bronze_table"]
    silver_table = table_ref["silver_table"]

    source_columns = [_normalized_column_name(column) for column in enriched_columns]
    source_columns = [column for column in source_columns if column]
    string_columns = [
        _normalized_column_name(column)
        for column in enriched_columns
        if str(column.get("data_type") or "").lower() in {"varchar", "nvarchar", "text", "char", "nchar"}
    ]
    pii_columns = [
        _normalized_column_name(column)
        for column in enriched_columns
        if column.get("is_pii_candidate") or column.get("is_pii") or column.get("semantic_type") == "PII"
    ]
    key_columns = [
        _normalized_column_name(column)
        for column in enriched_columns
        if column.get("is_join_key") or str(column.get("semantic_type") or "") in {"ID", "SURROGATE_KEY"}
    ]
    cast_rules = {
        _normalized_column_name(column): _datatype_cast(str(column.get("data_type") or ""))
        for column in enriched_columns
    }
    cast_rules = {key: value for key, value in cast_rules.items() if key and value}
    column_aliases = {
        bad_name: good_name
        for bad_name, good_name in COLUMN_NAME_CORRECTIONS.items()
        if bad_name in {str(column.get("column_name") or "").strip().lower() for column in enriched_columns}
    }

    return f'''
"""
AUTO-GENERATED SILVER TRANSFORMATION SCRIPT

Source table: {bronze_table}
Target table: {silver_table}
Expected runtime: Spark / Databricks with Delta support

POC rule: generated bronze scripts are treated as proof that bronze tables exist.
Runtime checks below still fail clearly if the Databricks table is missing.

DO NOT EDIT MANUALLY
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, current_timestamp, lit, trim, when

spark = SparkSession.builder.getOrCreate()

try:
    spark.sql("CREATE SCHEMA IF NOT EXISTS {silver_schema}")
except Exception:
    print("Could not create schema '{silver_schema}' in the current catalog")

RUN_ID = "{run_id}"
SOURCE_TABLE = "{bronze_table}"
TARGET_TABLE = "{silver_table}"
TEMP_VIEW = "silver_src_{table_name}"

EXPECTED_COLUMNS = {_safe_python_list(source_columns)}
STRING_COLUMNS = {_safe_python_list(string_columns)}
PII_COLUMNS = {_safe_python_list(pii_columns)}
KEY_COLUMNS = {_safe_python_list(key_columns)}
CAST_RULES = {repr(cast_rules)}
COLUMN_ALIASES = {repr(column_aliases)}

if not spark.catalog.tableExists(SOURCE_TABLE):
    raise ValueError(f"Missing bronze source table: {{SOURCE_TABLE}}")

df = spark.table(SOURCE_TABLE)

if df.limit(1).count() == 0:
    raise ValueError(f"Bronze source table has no rows: {{SOURCE_TABLE}}")

available_columns = set(df.columns)
for old_name, new_name in COLUMN_ALIASES.items():
    if old_name in available_columns and new_name not in available_columns:
        df = df.withColumnRenamed(old_name, new_name)

available_columns = set(df.columns)
metadata_columns = [
    name for name in ["run_id", "ingestion_timestamp", "source_system", "source_table"]
    if name in available_columns
]

def compact_name(name):
    return str(name).lower().replace("_", "")

available_by_compact = {{
    compact_name(name): name
    for name in df.columns
}}

if EXPECTED_COLUMNS:
    select_expressions = []
    missing_columns = []
    for expected_name in EXPECTED_COLUMNS:
        actual_name = available_by_compact.get(compact_name(expected_name))
        if actual_name:
            select_expressions.append(col(actual_name).alias(expected_name))
        else:
            missing_columns.append(expected_name)
else:
    select_expressions = [
        col(name)
        for name in df.columns
        if name not in metadata_columns
    ]
    missing_columns = []

if not select_expressions:
    raise ValueError(
        f"No expected business columns found in {{SOURCE_TABLE}}. "
        f"Available columns: {{df.columns}}"
    )

metadata_expressions = [col(name) for name in metadata_columns]
df = df.select(*select_expressions, *metadata_expressions)

if missing_columns:
    print(f"WARNING: Missing expected columns in {{SOURCE_TABLE}}: {{missing_columns}}")

for column_name in STRING_COLUMNS:
    if column_name in df.columns:
        df = df.withColumn(
            column_name,
            when(trim(col(column_name)) == "", None).otherwise(trim(col(column_name)))
        )

for column_name, target_type in CAST_RULES.items():
    if column_name in df.columns:
        df = df.withColumn(column_name, col(column_name).cast(target_type))

for column_name in PII_COLUMNS:
    if column_name in df.columns:
        df = df.withColumn(column_name, col(column_name).cast("string"))

dedup_keys = [column_name for column_name in KEY_COLUMNS if column_name in df.columns]
if dedup_keys:
    df = df.dropDuplicates(dedup_keys)
else:
    df = df.dropDuplicates()

df = (
    df
    .withColumn("silver_run_id", lit(RUN_ID))
    .withColumn("silver_processed_timestamp", current_timestamp())
)

df.createOrReplaceTempView(TEMP_VIEW)

create_table_sql = (
    f"CREATE TABLE IF NOT EXISTS {{TARGET_TABLE}} "
    f"USING DELTA "
    f"AS SELECT * FROM {{TEMP_VIEW}} WHERE 1 = 0"
)
spark.sql(create_table_sql)

(
    df.write
    .format("delta")
    .mode("append")
    .saveAsTable(TARGET_TABLE)
)

print(f"SUCCESS: Silver transformation completed for {{TARGET_TABLE}}")
'''


def _generate_one_table(
    table_ref: SilverTableRef,
    *,
    enriched_metadata: Dict[str, Any],
    run_id: str,
    silver_catalog: str,
    silver_schema: str,
) -> Dict[str, object]:
    table_name = table_ref["table_name"]
    enriched_columns = _columns_for_table(enriched_metadata, table_name)
    code = generate_silver_script(
        table_ref=table_ref,
        enriched_columns=enriched_columns,
        run_id=run_id,
        silver_catalog=silver_catalog,
        silver_schema=silver_schema,
    )
    _validate_python(code)

    output_dir = _silver_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(output_dir, f"silver_transform_{table_name}.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    return {
        "table": table_name,
        "database_name": table_ref["database_name"],
        "schema_name": table_ref["schema_name"],
        "source_table": table_ref["bronze_table"],
        "target_table": table_ref["silver_table"],
        "column_count": len(enriched_columns),
        "status": "APPROVED",
        "script_path": script_path,
    }


def _write_silver_readme(*, results: List[Dict[str, object]], generated_at: str) -> str:
    lines = [
        "# Silver Scripts",
        "",
        f"Generated at: `{generated_at}`",
        f"Script count: `{len(results)}`",
        "",
        "| Source Bronze | Target Silver | Columns | Script | Status |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for item in sorted(results, key=lambda row: str(row.get("table", ""))):
        script_path = str(item.get("script_path") or "")
        script_name = os.path.basename(script_path) if script_path else "-"
        lines.append(
            f"| `{item.get('source_table')}` | `{item.get('target_table')}` | "
            f"`{item.get('column_count', 0)}` | [{script_name}]({script_path}) | `{item.get('status')}` |"
        )

    readme_path = _silver_readme_path()
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return readme_path


def _write_silver_ui(*, results: List[Dict[str, object]], generated_at: str) -> str:
    rows: List[Dict[str, str]] = []
    for item in sorted(results, key=lambda row: str(row.get("table", ""))):
        script_path = str(item.get("script_path") or "")
        script_body = ""
        if script_path and os.path.exists(script_path):
            with open(script_path, "r", encoding="utf-8") as f:
                script_body = f.read()
        rows.append(
            {
                "table": str(item.get("table") or ""),
                "source_table": str(item.get("source_table") or ""),
                "target_table": str(item.get("target_table") or ""),
                "column_count": str(item.get("column_count") or 0),
                "status": str(item.get("status") or "-"),
                "script_body": script_body,
            }
        )

    payload = json.dumps(rows)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Silver Scripts Viewer</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Tahoma, sans-serif; background: #eef4f1; color: #1f2937; }}
    main {{ width: min(1100px, calc(100vw - 32px)); margin: 28px auto; }}
    .hero, .card {{ background: white; border: 1px solid #d9e4df; border-radius: 14px; padding: 18px; margin-bottom: 14px; }}
    input {{ width: 100%; padding: 11px; border: 1px solid #ccd8d3; border-radius: 8px; margin: 12px 0; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; border: 1px solid #d9e4df; border-radius: 8px; padding: 14px; overflow: auto; }}
    .meta {{ color: #667085; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Silver Scripts Viewer</h1>
      <p class="meta">Generated at: {generated_at} | Scripts: {len(rows)}</p>
      <input id="search" type="search" placeholder="Search silver scripts..." />
    </section>
    <section id="list"></section>
  </main>
  <script>
    const rows = {payload};
    const list = document.getElementById("list");
    const search = document.getElementById("search");
    function escapeHtml(value) {{
      return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }}
    function render() {{
      const query = search.value.trim().toLowerCase();
      const filtered = rows.filter((row) => [row.table, row.source_table, row.target_table].join(" ").toLowerCase().includes(query));
      list.innerHTML = filtered.map((row) => `
        <article class="card">
          <h3>${{row.table}}</h3>
          <p class="meta">Source: ${{row.source_table}} | Target: ${{row.target_table}} | Columns: ${{row.column_count}} | Status: ${{row.status}}</p>
          <pre><code>${{escapeHtml(row.script_body)}}</code></pre>
        </article>
      `).join("");
    }}
    search.addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""

    ui_path = _silver_ui_path()
    with open(ui_path, "w", encoding="utf-8") as f:
        f.write(html)
    return ui_path


def silver_code_generation_node(state: Stage01State) -> Stage01State:
    new_state = state.copy()
    table_refs = _resolve_tables_for_silver(state)

    if not table_refs:
        new_state["silver_generation_status"] = "SKIPPED"
        new_state["silver_generation_error"] = "No bronze generation results or bronze bundle found."
        return new_state

    enriched_metadata = (
        state.get("enrichment_review_artifact")
        or state.get("enriched_metadata")
        or {}
    )
    if isinstance(enriched_metadata, dict) and "enrichment_artifact" in enriched_metadata:
        enriched_metadata = enriched_metadata.get("enrichment_artifact") or {}

    run_id = str(state.get("run_id") or "SILVER_POC_RUN_001")
    silver_catalog = str(state.get("silver_catalog") or state.get("bronze_catalog") or "main")
    silver_schema = str(state.get("silver_schema") or "silver")

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=SILVER_MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                _generate_one_table,
                table_ref,
                enriched_metadata=enriched_metadata,
                run_id=run_id,
                silver_catalog=silver_catalog,
                silver_schema=silver_schema,
            )
            for table_ref in table_refs
        ]
        for future in as_completed(futures):
            results.append(future.result())

    generated_at = datetime.utcnow().isoformat()
    bundle = {
        "generated_at": generated_at,
        "script_count": len(results),
        "scripts": results,
    }

    os.makedirs(_silver_output_dir(), exist_ok=True)
    bundle_path = os.path.join(_silver_output_dir(), "silver_scripts.json")
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    readme_path = _write_silver_readme(results=results, generated_at=generated_at)
    ui_path = _write_silver_ui(results=results, generated_at=generated_at)

    new_state["silver_generation_status"] = "COMPLETED"
    new_state["silver_generation_error"] = None
    new_state["silver_generated_at"] = generated_at
    new_state["silver_generation_results"] = results
    new_state["silver_generation_bundle_path"] = bundle_path
    new_state["silver_generation_readme_path"] = readme_path
    new_state["silver_generation_ui_path"] = ui_path
    new_state["status"] = "PIPELINE_COMPLETED"

    logger.info("Silver generation completed: %d scripts", len(results), extra={"run_id": run_id, "node": "silver_generation"})
    return new_state
