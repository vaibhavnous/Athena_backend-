import argparse
import json
import logging
import os
import sys
import uuid
import warnings
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


def is_dev_mode() -> bool:
    return os.getenv("DEV_MODE", "").strip().lower() in {"1", "true", "yes", "on", "dev"}


def bootstrap_runtime(dev_mode: bool) -> None:
    """Silence noisy third-party libraries unless DEV_MODE is enabled."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["ATHENA_SUPPRESS_CONSOLE"] = "0" if dev_mode else "1"

    if dev_mode:
        return

    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r".*langchain.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r".*langchain.*")
    warnings.filterwarnings("ignore", message=r".*TOKENIZERS_PARALLELISM.*")

    # Suppress embedding model verbose output in non-dev
    if not dev_mode:
        logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.WARNING)

    noisy_loggers = [
        "langchain",
        "langchain_core",
        "langchain_community",
        "langgraph",
        "pinecone",
        "pinecone_plugin_interface",
        "sentence_transformers",
        "transformers",
        "httpx",
        "httpcore",
        "openai",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)


DEV_MODE = is_dev_mode()
bootstrap_runtime(DEV_MODE)
load_dotenv()

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from state import Stage01State
from utilis.db import config, get_connection
from utilis.logger import logger
from utilis.db import get_pending_items, update_hitl_item, get_completed_items
from rich.prompt import Prompt
from rich import print as rprint
from nodes.hitl import hitl_review_node


console = Console()


def list_pending_runs(gate: int = 1) -> List[str]:
    db_schema = config["azure_sql"]["schema_name"]
    conn = get_connection()
    try:
        cursor = conn.cursor()
        if gate == 1:
            cursor.execute(
                f"""
                SELECT DISTINCT run_id
                FROM [{db_schema}].[hitl_review_queue]
                WHERE gate_number = 1 AND gate_status = 'PENDING'
                ORDER BY run_id
                """
            )
        else:
            cursor.execute(
                f"""
                WITH latest_gate2 AS (
                    SELECT
                        run_id,
                        artifact_type,
                        ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY stored_at DESC) AS rn
                    FROM [{db_schema}].[ai_store]
                    WHERE artifact_type IN ('TABLE_NOMINATIONS', 'GATE2_CERTIFIED_TABLES')
                )
                SELECT run_id
                FROM latest_gate2
                WHERE rn = 1 AND artifact_type = 'TABLE_NOMINATIONS'
                ORDER BY run_id
                """
            )
        return [row.run_id for row in cursor.fetchall()]
    finally:
        conn.close()


def fetch_nominated_tables(run_id: str) -> List[Dict[str, Any]]:
    """Fetch nominated tables from ai_store for Gate 2 review."""
    db_schema = config["azure_sql"]["schema_name"]
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT TOP 1 payload
            FROM [{db_schema}].[ai_store]
            WHERE run_id = ? AND artifact_type = 'TABLE_NOMINATIONS'
            ORDER BY stored_at DESC
        """, (run_id,))
        row = cursor.fetchone()
        if not row or not row[0]:
            return []
        payload = json.loads(row[0])
        return payload.get("nominations", [])
    finally:
        conn.close()


def _score_color(score: float) -> str:
    """Return color based on confidence score."""
    if score >= 0.7:
        return "bold green"
    elif score >= 0.4:
        return "bold yellow"
    else:
        return "bold red"


def _score_emoji(score: float) -> str:
    """Return emoji based on confidence score."""
    if score >= 0.7:
        return "🔥"
    elif score >= 0.4:
        return "⭐"
    else:
        return "⚡"


def print_pending_tables(tables: List[Dict[str, Any]]):
    """Beautiful table display for Gate 2 nominations."""
    if not tables:
        console.print(Panel("No nominated tables found.", title="Gate 2", border_style="yellow"))
        return

    # Header panel with stats
    high_conf = sum(1 for t in tables if t.get("confidence_score", 0) >= 0.7)
    med_conf = sum(1 for t in tables if 0.4 <= t.get("confidence_score", 0) < 0.7)
    low_conf = sum(1 for t in tables if t.get("confidence_score", 0) < 0.4)

    console.print(Panel(
        f"[bold cyan]{len(tables)}[/] tables nominated\n"
        f"[green]{high_conf}[/] high confidence | [yellow]{med_conf}[/] medium | [red]{low_conf}[/] low",
        title="🗂️  Table Nominations Ready for Review",
        border_style="bright_blue",
    ))

    table = Table(show_lines=True, border_style="bright_black")
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Score", justify="right", width=8)
    table.add_column("Table", style="cyan")
    table.add_column("Schema", style="white")
    table.add_column("Database", style="white")
    table.add_column("Match Reason", style="magenta")
    table.add_column("Coverage", justify="right", width=8)
    table.add_column("Keywords", style="yellow")

    for idx, t in enumerate(tables, 1):
        score = t.get("confidence_score", 0)
        color = _score_color(score)
        emoji = _score_emoji(score)
        coverage = t.get("coverage_ratio", 0)
        keywords = ", ".join(t.get("matched_keywords", []))[:25] or "-"

        table.add_row(
            str(idx),
            f"[{color}]{score:.4f} {emoji}[/{color}]",
            t.get("table_name", "N/A"),
            t.get("schema_name", "N/A"),
            t.get("database_name", "N/A"),
            t.get("nomination_reason", "N/A")[:35],
            f"{coverage:.0%}",
            keywords,
        )
    console.print(table)


def print_table_detail(t: Dict[str, Any]):
    """Show detailed view of a single table nomination."""
    score = t.get("confidence_score", 0)
    color = _score_color(score)
    emoji = _score_emoji(score)
    keywords = ", ".join(t.get("matched_keywords", [])) or "None"
    coverage = t.get("coverage_ratio", 0)

    console.print(Panel(
        f"[bold {color}]{emoji} {t.get('table_name', 'N/A')}[/{color}]\n\n"
        f"[cyan]Database:[/]    {t.get('database_name', 'N/A')}\n"
        f"[cyan]Schema:[/]      {t.get('schema_name', 'N/A')}\n"
        f"[cyan]Score:[/]       [{color}]{score:.4f}[/{color}]\n"
        f"[cyan]Coverage:[/]    {coverage:.1%} of keywords\n"
        f"[cyan]Reason:[/]      {t.get('nomination_reason', 'N/A')}\n"
        f"[cyan]Keywords:[/]    {keywords}",
        border_style=color.replace("bold ", ""),
        title="Table Detail",
    ))


def print_table_review_summary(approved: List[Dict], rejected: List[Dict]) -> None:
    """Beautiful summary after Gate 2 review."""
    if not approved and not rejected:
        return

    table = Table(
        title="📋 Gate 2 Table Review Summary",
        show_lines=True,
        title_style="bold magenta",
        border_style="bright_black",
    )
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Table", style="cyan")
    table.add_column("Schema", style="white")
    table.add_column("Status", style="white", width=14, justify="center")
    table.add_column("Score", justify="right", width=10)
    table.add_column("Reason", style="white")

    all_items = [(t, "APPROVED") for t in approved] + [(t, "REJECTED") for t in rejected]
    # Sort by score descending
    all_items.sort(key=lambda x: x[0].get("confidence_score", 0), reverse=True)

    for idx, (t, status) in enumerate(all_items, 1):
        score = t.get("confidence_score", 0)
        if status == "APPROVED":
            status_style = "[bold green]✅ APPROVED[/]"
        else:
            status_style = "[bold red]❌ REJECTED[/]"

        table.add_row(
            str(idx),
            t.get("table_name", "N/A"),
            t.get("schema_name", "N/A"),
            status_style,
            f"{score:.4f}",
            t.get("nomination_reason", "N/A")[:30],
        )

    # Summary footer
    approved_score = sum(t.get("confidence_score", 0) for t in approved) / len(approved) if approved else 0
    rejected_score = sum(t.get("confidence_score", 0) for t in rejected) / len(rejected) if rejected else 0

    table.add_section()
    table.add_row(
        "",
        f"[bold]Total: {len(all_items)}[/]",
        "",
        f"[green]{len(approved)}[/] | [red]{len(rejected)}[/]",
        f"[green]μ={approved_score:.3f}[/] / [red]μ={rejected_score:.3f}[/]",
        "",
    )

    console.print("\n")
    console.print(table)
    console.print("\n")


def print_metadata_discovery_summary(result: Dict[str, Any]) -> None:
    discovered = result.get("discovered_metadata") or {}
    tables = discovered.get("tables", []) if isinstance(discovered, dict) else []

    if not tables:
        console.print(Panel("No metadata discovery results available.", title="Metadata Discovery", border_style="yellow"))
        return

    completed = sum(1 for table in tables if table.get("table_status") == "COMPLETED")
    failed = sum(1 for table in tables if table.get("table_status") == "FAILED")

    console.print(Panel(
        f"Status: {result.get('metadata_status', '-')}\n"
        f"Tables considered: {len(tables)}\n"
        f"Completed: {completed}\n"
        f"Failed: {failed}",
        title="Metadata Discovery Summary",
        border_style="bright_blue",
    ))

    table = Table(title="Discovered Tables", show_lines=True, border_style="bright_black")
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Database", style="white")
    table.add_column("Schema", style="white")
    table.add_column("Table", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Columns", justify="right", style="green")

    for idx, item in enumerate(tables, 1):
        status = str(item.get("table_status", "UNKNOWN"))
        status_style = "green" if status == "COMPLETED" else "red" if status == "FAILED" else "yellow"
        table.add_row(
            str(idx),
            str(item.get("database_name", "N/A")),
            str(item.get("schema_name", "N/A")),
            str(item.get("table_name", "N/A")),
            f"[{status_style}]{status}[/{status_style}]",
            str(item.get("column_count", 0)),
        )

    console.print(table)


def print_column_profiling_summary(result: Dict[str, Any]) -> None:
    payload = result.get("column_profiles") or {}
    if not isinstance(payload, dict) or not payload:
        console.print(Panel("No column profiling results available.", title="Column Profiling", border_style="yellow"))
        return

    table_results = payload.get("table_results", [])
    console.print(Panel(
        f"Status: {result.get('column_profiling_status', '-')}\n"
        f"Tables profiled: {payload.get('table_count', 0)}\n"
        f"Columns profiled: {payload.get('columns_profiled', 0)}\n"
        f"Columns failed: {payload.get('columns_failed', 0)}\n"
        f"Table status counts: success={payload.get('tables_success', 0)}, "
        f"partial={payload.get('tables_partial', 0)}, failed={payload.get('tables_failed', 0)}",
        title="Column Profiling Summary",
        border_style="bright_blue",
    ))

    if not table_results:
        return

    table = Table(title="Profiled Tables", show_lines=True, border_style="bright_black")
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("Database", style="white")
    table.add_column("Schema", style="white")
    table.add_column("Table", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Profiled", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Duration", justify="right", style="white")

    for idx, item in enumerate(table_results, 1):
        status = str(item.get("status", "UNKNOWN"))
        status_style = (
            "green" if status == "SUCCESS"
            else "yellow" if status == "PARTIAL"
            else "red" if status == "FAILED"
            else "white"
        )
        duration = float(item.get("duration_seconds", 0.0) or 0.0)
        table.add_row(
            str(idx),
            str(item.get("database_name", "N/A")),
            str(item.get("schema_name", "N/A")),
            str(item.get("table_name", "N/A")),
            f"[{status_style}]{status}[/{status_style}]",
            str(item.get("columns_profiled", 0)),
            str(item.get("columns_failed", 0)),
            f"{duration:.2f}s",
        )

    console.print(table)


def review_tables(run_id: str, cfg: Optional[Dict[str, Any]] = None):
    """Gate 2 — Interactive table nomination review with batch operations."""
    from nodes.hitl import hitl_table_review_node

    tables = fetch_nominated_tables(run_id)
    if not tables:
        console.print(Panel(
            f"No nominated tables found for {run_id}\n"
            f"Expected a TABLE_NOMINATIONS artifact in ai_store.",
            title="Gate 2 Error",
            border_style="red",
        ))
        return False

    print_pending_tables(tables)

    # Batch operation option
    console.print("\n[bold cyan]Batch Options:[/]")
    batch_action = Prompt.ask(
        "[A]pprove all  [R]eject all  [I]ndividual review",
        choices=["A", "R", "I"],
        default="I",
    ).upper()

    approved: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    if batch_action == "A":
        approved = tables[:]
        console.print(f"[green]✅ Approved all {len(tables)} tables[/]")
    elif batch_action == "R":
        rejected = tables[:]
        console.print(f"[red]❌ Rejected all {len(tables)} tables[/]")
    else:
        # Individual review
        for idx, t in enumerate(tables, 1):
            name = t.get("table_name", "N/A")
            score = t.get("confidence_score", 0)
            color = _score_color(score)

            console.print(f"\n[bold]─── Table {idx}/{len(tables)} ───[/]")
            print_table_detail(t)

            action = Prompt.ask(
                f"[{color}]Approve {name}?[/{color}]",
                choices=["A", "R", "S"],  # S = skip (decide later)
                default="A",
            ).upper()

            if action == "A":
                approved.append(t)
                console.print(f"[green]✅ Approved {name}[/]")
            elif action == "R":
                rejected.append(t)
                console.print(f"[red]❌ Rejected {name}[/]")
            else:
                console.print(f"[yellow]⏭️  Skipped {name}[/]")

        # Handle skipped items (default to approve)
        skipped = [t for t in tables if t not in approved and t not in rejected]
        if skipped:
            console.print(f"\n[yellow]⏭️  {len(skipped)} table(s) skipped — defaulting to APPROVE[/]")
            approved.extend(skipped)

    print_table_review_summary(approved, rejected)

    # Confirm before certifying
    if not approved:
        console.print(Panel("No tables approved. Aborting Gate 2.", title="Warning", border_style="red"))
        return False

    if not Confirm.ask(f"Certify {len(approved)} approved table(s)?", default=True):
        console.print("[yellow]Gate 2 aborted by user.[/]")
        return False

    # Resume pipeline with certified tables
    resumed_input = load_checkpoint_state(run_id) or {"run_id": run_id}
    resumed_input["human_table_decision"] = "COMPLETED"
    resumed_input["certified_tables"] = approved
    resumed = hitl_table_review_node(resumed_input)

    if resumed.get("status") == "FAILED":
        console.print(Panel(
            resumed.get("error", "Gate 2 certification failed."),
            title="Gate 2 Error",
            border_style="red",
        ))
        return False

    from nodes.metadata_discovery import metadata_discovery_node
    from nodes.column_profiling import column_profiling_node

    discovered = metadata_discovery_node(resumed)
    profiled = column_profiling_node(discovered)

    console.print(Panel(
        f"[bold green]Gate 2 Complete[/]\n\n"
        f"Status: {profiled.get('column_profiling_status', profiled.get('metadata_status', 'GATE2_COMPLETE'))}\n"
        f"Certified Tables: {len(approved)}\n"
        f"Metadata: {profiled.get('metadata_status', '-')}\n"
        f"Column Profiling: {profiled.get('column_profiling_status', '-')}",
        title="🎉 HITL Table Certification",
        border_style="green",
    ))

    if sys.stdin.isatty():
        if ask_yes_no("Show metadata discovery summary?", default=False):
            print_metadata_discovery_summary(profiled)
        if ask_yes_no("Show column profiling summary?", default=False):
            print_column_profiling_summary(profiled)

    return True


def print_pending_kpis(items: List[Dict[str, Any]]):
    table = Table(title="Pending KPIs for Review", show_lines=True)
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Confidence")
    for item in items:
        kpi = item["kpi"]
        table.add_row(
            item["item_id"][-8:],
            kpi.get("kpi_name", "N/A"),
            kpi.get("kpi_description", "N/A")[:60] + "...",
            f"{kpi.get('ai_confidence_score', 0):.2f}",
        )
    console.print(table)


def print_review_summary(reviewed_items: List[Dict[str, Any]]) -> None:
    """Print a beautiful summary table of HITL review decisions."""
    if not reviewed_items:
        return

    table = Table(
        title="📋 HITL Review Summary",
        show_lines=True,
        title_style="bold magenta",
        border_style="bright_black",
    )
    table.add_column("#", style="dim", width=4, justify="center")
    table.add_column("KPI Name", style="cyan")
    table.add_column("Status", style="white", width=14, justify="center")
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Rejection Reason", style="white")

    for idx, item in enumerate(reviewed_items, 1):
        status = item["status"]
        if status == "APPROVED":
            status_style = "[bold green]✅ APPROVED[/]"
        elif status == "EDITED":
            status_style = "[bold blue]✏️ EDITED[/]"
        elif status == "REJECTED":
            status_style = "[bold red]❌ REJECTED[/]"
        else:
            status_style = status

        reason = item.get("reason", "") or "-"
        if status != "REJECTED":
            reason = "-"

        conf = item.get("confidence", 0)
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)

        table.add_row(
            str(idx),
            item.get("kpi_name", "N/A"),
            status_style,
            conf_str,
            reason,
        )

    # Summary footer
    approved_count = sum(1 for i in reviewed_items if i["status"] == "APPROVED")
    edited_count = sum(1 for i in reviewed_items if i["status"] == "EDITED")
    rejected_count = sum(1 for i in reviewed_items if i["status"] == "REJECTED")

    table.add_section()
    table.add_row(
        "",
        f"[bold]Total: {len(reviewed_items)}[/]",
        f"[green]{approved_count}[/] | [blue]{edited_count}[/] | [red]{rejected_count}[/]",
        "",
        "",
    )

    console.print("\n")
    console.print(table)
    console.print("\n")


def review_run(run_id: str, gate: int = 1, cfg: Optional[Dict[str, Any]] = None):
    if gate == 2:
        return review_tables(run_id, cfg)

    pending = get_pending_items(run_id, gate)
    if not pending:
        console.print(Panel(f"No pending for {run_id}", title="Complete"))
        return True
    
    print_pending_kpis(pending)

    reviewed_items: List[Dict[str, Any]] = []

    for item in pending:
        console.print(f"\n[bold yellow]Review {item['item_id']}[/]")
        kpi = item["kpi"]
        rprint(Panel(f"**{kpi['kpi_name']}**\n{kpi['kpi_description']}\nConf: {kpi['ai_confidence_score']:.2f}", title="KPI"))
        
        action = Prompt.ask("[A]pprove [R]eject [E]dit", choices=["A", "R", "E"], default="A").upper()
        
        if action == "A":
            update_hitl_item(item["item_id"], "APPROVED")
            console.print("[green]✅ Approved[/]")
            reviewed_items.append({
                "kpi_name": kpi.get("kpi_name", "N/A"),
                "status": "APPROVED",
                "confidence": kpi.get("ai_confidence_score", 0),
                "reason": None,
            })
        elif action == "R":
            reason = Prompt.ask("Reason")
            update_hitl_item(item["item_id"], "REJECTED", rejection_reason=reason)
            console.print("[red]❌ Rejected[/]")
            reviewed_items.append({
                "kpi_name": kpi.get("kpi_name", "N/A"),
                "status": "REJECTED",
                "confidence": kpi.get("ai_confidence_score", 0),
                "reason": reason,
            })
        elif action == "E":
            name = Prompt.ask("Name", default=kpi["kpi_name"])
            desc = Prompt.ask("Description", default=kpi["kpi_description"])
            edited = kpi.copy()
            edited["kpi_name"] = name
            edited["kpi_description"] = desc
            update_hitl_item(item["item_id"], "APPROVED", json.dumps(edited))
            console.print("[blue]✏️ Edited/Approved[/]")
            reviewed_items.append({
                "kpi_name": name,
                "status": "EDITED",
                "confidence": kpi.get("ai_confidence_score", 0),
                "reason": None,
            })

    remaining = get_pending_items(run_id, gate)
    if remaining:
        return False

    print_review_summary(reviewed_items)

    certified = get_completed_items(run_id, gate)
    resumed_input = load_checkpoint_state(run_id) or {"run_id": run_id}
    resumed_input["human_decision"] = "COMPLETED"
    resumed_input["certified_kpis"] = [i["kpi"] for i in certified]
    resumed = hitl_review_node(resumed_input)
    console.print(Panel(f"Resumed successfully.\nStatus: {resumed.get('status', 'GATE1_COMPLETE')}", title="HITL Resume"))

    if resumed.get("status") == "FAILED":
        console.print(Panel(resumed.get("error", "Gate 1 certification failed."), title="Gate 1 Error", border_style="red"))
        return False

    from nodes.table_nomination import table_nomination_node

    nominated = table_nomination_node(resumed)
    if nominated.get("status") == "FAILED":
        console.print(Panel(
            nominated.get("table_nomination_error", nominated.get("error", "Table nomination failed.")),
            title="Gate 2 Error",
            border_style="red",
        ))
        return False

    nomination_count = len(nominated.get("nominated_tables", []) or [])
    console.print(Panel(
        f"Gate 1 certified KPIs: {len(certified)}\n"
        f"Gate 2 nominated tables: {nomination_count}",
        title="Pipeline Advanced",
        border_style="green",
    ))

    if sys.stdin.isatty() and nominated.get("human_table_decision") == "PENDING":
        if ask_yes_no("Start Gate 2 table review now?", default=True):
            return review_tables(run_id, cfg)

    console.print(Panel(f"{run_id} CERTIFIED!", title="Gate 1 Complete", style="green"))
    return True

    if False:

        print_review_summary(reviewed_items)

        certified = get_completed_items(run_id, gate)
        resumed_input = load_checkpoint_state(run_id) or {"run_id": run_id}
        resumed_input["human_decision"] = "COMPLETED"
        resumed_input["certified_kpis"] = [i["kpi"] for i in certified]
        resumed = hitl_review_node(resumed_input)
        console.print(Panel(f"Resumed successfully.\nStatus: {resumed.get('status', 'GATE1_COMPLETE')}", title="HITL Resume"))
        console.print(Panel(f"{run_id} CERTIFIED!", title="🎉", style="green"))
        return True
    return False


console = Console()

NODE_STATUS_TEXT = {
    "ingestion_node": "Validating input",
    "parse_input": "Preparing document",
    "acquire_and_validate": "Validating document",
    "estimate_and_fingerprint": "Fingerprinting document",
    "validate_budget": "Checking token budget",
    "validate_pricing": "Preparing pricing metadata",
    "validate_schema": "Running schema validation",
    "store_and_register": "Registering run",
    "memory_lookup": "Looking up memory",
    "req_extraction": "Extracting requirements",
    "kpi_extraction": "Extracting KPIs",
    "ai_store_writer": "Writing results",
    "chunk_and_embed": "Updating embeddings",
}

COMPLETION_MESSAGES = {
    "acquire_and_validate": "Input checked",
    "memory_lookup": "Memory checked",
    "req_extraction": "Requirements extracted",
    "kpi_extraction": "KPIs extracted",
    "store_and_register": "Run registered",
}


class SpinnerStatusHandler(logging.Handler):
    def __init__(self, status, console: Console) -> None:
        super().__init__(level=logging.INFO)
        self.status = status
        self.console = console
        self.last_message: Optional[str] = None
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        node = getattr(record, "node", None)
        message = self._message_for_record(record, node)
        if message:
            if message != self.last_message:
                self.status.update(message)
                self.last_message = message
        completion = self._completion_for_record(record, node)
        if completion:
            self.console.print(f"[green]{completion}[/green]")

    def _message_for_record(self, record: logging.LogRecord, node: Optional[str]) -> Optional[str]:
        if node in NODE_STATUS_TEXT:
            return f"[bold cyan]{NODE_STATUS_TEXT[node]}[/bold cyan]"

        message = record.getMessage()
        if "Requirement Extraction" in message:
            return "[bold cyan]Extracting requirements[/bold cyan]"
        if "KPI LLM attempt" in message or "KPI Extraction" in message:
            return "[bold cyan]Extracting KPIs[/bold cyan]"
        if "semantic lookup" in message or "memory_lookup" in message:
            return "[bold cyan]Looking up memory[/bold cyan]"
        return None

    def _completion_for_record(self, record: logging.LogRecord, node: Optional[str]) -> Optional[str]:
        message = record.getMessage()
        if node == "acquire_and_validate" and "END: _acquire_and_validate_brd" in message:
            return COMPLETION_MESSAGES["acquire_and_validate"]
        if node == "memory_lookup" and message.startswith("END memory_lookup"):
            return COMPLETION_MESSAGES["memory_lookup"]
        if node == "req_extraction" and "success" in message.lower():
            return COMPLETION_MESSAGES["req_extraction"]
        if node == "kpi_extraction" and "KPI Extraction success" in message:
            return COMPLETION_MESSAGES["kpi_extraction"]
        if node == "store_and_register" and "Run successfully registered" in message:
            return COMPLETION_MESSAGES["store_and_register"]
        return None


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    return Confirm.ask(prompt, default=default, console=console)


def print_requirements(result: Dict[str, Any]) -> None:
    req_keys = [
        "req_business_objective",
        "req_data_domains",
        "req_reporting_frequency",
        "req_target_audience",
        "req_constraints",
    ]
    table = Table(title="Extracted Requirements", show_lines=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    for key in req_keys:
        value = result.get(key, "N/A")
        if isinstance(value, list):
            value = ", ".join(map(str, value)) if value else "-"
        table.add_row(key.replace("req_", ""), str(value))
    console.print(table)


def print_certified_kpis(result: Dict[str, Any], certified_kpis: List[Dict[str, Any]]) -> None:
    table = Table(title="Certified KPIs (post-HITL)", show_lines=True, title_style="bold green")
    table.add_column("#", style="dim", width=4)
    table.add_column("Status", style="cyan", width=12)
    table.add_column("KPI", style="white")
    table.add_column("Description", style="white")
    table.add_column("Confidence", justify="right")

    kpis_map = {kpi.get('kpi_name', 'Unknown'): kpi for kpi in result.get('kpis', [])}
    
    for idx, certified in enumerate(certified_kpis, 1):
        kpi_name = certified['kpi_name']
        orig_conf = kpis_map.get(kpi_name, {}).get('ai_confidence_score', 'N/A')
        status = "[green]APPROVED[/]" if not certified.get('edited') else "[blue]EDITED[/]"
        table.add_row(
            str(idx),
            status,
            kpi_name,
            certified['kpi_description'][:60] + "..." if len(certified['kpi_description']) > 60 else certified['kpi_description'],
            f"{orig_conf:.2f}" if isinstance(orig_conf, (int, float)) else orig_conf,
        )
    console.print(table)


def print_kpis(result: Dict[str, Any]) -> None:
    kpis = result.get("kpis", []) or []
    if not kpis:
        console.print(Panel("No KPIs extracted.", title="KPIs", border_style="yellow"))
        return

    table = Table(title=f"Extracted KPIs ({len(kpis)})", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("KPI", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Source", style="magenta")

    source = result.get('kpi_source', 'unknown').upper()
    source_style = "bold yellow" if source == 'LLM' else "bold green"

    for idx, kpi in enumerate(kpis, 1):
        table.add_row(
            str(idx),
            str(kpi.get("kpi_name", "N/A")),
            str(kpi.get("kpi_description", "N/A")),
            str(kpi.get("ai_confidence_score", "N/A")),
            f"[{source_style}]{source}[/{source_style}]",
        )
    console.print(table)


def fetch_run_db_entries(run_id: str) -> List[Any]:
    conn = get_connection()
    try:
        db_schema = config["azure_sql"]["schema_name"]
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT TOP 10 stored_at, stage, artifact_type, payload
            FROM [{db_schema}].[ai_store]
            WHERE run_id = ? OR fingerprint LIKE ?
            ORDER BY stored_at DESC
            """,
            (run_id, f"%{run_id}%"),
        )
        return cursor.fetchall()
    finally:
        conn.close()


def print_db_entries(run_id: str) -> None:
    try:
        rows = fetch_run_db_entries(run_id)
    except Exception as exc:
        console.print(Panel(f"DB query failed: {exc}", title="Database Error", border_style="red"))
        return

    if not rows:
        console.print(Panel(f"No DB entries found for run {run_id}.", title="Database", border_style="yellow"))
        return

    table = Table(title=f"Stored DB Entries for {run_id}", show_lines=True)
    table.add_column("Stored At", style="cyan")
    table.add_column("Stage", style="white")
    table.add_column("Type", style="magenta")
    table.add_column("KPIs", justify="right", style="green")

    for row in rows:
        payload = json.loads(row[3]) if row[3] else {}
        table.add_row(str(row[0]), str(row[1]), str(row[2]), str(len(payload.get("kpis", []))))
    console.print(table)


def fetch_run_summary(run_id: str) -> List[Any]:
    conn = get_connection()
    try:
        db_schema = config["azure_sql"]["schema_name"]
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                stage,
                artifact_type,
                faithfulness_status,
                retry_count,
                input_tokens,
                output_tokens,
                token_count,
                cost_usd,
                stored_at
            FROM [{db_schema}].[ai_store]
            WHERE run_id = ?
            ORDER BY stored_at
            """,
            (run_id,),
        )
        return cursor.fetchall()
        console.print(Panel("Pipeline interrupted by user.", title="Interrupted", border_style="yellow"))
        return
    except Exception as exc:
        logger.exception("CLI execution failed", extra={"node": "cli"})
        console.print(Panel(f"{type(exc).__name__}: {exc}", title="Pipeline Error", border_style="red"))
        if DEV_MODE:
            raise
        return

    run_id = result.get("run_id", "unknown")
    pipeline_status = str(result.get("status", "unknown"))
    error_message = result.get("error")

    if pipeline_status == "FAILED" or error_message:
        message = error_message or "The pipeline reported a failure."
        console.print(Panel(message, title=f"Pipeline Failed ({run_id})", border_style="red"))
    else:
        console.print(Panel(f"Run ID: {run_id}\nStatus: {pipeline_status}", title="Pipeline Complete", border_style="green"))

    display_run_summary(run_id, pipeline_status)

    interactive = not args.no_prompt and sys.stdin.isatty()
    show_kpis = args.show_kpis
    show_reqs = args.show_reqs
    show_db = args.show_db
    show_payload = args.show_payload

    if interactive:
        # Memory match info
        source = result.get('kpi_source', 'LLM')
        fingerprint = result.get('fingerprint', 'N/A')
        if source != 'LLM':
            console.print(f"[bold green]🧠 MEMORY MATCH FOUND[/] Fingerprint: [cyan]{fingerprint}[/]")
        
        if not show_kpis:
            show_kpis = ask_yes_no("Show extracted KPIs?", default=True)
        if not show_reqs:
            show_reqs = ask_yes_no("Show extracted requirements?", default=False)
        if not show_db:
            show_db = ask_yes_no("Show stored DB entries for this run?", default=False)
        if not show_payload:
            show_payload = ask_yes_no("Inspect the final extracted payload?", default=False)
        if result.get("human_decision") == "PENDING":
            if ask_yes_no("Start HITL review now?", default=True):
                review_run(run_id, args.gate, {"configurable": {"thread_id": run_id}})

    if show_kpis:
        print_kpis(result)
    if show_reqs:
        print_requirements(result)
    if show_db:
        print_db_entries(run_id)
    if show_payload:
        render_payload(result)

def fetch_run_summary(run_id: str) -> List[Any]:
    conn = get_connection()
    try:
        db_schema = config["azure_sql"]["schema_name"]
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                stage,
                artifact_type,
                faithfulness_status,
                retry_count,
                input_tokens,
                output_tokens,
                token_count,
                cost_usd,
                stored_at
            FROM [{db_schema}].[ai_store]
            WHERE run_id = ?
            ORDER BY stored_at
            """,
            (run_id,),
        )
        return cursor.fetchall()
    finally:
        conn.close()


def load_checkpoint_state(run_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    try:
        db_schema = (
            config["azure_sql"].get("pipeline_schema")
            or config["azure_sql"]["schema_name"]
        )
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT TOP 1 full_state_json
            FROM [{db_schema}].[kpi_checkpoints]
            WHERE run_id = ?
            ORDER BY checkpoint_at DESC
            """,
            (run_id,),
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    finally:
        conn.close()


def display_run_summary(run_id: str, pipeline_status: str) -> None:
    try:
        rows = fetch_run_summary(run_id)
    except Exception as exc:
        console.print(Panel(f"Could not load run summary: {exc}", title="Summary Error", border_style="yellow"))
        return

    if not rows:
        return

    table = Table(title=f"Run Summary: {run_id}", show_lines=True)
    table.add_column("Stage", style="cyan")
    table.add_column("Artifact", style="magenta")
    table.add_column("Faithfulness", style="white")
    table.add_column("Tokens", justify="right", style="green")
    table.add_column("Stored At", style="white")

    total_tokens = 0
    for row in rows:
        token_count = row.token_count or 0
        total_tokens += token_count
        table.add_row(
            str(row.stage),
            str(row.artifact_type),
            str(row.faithfulness_status or "-"),
            str(token_count),
            str(row.stored_at),
        )

    console.print(table)
    console.print(
        Panel(
            f"Status: {pipeline_status}\nArtifacts: {len(rows)}\nTotal tokens: {total_tokens}",
            title="Summary Totals",
            border_style="bright_black",
        )
    )


def render_payload(result: Dict[str, Any]) -> None:
    console.print(Panel(JSON.from_data(result), title="Final Payload", border_style="cyan"))


def print_pending_run_list(runs: List[str], gate: int) -> None:
    if not runs:
        console.print(Panel(f"No pending Gate {gate} runs found.", title="Pending Runs", border_style="yellow"))
        return

    table = Table(title=f"Pending Gate {gate} Runs", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Run ID", style="cyan")
    for idx, run_id in enumerate(runs, 1):
        table.add_row(str(idx), run_id)
    console.print(table)


def build_initial_state(args: argparse.Namespace) -> Stage01State:
    brd_text = args.text or args.input
    run_id = args.run_id or str(uuid.uuid4())
    default_source_db = (
        config["azure_sql"].get("source_database")
        or config["azure_sql"].get("target_catalog")
        or config["azure_sql"]["database_name"]
    )
    source_databases = args.source_database or [default_source_db]

    return {
        "brd_text": brd_text,
        "run_id": run_id,
        "metadata": {},
        "status": "PENDING",
        "source_databases": source_databases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Athena pipeline CLI with Gate 1 and Gate 2 HITL review.")
    parser.add_argument("input", nargs="?", help="Raw BRD text or a path to a .txt/.docx file.")
    parser.add_argument(
        "--mode",
        choices=["run", "review", "list"],
        help="Backward-compatible mode flag. 'run' starts a pipeline, 'review' reviews an existing run, and 'list' lists pending runs.",
    )
    parser.add_argument("--text", help="Inline BRD text. Overrides positional input if both are provided.")
    parser.add_argument("--run-id", help="Run ID to use for a new pipeline run or to review an existing run.")
    parser.add_argument("--gate", type=int, choices=[1, 2], default=1, help="HITL gate to review when resuming.")
    parser.add_argument("--source-database", action="append", help="Source database(s) for table nomination. Repeat to add more.")
    parser.add_argument("--list-pending", action="store_true", help="List runs currently waiting on the selected gate.")
    parser.add_argument("--show-kpis", action="store_true", help="Show extracted KPIs after the run.")
    parser.add_argument("--show-reqs", action="store_true", help="Show extracted requirements after the run.")
    parser.add_argument("--show-db", action="store_true", help="Show stored DB entries for the run.")
    parser.add_argument("--show-payload", action="store_true", help="Show the final result payload as JSON.")
    parser.add_argument("--no-prompt", action="store_true", help="Disable interactive prompts.")
    args = parser.parse_args()

    mode = args.mode or ("list" if args.list_pending else None)

    if mode == "list" or args.list_pending:
        print_pending_run_list(list_pending_runs(args.gate), args.gate)
        return

    if mode == "review":
        if not args.run_id:
            parser.error("--mode review requires --run-id.")
        review_run(args.run_id, args.gate, {"configurable": {"thread_id": args.run_id}})
        return

    if args.run_id and not args.input and not args.text:
        review_run(args.run_id, args.gate, {"configurable": {"thread_id": args.run_id}})
        return

    if not args.input and not args.text:
        parser.error("Provide BRD input via positional input or --text, or pass --run-id to review an existing run.")

    from graph import app

    initial_state = build_initial_state(args)
    run_id = initial_state["run_id"] or str(uuid.uuid4())
    graph_config = {"configurable": {"thread_id": run_id}}

    spinner_handler: Optional[SpinnerStatusHandler] = None
    result: Optional[Dict[str, Any]] = None

    try:
        with console.status("[bold cyan]Starting Athena pipeline[/bold cyan]", spinner="dots") as status:
            spinner_handler = SpinnerStatusHandler(status, console)
            logger.addHandler(spinner_handler)
            result = app.invoke(initial_state, graph_config)
    except KeyboardInterrupt:
        console.print(Panel("Pipeline interrupted by user.", title="Interrupted", border_style="yellow"))
        return
    except Exception as exc:
        logger.exception("CLI execution failed", extra={"node": "cli"})
        console.print(Panel(f"{type(exc).__name__}: {exc}", title="Pipeline Error", border_style="red"))
        if DEV_MODE:
            raise
        return
    finally:
        if spinner_handler is not None:
            logger.removeHandler(spinner_handler)

    result = result or {}
    run_id = result.get("run_id", run_id)
    pipeline_status = str(result.get("status", "unknown"))
    error_message = result.get("error")

    if pipeline_status == "FAILED" or error_message:
        message = error_message or "The pipeline reported a failure."
        console.print(Panel(message, title=f"Pipeline Failed ({run_id})", border_style="red"))
    else:
        console.print(Panel(f"Run ID: {run_id}\nStatus: {pipeline_status}", title="Pipeline Complete", border_style="green"))

    display_run_summary(run_id, pipeline_status)

    interactive = not args.no_prompt and sys.stdin.isatty()
    show_kpis = args.show_kpis
    show_reqs = args.show_reqs
    show_db = args.show_db
    show_payload = args.show_payload

    if interactive:
        source = result.get("kpi_source", "LLM")
        fingerprint = result.get("fingerprint", "N/A")
        if source != "LLM":
            console.print(f"[bold green]ðŸ§  MEMORY MATCH FOUND[/] Fingerprint: [cyan]{fingerprint}[/]")

        if not show_kpis:
            show_kpis = ask_yes_no("Show extracted KPIs?", default=True)
        if not show_reqs:
            show_reqs = ask_yes_no("Show extracted requirements?", default=False)
        if not show_db:
            show_db = ask_yes_no("Show stored DB entries for this run?", default=False)
        if not show_payload:
            show_payload = ask_yes_no("Inspect the final extracted payload?", default=False)

        if result.get("human_decision") == "PENDING":
            if ask_yes_no("Start Gate 1 HITL review now?", default=True):
                review_run(run_id, 1, graph_config)
        elif result.get("human_table_decision") == "PENDING":
            if ask_yes_no("Start Gate 2 table review now?", default=True):
                review_tables(run_id, graph_config)

    if show_kpis:
        print_kpis(result)
    if show_reqs:
        print_requirements(result)
    if show_db:
        print_db_entries(run_id)
    if show_payload:
        render_payload(result)


if __name__ == "__main__":
    main()
