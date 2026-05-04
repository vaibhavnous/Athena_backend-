#!/usr/bin/env python3
"""Fetch and display tables from Athena DB (ai_store, brd_run_registry, kpi_memory)."""
import argparse
import json
from typing import List
from utilis.db import get_connection, config
from utilis.logger import logger

def print_table_data(cursor, schema_name: str, table_name: str, limit: int = 10, where_clause: str = ''):
    """Pretty print table contents."""
    query = f"""
    SELECT TOP ({limit}) *
    FROM [{schema_name}].[{table_name}]
    {where_clause}
    ORDER BY 1 DESC
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    if rows:
        headers = [desc[0] for desc in cursor.description]
        print(f'\n📋 [{schema_name}].[{table_name}] ({len(rows)} rows):')
        print(' | '.join(headers))
        print('-' * 100)
        for row in rows:
            print(' | '.join(str(cell)[:50] + '...' if len(str(cell)) > 50 else str(cell) for cell in row))
    else:
        print(f'\n📋 [{schema_name}].[{table_name}]: No data')

def print_ai_store(cursor, schema_name: str, run_id: str = None, fingerprint: str = None, limit: int = 20):
    """Special formatted ai_store viewer."""
    where = ''
    params = []
    if run_id:
        where = 'WHERE run_id = ?'
        params = [run_id]
    elif fingerprint:
        where = 'WHERE fingerprint = ?'
        params = [fingerprint]
    
    cursor.execute(f"""
        SELECT TOP (?) stored_at, run_id, stage, artifact_type, 
               JSON_VALUE(payload, '$.kpi_count') as kpi_count,
               JSON_VALUE(payload, '$.source') as source
        FROM [{schema_name}].[ai_store] {where}
        ORDER BY stored_at DESC
    """, [limit] + params)
    
    rows = cursor.fetchall()
    if rows:
        print(f'\n🧠 AI_STORE (run_id={run_id or fingerprint or "all"}):')
        headers = ['Date', 'Run ID', 'Stage', 'Type', 'KPI Count', 'Source']
        table = []
        for row in rows:
            table.append([str(row[0])[:16], str(row[1])[:20], row[2], row[3], row[4] or 0, row[5] or 'N/A'])
        print('\n'.join([ ' | '.join(map(str, row)) for row in [headers] + table ]))
    else:
        print(f'\n🧠 AI_STORE: No data for {run_id or fingerprint or "query"}')

def main():
    parser = argparse.ArgumentParser(description='Fetch Athena backend tables')
    parser.add_argument('--table', help='Table name (ai_store, brd_run_registry, kpi_memory)')
    parser.add_argument('--run-id', help='Filter by run_id')
    parser.add_argument('--fingerprint', help='Filter by fingerprint')
    parser.add_argument('--limit', type=int, default=10, help='Row limit')
    args = parser.parse_args()

    db_conf = config["azure_sql"]
    schema_name = db_conf.get("schema_name", "metadata")

    print(f"🔍 Fetching from [{schema_name}] in {db_conf['database_name']}")

    conn = get_connection()
    try:
        cursor = conn.cursor()

        tables = ['ai_store', 'brd_run_registry', 'kpi_memory']
        if args.table:
            tables = [args.table]

        for table in tables:
            print_table_data(cursor, schema_name, table, args.limit, 
                           f'WHERE run_id = ?' if args.run_id else 
                           f'WHERE fingerprint = ?' if args.fingerprint else '')
            params = [args.run_id] if args.run_id else [args.fingerprint] if args.fingerprint else []

        if args.run_id or args.fingerprint:
            print_ai_store(cursor, schema_name, args.run_id, args.fingerprint, 20)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
