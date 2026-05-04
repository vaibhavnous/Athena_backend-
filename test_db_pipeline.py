import time
from nodes.table_nomination import build_nomination_graph

def run_database_test():
    print("\n" + "="*60)
    print("🚀 RUNNING LIVE AZURE SQL PIPELINE TEST")
    print("="*60)

    # source_databases: The database(s) to discover tables from
    test_state = {
        "run_id": "live_test_001",
        "status": "RUNNING",
        "certified_kpis": ["Total Premium Revenue", "Active Policies"], 
        "source_databases": ["insurance"]  # DB name is "insurance" (no underscore)
    }

    print(f"Target Source Database(s): {test_state['source_databases']}")
    print(f"Target KPIs: {test_state['certified_kpis']}\n")

    # Compile the Graph
    app = build_nomination_graph().compile()
    
    # Execute and time the run
    start_time = time.time()
    try:
        final_state = app.invoke(test_state)
    except Exception as e:
        print(f"❌ PIPELINE CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    execution_time = time.time() - start_time

    # Print Diagnostics
    status = final_state.get("table_nomination_status")
    
    if status == "FAILED":
        print("❌ PIPELINE FAILED GRACEFULLY")
        print(f"Error: {final_state.get('table_nomination_error')}")
    else:
        print(f"✅ PIPELINE SUCCEEDED in {execution_time:.2f} seconds!")
        tables = final_state.get("nominated_tables", [])
        print(f"Total Tables Nominated: {len(tables)}\n")
        
        print("--- NOMINATED TABLES ---")
        for t in tables:
            name = t.get('table_name') if isinstance(t, dict) else t.table_name
            score = t.get('confidence_score') if isinstance(t, dict) else t.confidence_score
            reason = t.get('nomination_reason') if isinstance(t, dict) else t.nomination_reason
            
            print(f"➔ {name.ljust(25)} | Score: {score} | Reason: {reason}")

if __name__ == "__main__":
    run_database_test()
