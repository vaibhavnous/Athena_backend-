from utilis.db import get_pipeline_connection, get_client_connection, execute_source_sql

print("🔹 Testing Pipeline DB (ai_store)...")

try:
    conn = get_pipeline_connection()
    cursor = conn.cursor()

    # ✅ FIXED: use metadata schema
    cursor.execute("SELECT TOP 1 * FROM metadata.ai_store")

    rows = cursor.fetchall()
    print("✅ Pipeline DB working (ai_store accessible)")
    conn.close()

except Exception as e:
    print("❌ Pipeline DB FAILED:", e)


print("\n🔹 Testing Client DB (schema access)...")

try:
    # ✅ FIXED: correct DB name
    conn = get_client_connection("insurance")
    cursor = conn.cursor()

    cursor.execute("SELECT TOP 5 TABLE_NAME FROM INFORMATION_SCHEMA.TABLES")

    rows = cursor.fetchall()
    print("✅ Client DB working (schema accessible)")

    for r in rows:
        print("   →", r.TABLE_NAME)

    conn.close()

except Exception as e:
    print("❌ Client DB FAILED:", e)


print("\n🔹 Testing execute_source_sql()...")

try:
    # ✅ FIXED: correct DB name
    rows = execute_source_sql(
        "insurance",
        "SELECT TOP 5 TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"
    )

    print("✅ execute_source_sql working")

    for r in rows:
        print("   →", r.TABLE_NAME)

except Exception as e:
    print("❌ execute_source_sql FAILED:", e)