
import os
from dotenv import load_dotenv
load_dotenv()

from graph import app
from state import Stage01State

# Test input
test_input: Stage01State = {
    "brd_text": """
    Business stakeholders require monthly KPI dashboard for sales managers.
    Key metrics: Revenue growth rate, Customer acquisition cost, Monthly recurring revenue.
    Target users: Regional sales directors. Constraints: Real-time data, GDPR compliant.
    """,
    "run_id": "test-kpi-standalone",
    "fingerprint": "test-fingerprint-123",
    "token_estimate": 500,
    "metadata": {},
    "status": "RUNNING"
}

print("Running KPI extraction test...")
result = app.invoke(test_input)
print("\n=== RESULTS ===")
print("Requirements:", {
    k: result.get(k) for k in result if k.startswith('req_')
})
print("\nKPIs:", result.get('kpis', []))
print("Source:", result.get('kpi_source'))
print("Tokens:", result.get('kpi_tokens_used'))
print("Cost: $", result.get('kpi_cost_usd'))
print("Status:", result.get('status'))

if result.get('kpis'):
    print("\n✅ SUCCESS: Extracted KPIs!")
else:
    print("\n❌ No KPIs extracted")
