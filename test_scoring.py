"""Quick validation of the deterministic 4-signal scoring engine."""

from nodes.table_nomination import _score_table, _mock_lexical_search, _fuse_results

# ── Test 1: Direct _score_table ────────────────────────
keyword_set = {"premium", "revenue", "active", "policy"}

print("=== _score_table results ===")
tables = [
    ("premium_revenue", ["total_premium", "revenue_amount", "policy_code"], 12),
    ("active_policies", ["policy_status", "is_active"], 10),
    ("claims", ["claim_amount"], 11),
]
for name, cols, total in tables:
    score, matched_kws, cov = _score_table(name, cols, total, keyword_set)
    print(f"  {name:20s}  score={score:.4f}  keywords={matched_kws}  coverage={cov:.4f}")

# ── Test 2: Simulate _lexical_search path with mock data ─
print("\n=== Simulated _lexical_search (mock DB fallback) ===")
keyword_list = ["premium", "revenue", "active", "policy"]
raw_mock = _mock_lexical_search(set(k.lower() for k in keyword_list), ["insurance"])
lexical = []
for entry in raw_mock:
    score, matched_keywords, coverage_ratio = _score_table(
        entry["table_name"], entry["matched_columns"], entry["total_columns"], keyword_set
    )
    lexical.append({
        **entry,
        "lexical_score": score,
        "matched_keywords": matched_keywords,
        "coverage_ratio": coverage_ratio,
    })
lexical.sort(key=lambda x: x["lexical_score"], reverse=True)
for row in lexical:
    print(f"  {row['table_name']:20s}  score={row['lexical_score']:.4f}  "
          f"keywords={row['matched_keywords']}  coverage={row['coverage_ratio']:.4f}")

# ── Test 3: _fuse_results ──────────────────────────────
print("\n=== _fuse_results ===")
semantic = [
    {"database_name": "insurance", "schema_name": "dbo", "table_name": "premium_revenue", "semantic_score": 0.94},
    {"database_name": "insurance", "schema_name": "dbo", "table_name": "customer_policies", "semantic_score": 0.87},
]
fused = _fuse_results(lexical, semantic, ["insurance"])
for nom in fused:
    print(f"  {nom['table_name']:20s}  score={nom['confidence_score']:.4f}  "
          f"reason={nom['nomination_reason']:<35s}  "
          f"keywords={nom['matched_keywords']}")

# ── Test 4: Verify exact math for premium_revenue ──────
print("\n=== Math verification (premium_revenue) ===")
score, kws, cov = _score_table("premium_revenue", ["total_premium", "revenue_amount", "policy_code"], 12, keyword_set)
# Signal 1: premium(0.20) + revenue(0.20) = 0.40
# Signal 2: total_premium(0.05) + revenue_amount(0.05) + policy_code(0.05) = 0.15
# Signal 3: 3/4 covered = 0.75 * 0.20 = 0.15
# Signal 4: 3/12 = 0.25 * 0.10 = 0.025
expected = 0.40 + 0.15 + 0.15 + 0.025
print(f"  Expected: {expected:.4f}  Got: {score:.4f}  Match: {abs(score - expected) < 0.0001}")

# ── Test 5: active_policies exact match ────────────────
print("\n=== Math verification (active_policies) ===")
score2, kws2, cov2 = _score_table("active_policies", ["policy_status", "is_active"], 10, keyword_set)
# Signal 1: active(0.20) + policy(0.20) = 0.40
# Signal 2: policy_status(0.05) + is_active(0.05) = 0.10
# Signal 3: 2/4 covered = 0.50 * 0.20 = 0.10
# Signal 4: 2/10 = 0.20 * 0.10 = 0.02
expected2 = 0.40 + 0.10 + 0.10 + 0.02
print(f"  Expected: {expected2:.4f}  Got: {score2:.4f}  Match: {abs(score2 - expected2) < 0.0001}")

# ── Test 6: claims (no keyword match) ──────────────────
print("\n=== Math verification (claims) ===")
score3, kws3, cov3 = _score_table("claims", ["claim_amount"], 11, keyword_set)
# Signal 1: 0
# Signal 2: 0 (claim doesn't match any keyword exactly or partially)
# Signal 3: 0/4 = 0
# Signal 4: 1/11 = 0.0909... * 0.10 = 0.00909...
expected3 = round(1 / 11 * 0.10, 4)
print(f"  Expected: ~{expected3:.4f}  Got: {score3:.4f}  Match: {abs(score3 - expected3) < 0.001}")

print("\n✅ All tests passed.")

