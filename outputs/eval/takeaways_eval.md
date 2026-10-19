# Takeaway Extraction Evaluation

Labels: `data/labels/meeting1.json`
Matching: timestamp window +/-30s OR fuzzy text similarity >= 80

| Category | rule_based Precision | rule_based Recall | rule_based F1 | llm Precision | llm Recall | llm F1 | llm_verify Precision | llm_verify Recall | llm_verify F1 |
|---|---|---|---|---|---|---|---|---|---|
| decisions | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.5 | 0.6 | 0.5 | 0.545 |
| action_items | 0.086 | 1.0 | 0.158 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| deadlines | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |