# Validation Report — specialist-llama3-coder

## Summary
- Total: 29
- Success (attendibile): 25
- Success (strict): 25
- Failures: 4
- Accuracy (attendibile): 86.2%  |  Accuracy (strict): 86.2%
- Threshold: min_query_sim=0.80
- Avg Time: 25.13s
- Avg Jaccard: 0.857
- Avg Query Similarity: 0.873

## Tests
| # | Test ID | PASS | Att | Jac | QuerySim | Gen Time (s) | Note |
|:-:|:--------|:----:|:---:|:---:|:--------:|:------------:|:-----|
| 1 | TEST-31-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 26.55 |  |
| 2 | TEST-32-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 0.862 | 23.39 |  |
| 3 | TEST-33-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 22.71 |  |
| 4 | TEST-34-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 25.05 |  |
| 5 | TEST-35-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ❌ | 1.000 | 0.799 | 32.78 |  |
| 6 | TEST-36-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 0.800 | 25.13 |  |
| 7 | TEST-37-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 13.20 |  |
| 8 | TEST-38-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 12.15 |  |
| 9 | TEST-39-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 16.30 |  |
| 10 | TEST-40-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 35.87 |  |
| 11 | TEST-41-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 13.10 |  |
| 12 | TEST-42-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ❌ | 1.000 | 0.547 | 17.79 |  |
| 13 | TEST-43-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 30.46 |  |
| 14 | TEST-44-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 29.52 |  |
| 15 | TEST-45-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 27.17 |  |
| 16 | TEST-46-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 27.94 |  |
| 17 | TEST-47-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ❌ | 0.000 | 0.067 | 36.40 |  |
| 18 | TEST-48-Genera la query Cypher per rispondere a questo input in italiano | ❌ | ❌ | 0.000 | 0.048 | 31.87 | Results mismatch |
| 19 | TEST-49-Genera la query Cypher per rispondere a questo input in italiano | ❌ | ❌ | 0.000 | 0.524 | 23.43 | Results mismatch |
| 20 | TEST-62-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 49.29 |  |
| 21 | TEST-63-Genera la query Cypher per rispondere a questo input in italiano | ❌ | ❌ | - | - | - | API error 500: {'detail': "Errore durante l'elaborazione: {code: Neo.ClientError.Statement.ArithmeticError} {message: / by zero}"} |
| 22 | TEST-64-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 24.20 |  |
| 23 | TEST-65-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 24.37 |  |
| 24 | TEST-66-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 29.53 |  |
| 25 | TEST-67-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 20.79 |  |
| 26 | TEST-68-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 26.10 |  |
| 27 | TEST-69-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 26.43 |  |
| 28 | TEST-70-Genera la query Cypher per rispondere a questo input in italiano | ❌ | ❌ | 0.000 | 0.794 | 27.91 | Results mismatch |
| 29 | TEST-71-Genera la query Cypher per rispondere a questo input in italiano | ✅ | ✅ | 1.000 | 1.000 | 29.45 |  |
