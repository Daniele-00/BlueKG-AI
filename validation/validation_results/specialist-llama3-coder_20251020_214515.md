# Validation Report — specialist-llama3-coder

## Summary
- Total: 21
- Success (attendibile): 16
- Success (strict): 13
- Failures: 5
- Accuracy (attendibile): 76.2%  |  Accuracy (strict): 61.9%
- Threshold: min_query_sim=0.80
- Avg Time: 27.87s
- Avg Jaccard: 0.333
- Avg Query Similarity: 0.807

## Tests
| # | Test ID | PASS | Att | Jac | QuerySim | Gen Time (s) | Note |
|:-:|:--------|:----:|:---:|:---:|:--------:|:------------:|:-----|
| 1 | TEST_00_FATTURATO_CLIENTE_SPECIFICO | ✅ | ❌ | 0.000 | 0.757 | 36.32 |  |
| 2 | TEST_01_CLIENTE_TOP_FATTURATO | ✅ | ✅ | 0.000 | 0.974 | 27.74 |  |
| 3 | TEST_02_LISTA_CLIENTI_PER_REGIONE | ✅ | ✅ | 1.000 | 1.000 | 12.90 |  |
| 4 | TEST_03_ANALISI_COMPLESSA_TOP_N | ✅ | ✅ | 1.000 | 0.998 | 24.71 |  |
| 5 | TEST_04_FATTURATO_PER_DITTA | ✅ | ✅ | 0.000 | 0.926 | 39.99 |  |
| 6 | TEST_05_ANNO_TOP_FATTURATO | ✅ | ✅ | 1.000 | 0.918 | 49.16 |  |
| 7 | TEST_06_FAMIGLIA_PIU_ACQUISTATA | ✅ | ✅ | 1.000 | 0.977 | 31.55 |  |
| 8 | TEST_07_PRODOTTO_PIU_VENDUTO | ✅ | ✅ | 0.000 | 0.962 | 28.84 |  |
| 9 | TEST_08_COSTO_ACQUISTO_PRODOTTO | ✅ | ✅ | 0.000 | 0.963 | 25.32 |  |
| 10 | TEST_09_PREZZO_VENDITA_PRODOTTO | ✅ | ❌ | 1.000 | 0.495 | 35.87 |  |
| 11 | TEST_11_DITTA_PER_FAMIGLIA | ✅ | ✅ | 1.000 | 0.976 | 35.02 |  |
| 12 | TEST_12_CONTEGGIO_CLIENTI_UNICI | ✅ | ✅ | 1.000 | 1.000 | 22.47 |  |
| 13 | TEST_13_FORNITORE_CON_PIU_DOCUMENTI | ✅ | ✅ | 0.000 | 0.801 | 26.05 |  |
| 14 | TEST_14_CLIENTI_SENZA_PRODOTTO | ❌ | ❌ | 0.000 | 0.333 | 36.26 | Results mismatch |
| 15 | TEST_15_LOCALITA_CON_PIU_FORNITORI | ❌ | ❌ | 0.000 | 0.451 | 22.46 | Results mismatch |
| 16 | TEST_16_IMPORTO_MEDIO_RIGA | ✅ | ✅ | 0.000 | 0.901 | 26.70 |  |
| 17 | TEST_17_FILTRO_MULTIPLO | ❌ | ❌ | 0.000 | 0.484 | 25.00 | Results mismatch |
| 18 | Genera la query Cypher per rispondere a questa domanda in italiano | ❌ | ❌ | 0.000 | 0.751 | 23.85 | Results mismatch |
| 19 | Genera una query Cypher per rispondere a questa domanda in italiano | ❌ | ❌ | 0.000 | 0.608 | 16.49 | Results mismatch |
| 20 | Genera una query Cypher per rispondere a questa domanda in italiano | ✅ | ✅ | 0.000 | 0.812 | 18.88 |  |
| 21 | Genera una query Cypher per rispondere a questa domanda in italiano | ✅ | ✅ | 0.000 | 0.863 | 19.62 |  |
