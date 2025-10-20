# Validation Report — specialist-gemini-2.5-pro

## Summary
- Total: 18
- Success: 16
- Failures: 2
- Accuracy: 88.9%
- Avg Time: 31.10s
- Avg Jaccard: 0.426
- Avg Jaccard (value-only): 0.926
- Avg Jaccard (proj exp cols): 0.426
- Avg Query Similarity: 0.921

## Tests
| # | Test ID | PASS | Jac | JacVal | JacProj | QuerySim | Gen Time (s) | Note |
|:-:|:--------|:----:|:---:|:------:|:-------:|:--------:|:------------:|:-----|
| 1 | TEST_00_FATTURATO_CLIENTE_SPECIFICO | ✅ | 0.000 | 1.000 | 0.000 | 0.976 | 27.36 |  |
| 2 | TEST_01_CLIENTE_TOP_FATTURATO | ✅ | 0.000 | 1.000 | 0.000 | 0.951 | 28.26 |  |
| 3 | TEST_02_LISTA_CLIENTI_PER_REGIONE | ✅ | 1.000 | 1.000 | 1.000 | 1.000 | 22.13 |  |
| 4 | TEST_03_ANALISI_COMPLESSA_TOP_N | ✅ | 1.000 | 1.000 | 1.000 | 1.000 | 29.00 |  |
| 5 | TEST_04_FATTURATO_PER_DITTA | ✅ | 0.000 | 1.000 | 0.000 | 0.955 | 29.39 |  |
| 6 | TEST_05_ANNO_TOP_FATTURATO | ✅ | 1.000 | 1.000 | 1.000 | 0.985 | 35.98 |  |
| 7 | TEST_06_FAMIGLIA_PIU_ACQUISTATA | ✅ | 1.000 | 1.000 | 1.000 | 0.988 | 27.76 |  |
| 8 | TEST_07_PRODOTTO_PIU_VENDUTO | ✅ | 0.000 | 1.000 | 0.000 | 0.797 | 31.45 |  |
| 9 | TEST_08_COSTO_ACQUISTO_PRODOTTO | ✅ | 0.000 | 1.000 | 0.000 | 0.967 | 34.54 |  |
| 10 | TEST_09_PREZZO_VENDITA_PRODOTTO | ✅ | 0.000 | 1.000 | 0.000 | 0.987 | 40.03 |  |
| 11 | TEST_10_TREND_DITTA | ✅ | 1.000 | 1.000 | 1.000 | 1.000 | 37.30 |  |
| 12 | TEST_11_DITTA_PER_FAMIGLIA | ❌ | 0.667 | 0.667 | 0.667 | 0.942 | 34.02 | Results mismatch |
| 13 | TEST_12_CONTEGGIO_CLIENTI_UNICI | ✅ | 0.000 | 1.000 | 0.000 | 0.629 | 26.41 |  |
| 14 | TEST_13_FORNITORE_CON_PIU_DOCUMENTI | ✅ | 0.000 | 1.000 | 0.000 | 0.791 | 25.30 |  |
| 15 | TEST_14_CLIENTI_SENZA_PRODOTTO | ✅ | 1.000 | 1.000 | 1.000 | 1.000 | 31.58 |  |
| 16 | TEST_15_LOCALITA_CON_PIU_FORNITORI | ✅ | 0.000 | 1.000 | 0.000 | 0.875 | 30.84 |  |
| 17 | TEST_16_IMPORTO_MEDIO_RIGA | ❌ | 0.000 | 0.000 | 0.000 | 0.736 | 35.67 | Results mismatch |
| 18 | TEST_17_FILTRO_MULTIPLO | ✅ | 1.000 | 1.000 | 1.000 | 1.000 | 32.86 |  |
