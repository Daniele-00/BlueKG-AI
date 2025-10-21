# test_embeddings.py
from example_retriever import ExampleRetriever

retriever = ExampleRetriever()

# Test domande simili
test_cases = [
    ("Attualmente la ditta 2 è in perdita o in crescita?", "SALES_CYCLE"),
    ("Chi è il cliente con maggior fatturato?", "SALES_CYCLE"),
    ("Qual è la famiglia più acquistata?", "PURCHASE_CYCLE"),
    ("Mostrami documenti FVC di gennaio", "GENERAL_QUERY"),
]

for question, specialist in test_cases:
    print(f"\n{'='*80}")
    print(f"Q: {question}")
    print(f"Specialist: {specialist}")

    examples = retriever.retrieve(question, specialist, top_k=3)

    print(f"\nTop-3 recuperati:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. [{ex['id']}]")
        print(f"     {ex['question']}")
        # Check: È rilevante?
        is_relevant = input(f"     Rilevante? (y/n): ")
        if is_relevant.lower() != "y":
            print(f"     ❌ NON RILEVANTE!")

print("\n✅ Test completato")
