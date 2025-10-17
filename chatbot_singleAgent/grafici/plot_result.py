import json
import matplotlib.pyplot as plt

# --- 1. Carica i dati dei risultati ---
try:
    with open("validation_results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print("❌ Errore: File 'validation_results.json' non trovato.")
    print("Esegui prima lo script 'run_validation.py' per generare i risultati.")
    exit()

# --- 2. Prepara i dati per il grafico ---
labels = "Risposte Corrette", "Risposte Sbagliate"
sizes = [results["success"], results["failures"]]
colors = ["#4CAF50", "#F44336"]  # Verde per successo, Rosso per fallimento
explode = (0.1, 0)  # "esplode" leggermente la fetta dei successi

# Calcola il titolo con la percentuale di accuratezza
accuracy = (
    (results["success"] / results["total_tests"]) * 100
    if results["total_tests"] > 0
    else 0
)
title = f"Accuratezza del Chatbot sul Validation Set ({accuracy:.2f}%)"

# --- 3. Crea e personalizza il grafico a torta ---
fig1, ax1 = plt.subplots()
ax1.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",  # Mostra le percentuali sulle fette
    shadow=True,
    startangle=90,
)

ax1.axis("equal")  # Assicura che la torta sia un cerchio.
plt.title(title)

# --- 4. Salva il grafico e mostralo a schermo ---
plt.savefig("validation_chart.png")
print("✅ Grafico salvato come 'validation_chart.png'")
plt.show()
