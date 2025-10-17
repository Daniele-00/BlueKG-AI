import pandas as pd
import matplotlib.pyplot as plt

# Carica i dati
df = pd.read_csv("validation_comprehensive_results.csv")

# Calcola le metriche aggregate per ogni modello
summary = (
    df.groupby("model")
    .agg(
        accuracy_perc=("success", lambda x: 100 * x.sum() / len(x)),
        avg_gen_time_s=("tempo_generazione_query", "mean"),
    )
    .reset_index()
)

print("📊 Report Comparativo dei Modelli 📊")
print(summary)

# --- Grafico 1: Accuratezza ---
plt.figure(figsize=(10, 6))
plt.bar(
    summary["model"], summary["accuracy_perc"], color=["#4CAF50", "#FFC107", "#2196F3"]
)
plt.title("Accuratezza per Modello (%)")
plt.ylabel("Accuratezza (%)")
plt.ylim(0, 105)
for index, value in enumerate(summary["accuracy_perc"]):
    plt.text(index, value + 1, f"{value:.1f}%", ha="center")
plt.savefig("comparison_accuracy.png")
plt.show()

# --- Grafico 2: Tempo di Generazione Query ---
plt.figure(figsize=(10, 6))
plt.bar(
    summary["model"], summary["avg_gen_time_s"], color=["#4CAF50", "#FFC107", "#2196F3"]
)
plt.title("Tempo Medio di Generazione Query (secondi)")
plt.ylabel("Secondi")
for index, value in enumerate(summary["avg_gen_time_s"]):
    plt.text(index, value, f"{value:.2f}s", ha="center", va="bottom")
plt.savefig("comparison_timing.png")
plt.show()
