#!/usr/bin/env python3
"""
Genera grafici da risultati di validazione.

Usage:
    python generate_charts.py validation_results/gpt4o-mini_20251015_172810.json
    python generate_charts.py validation_results/*.json  # Confronta tutti
"""

import json
import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Stile grafici migliorato
sns.set_theme(
    style="whitegrid",
    palette="Set2",
    rc={
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "figure.titlesize": 18,
        "legend.fontsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.facecolor": "#f8f9fa",
        "figure.facecolor": "white",
    },
)
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"

# Palette colori moderna
COLORS = {
    "primary": "#2E86AB",
    "success": "#06A77D",
    "warning": "#F77F00",
    "danger": "#D62828",
    "info": "#4ECDC4",
    "purple": "#9D4EDD",
}

# =======================================================================
# FUNZIONI UTILITÀ
# =======================================================================


def load_results(json_paths):
    """Carica uno o più file JSON di risultati."""
    results = []
    for path in json_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"⚠️ Errore caricamento {path}: {e}")
    return results


def build_config_label(result):
    """
    Ritorna etichetta con config.
    Aggiunge top-k solo se presente e diverso da None.
    """
    # Cerca top_k nel livello principale o nel summary
    top_k = result.get("examples_top_k")

    if top_k is None and "summary" in result:
        top_k = result["summary"].get("examples_top_k")

    if top_k is not None:
        return f"{result['config']}\n(k={top_k})"
    else:
        return result["config"]


def extract_timing_data(results_list):
    """Estrae dati timing per analisi."""
    data = []
    for result_file in results_list:
        config_label = build_config_label(result_file)
        for test in result_file["results"]:
            data.append(
                {
                    "config": config_label,
                    "test_id": test["test_id"],
                    "success": test["success"],
                    "time_total": test.get("time_total", 0),
                    "time_generation": test.get("time_generation", 0),
                    "time_db": test.get("time_db", 0),
                    "time_synthesis": test.get("time_synthesis", 0),
                }
            )
    return pd.DataFrame(data)


# =======================================================================
# GRAFICI
# =======================================================================


def plot_accuracy_comparison(results_list, output_dir):
    """Grafico a barre: Accuracy per config."""
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = [build_config_label(r) for r in results_list]
    x = np.arange(len(labels))
    width = 0.6

    # FIX: Usa .get con fallback alla vecchia chiave 'accuracy'
    values = [
        r["summary"].get("accuracy_strict", r["summary"].get("accuracy", 0.0))
        for r in results_list
    ]

    color_map = {
        "gpt-4o": "#2ecc71",
        "gpt4o": "#2ecc71",
        "llama3": "#3498db",
        "llama3-groq": "#9b59b6",
        "gemini": "#f1c40f",
        "default": "#95a5a6",
    }

    colors = []
    model_names_in_data = set()

    for r in results_list:
        specialist_name = r.get("config", "default").lower()
        found_key = "default"
        for model_key in color_map:
            if model_key in specialist_name:
                found_key = model_key
                break
        colors.append(color_map[found_key])
        if found_key != "default":
            model_names_in_data.add(found_key)

    wrapped_labels = [
        "\n".join(textwrap.wrap(l, width=15, break_long_words=False)) for l in labels
    ]

    bars = ax.bar(
        x,
        values,
        width,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#2d3436",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    has_top_k = any(
        (r.get("examples_top_k") is not None)
        or (r.get("summary", {}).get("examples_top_k") is not None)
        for r in results_list
    )

    x_label_text = "Configuration (Top-K)" if has_top_k else "Configuration"

    ax.set_xlabel(x_label_text, fontsize=13, fontweight="bold")
    ax.set_title(
        "Strict Accuracy by Configuration",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)

    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.5, linewidth=2)

    legend_patches = []
    for model_name in sorted(list(model_names_in_data)):
        if model_name in color_map:
            patch = mpatches.Patch(color=color_map[model_name], label=model_name)
            legend_patches.append(patch)

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            title="Modello",
            fontsize=11,
            title_fontsize=13,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    plt.subplots_adjust(right=0.85)

    plt.savefig(
        output_dir / "accuracy_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"✅ Salvato: accuracy_comparison.png")


def plot_timing_breakdown(results_list, output_dir):
    """Grafico stacked bar: Breakdown tempi."""
    fig, ax = plt.subplots(figsize=(13, 7))

    labels = [build_config_label(r) for r in results_list]

    avg_generation = []
    avg_db = []
    avg_synthesis = []

    for result in results_list:
        n = result["summary"]["total"]
        if n == 0:
            n = 1  # avoid zero division
        gen = sum((t.get("time_generation") or 0) for t in result["results"]) / n
        db = sum((t.get("time_db") or 0) for t in result["results"]) / n
        syn = sum((t.get("time_synthesis") or 0) for t in result["results"]) / n

        avg_generation.append(gen)
        avg_db.append(db)
        avg_synthesis.append(syn)

    x = np.arange(len(labels))
    width = 0.65

    p1 = ax.bar(
        x,
        avg_generation,
        width,
        label="Query Generation",
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=2,
    )
    p2 = ax.bar(
        x,
        avg_db,
        width,
        bottom=avg_generation,
        label="DB Execution",
        color=COLORS["success"],
        edgecolor="white",
        linewidth=2,
    )
    p3 = ax.bar(
        x,
        avg_synthesis,
        width,
        bottom=np.array(avg_generation) + np.array(avg_db),
        label="Response Synthesis",
        color=COLORS["warning"],
        edgecolor="white",
        linewidth=2,
    )

    totals = [sum(x) for x in zip(avg_generation, avg_db, avg_synthesis)]
    for i, total in enumerate(totals):
        ax.text(
            i,
            total + 0.3,
            f"{total:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8
            ),
        )

    ax.set_ylabel("Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Configuration (Top-K)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Average Time Breakdown per Configuration",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(loc="upper left", frameon=True, shadow=True, fancybox=True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        output_dir / "timing_breakdown.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"✅ Salvato: timing_breakdown.png")


def plot_success_rate_by_test(results_list, output_dir):
    return


def plot_time_vs_accuracy(results_list, output_dir):
    """Scatter: Accuracy vs Tempo medio."""
    fig, ax = plt.subplots(figsize=(11, 8))

    labels = [build_config_label(r) for r in results_list]

    # --- FIX HERE: Fallback to 'accuracy' if 'accuracy_strict' is missing ---
    accuracies = [
        r["summary"].get("accuracy_strict", r["summary"].get("accuracy", 0.0))
        for r in results_list
    ]

    avg_times = [r["summary"]["avg_time"] for r in results_list]

    colors = sns.color_palette("husl", len(labels))

    for i, (label, acc, time) in enumerate(zip(labels, accuracies, avg_times)):
        ax.scatter(
            time,
            acc,
            s=600,
            color=colors[i],
            edgecolor="black",
            linewidth=2.5,
            alpha=0.75,
            zorder=3,
        )
        ax.annotate(
            label,
            (time, acc),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                alpha=0.7,
            ),
        )

    ax.set_xlabel("Average Time per Query (seconds)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold")
    ax.set_title("Trade-off: Accuracy vs Speed", fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.set_xlim(left=-0.5)
    ax.set_ylim(0, 105)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        output_dir / "accuracy_vs_time.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"✅ Salvato: accuracy_vs_time.png")


def plot_jaccard_metrics(results_list, output_dir):
    """Grafico a barre raggruppate: medie Jaccard per config."""
    configs = [build_config_label(r) for r in results_list]
    avg_jac = [r.get("summary", {}).get("avg_jaccard") for r in results_list]

    color_map = {
        "gpt-4o": "#2ecc71",
        "gpt4o": "#2ecc71",
        "llama3": "#3498db",
        "llama3-groq": "#9b59b6",
        "gemini": "#f1c40f",
        "default": "#95a5a6",
    }
    colors = []
    model_names_in_data = set()

    for r in results_list:
        specialist_name = r.get("config", "default").lower()
        found_key = "default"
        for model_key in color_map:
            if model_key in specialist_name:
                found_key = model_key
                break
        colors.append(color_map[found_key])
        if found_key != "default":
            model_names_in_data.add(found_key)

    wrapped_labels = [
        "\n".join(textwrap.wrap(l, width=15, break_long_words=False)) for l in configs
    ]
    if not any(x is not None for x in (avg_jac)):
        return

    x = np.arange(len(configs))
    width = 0.5

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(
        x,
        [v if v is not None else 0 for v in avg_jac],
        width,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        zorder=3,
    )

    ax.set_ylabel("Average Jaccard", fontsize=13, fontweight="bold")
    ax.set_title(
        "Average Jaccard Metrics by Configuration",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.4, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )

    legend_patches = []
    for model_name in sorted(list(model_names_in_data)):
        if model_name in color_map:
            patch = mpatches.Patch(color=color_map[model_name], label=model_name)
            legend_patches.append(patch)

    if legend_patches:
        ax.legend(
            handles=legend_patches,
            title="Modello",
            fontsize=11,
            title_fontsize=13,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.savefig(
        output_dir / "jaccard_metrics.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"✅ Salvato: jaccard_metrics.png")


def plot_jaccard_distribution(results_list, output_dir):
    """Bar chart: Distribuzione dei punteggi Jaccard per configurazione."""
    bins = np.linspace(0, 1, 11)
    bin_labels = [f"{b:.1f}" for b in bins[:-1]]

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(bin_labels))
    width = 0.8 / len(results_list)

    colors = sns.color_palette("Set2", len(results_list))

    for idx, result in enumerate(results_list):
        label = build_config_label(result)
        values = [
            v
            for v in (test.get("jaccard_index") for test in result["results"])
            if isinstance(v, (int, float))
        ]
        if not values:
            continue

        counts, _ = np.histogram(values, bins=bins)

        offset = (idx - len(results_list) / 2) * width + width / 2
        bars = ax.bar(
            x + offset,
            counts,
            width,
            label=label,
            color=colors[idx],
            edgecolor="black",
            linewidth=1,
            alpha=0.85,
        )

        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Jaccard Score Range", fontsize=13, fontweight="bold")
    ax.set_ylabel("Number of Queries", fontsize=13, fontweight="bold")
    ax.set_title(
        "Distribution of Jaccard Scores", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=0)
    ax.grid(axis="y", alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(frameon=True, shadow=True, fancybox=True, loc="upper center")

    plt.tight_layout()
    plt.savefig(
        output_dir / "jaccard_distribution.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"✅ Salvato: jaccard_distribution.png")


def plot_timing_distribution(df, output_dir):
    """Box plot: Distribuzione tempi per fase."""
    if df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    phases = [
        ("time_generation", "Query Generation", axes[0, 0]),
        ("time_db", "DB Execution", axes[0, 1]),
        ("time_synthesis", "Response Synthesis", axes[1, 0]),
        ("time_total", "Total Time", axes[1, 1]),
    ]

    for phase, title, ax in phases:
        data_to_plot = [
            df[df["config"] == config][phase].dropna().values
            for config in df["config"].unique()
        ]

        if not data_to_plot:
            continue

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=df["config"].unique(),
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="red", markersize=6),
            medianprops=dict(color="darkblue", linewidth=2),
        )

        colors = sns.color_palette("Set2", len(data_to_plot))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.5)

        ax.set_title(f"{title}", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Time (seconds)", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Time Distribution Analysis", fontsize=18, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(
        output_dir / "timing_distribution.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"✅ Salvato: timing_distribution.png")


def plot_repair_metrics(results_list, output_dir):
    """
    Genera grafici specifici per l'analisi del Repair Loop.
    """
    # Prepariamo i dati
    data = []
    for r in results_list:
        config_label = build_config_label(r)

        # Calcoli per config
        total_tests = len(r["results"])
        repair_triggered = sum(1 for t in r["results"] if t.get("repair_used"))

        # Di quelli dove il repair è scattato, quanti sono finiti con successo (strict match)?
        repair_success = sum(
            1
            for t in r["results"]
            if t.get("repair_used") and t.get("strict_match_final")
        )

        # Errori tecnici
        errors_before = sum(1 for t in r["results"] if t.get("neo4j_error_before"))
        errors_after = sum(1 for t in r["results"] if t.get("neo4j_error_after"))

        data.append(
            {
                "config": config_label,
                "repair_triggered": repair_triggered,
                "repair_success": repair_success,
                "repair_fail": repair_triggered - repair_success,
                "errors_before": errors_before,
                "errors_after": errors_after,
            }
        )

    df = pd.DataFrame(data)

    # Se nessun modello ha usato il repair, saltiamo
    if df.empty or df["repair_triggered"].sum() == 0:
        print(
            "⚠️ Nessun dato di Repair trovato nei risultati. Salto i grafici relativi."
        )
        return

    # --- GRAFICO 1: EFFICACIA REPAIR (Stacked Bar) ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    x = np.arange(len(df["config"]))
    width = 0.5

    # Barra totale triggerata (Sfondo rosso = falliti)
    p1 = ax1.bar(
        x,
        df["repair_fail"],
        width,
        label="Repair Failed (Wrong Result)",
        color="#e74c3c",
        edgecolor="black",
        bottom=df["repair_success"],
    )
    # Barra successi (Verde = corretti)
    p2 = ax1.bar(
        x,
        df["repair_success"],
        width,
        label="Repair Succeeded (Strict Match)",
        color="#2ecc71",
        edgecolor="black",
    )

    ax1.set_ylabel("Number of Queries", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Repair Loop Effectiveness\n(Triggered vs Fixed)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["config"], rotation=15, ha="right")
    ax1.legend()

    # Annotazioni
    for i in x:
        total_trig = df.iloc[i]["repair_triggered"]
        success = df.iloc[i]["repair_success"]
        if total_trig > 0:
            rate = (success / total_trig) * 100
            ax1.text(
                i,
                total_trig + 0.5,
                f"{rate:.0f}% Fixed",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_dir / "repair_effectiveness.png", dpi=300)
    print(f"✅ Salvato: repair_effectiveness.png")

    # --- GRAFICO 2: RIDUZIONE ERRORI TECNICI (Grouped Bar) ---
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    width = 0.35

    rects1 = ax2.bar(
        x - width / 2,
        df["errors_before"],
        width,
        label="Syntax Errors BEFORE Repair",
        color="#f39c12",
    )
    rects2 = ax2.bar(
        x + width / 2,
        df["errors_after"],
        width,
        label="Syntax Errors AFTER Repair",
        color="#8e44ad",
    )

    ax2.set_ylabel("Number of Technical Errors", fontsize=13, fontweight="bold")
    ax2.set_title(
        "Technical Error Mitigation\n(Before vs After Repair Loop)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["config"], rotation=15, ha="right")
    ax2.legend()

    ax2.bar_label(rects1, padding=3)
    ax2.bar_label(rects2, padding=3)

    plt.tight_layout()
    plt.savefig(output_dir / "repair_error_mitigation.png", dpi=300)
    print(f"✅ Salvato: repair_error_mitigation.png")


def generate_summary_report(results_list, output_dir):
    """Genera report testuale riassuntivo."""
    report_path = output_dir / "summary_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("VALIDATION SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for result in results_list:
            f.write(f"\n{'='*60}\n")
            f.write(f"Configuration: {result['config']}\n")
            f.write(f"{'='*60}\n")

            # Safe access to accuracy
            acc = result["summary"].get(
                "accuracy_strict", result["summary"].get("accuracy", 0.0)
            )
            f.write(f"Accuracy:      {acc:.2f}%\n")

            success_total = (
                result["summary"].get("success_attendibile")
                or result["summary"].get("success")
                or 0
            )
            f.write(f"Success:       {success_total}/{result['summary']['total']}\n")
            f.write(f"Avg Time:      {result['summary']['avg_time']:.2f}s\n")

            # Dati repair se presenti
            trig = sum(1 for t in result["results"] if t.get("repair_used"))
            succ = sum(
                1
                for t in result["results"]
                if t.get("repair_used") and t.get("strict_match_final")
            )
            f.write(f"Repair Triggered: {trig}\n")
            f.write(f"Repair Fixed:     {succ}\n")

    print(f"✅ Salvato: summary_report.txt")


# =======================================================================
# MAIN
# =======================================================================


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_charts.py <json_file1> [json_file2] ...")
        print("Example: python generate_charts.py validation_results/*.json")
        sys.exit(1)

    # Carica risultati
    json_files = sys.argv[1:]
    print(f"\n📂 Caricamento {len(json_files)} file...")
    results = load_results(json_files)

    # Output directory
    base_dir = Path("charts")
    base_dir.mkdir(exist_ok=True)
    unique_configs = {r["config"] for r in results}
    if len(unique_configs) == 1:
        config_name = next(iter(unique_configs)).replace(os.sep, "_")
        output_dir = base_dir / config_name
    else:
        output_dir = base_dir / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎨 Generazione grafici...")
    print("=" * 60)

    # Genera tutti i grafici
    plot_accuracy_comparison(results, output_dir)
    plot_timing_breakdown(results, output_dir)
    plot_success_rate_by_test(results, output_dir)
    plot_time_vs_accuracy(results, output_dir)
    plot_jaccard_metrics(results, output_dir)
    plot_jaccard_distribution(results, output_dir)
    plot_repair_metrics(results, output_dir)
    # Timing distribution (serve DataFrame)
    df = extract_timing_data(results)
    plot_timing_distribution(df, output_dir)

    # Report testuale
    generate_summary_report(results, output_dir)

    print("\n" + "=" * 60)
    print(f"✅ Tutti i grafici salvati in: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
