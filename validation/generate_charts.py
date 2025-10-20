#!/usr/bin/env python3
"""
Genera grafici da risultati di validazione.

Usage:
    python generate_charts.py validation_results/gpt4o-mini_20251015_172810.json
    python generate_charts.py validation_results/*.json  # Confronta tutti
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Stile grafici
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10

# =======================================================================
# FUNZIONI UTILITÀ
# =======================================================================


def load_results(json_paths):
    """Carica uno o più file JSON di risultati."""
    results = []
    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
            results.append(data)
    return results


def extract_timing_data(results_list):
    """Estrae dati timing per analisi."""
    data = []
    for result_file in results_list:
        config = result_file["config"]
        for test in result_file["results"]:
            data.append(
                {
                    "config": config,
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
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = [r["config"] for r in results_list]
    accuracies = [r["summary"]["accuracy"] for r in results_list]
    successes = [r["summary"]["success"] for r in results_list]
    totals = [r["summary"]["total"] for r in results_list]

    colors = sns.color_palette("husl", len(configs))
    bars = ax.bar(configs, accuracies, color=colors, edgecolor="black", linewidth=1.5)

    # Annotazioni
    for i, (bar, success, total) in enumerate(zip(bars, successes, totals)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{success}/{total}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Configuration", fontsize=12, fontweight="bold")
    ax.set_title(
        "Accuracy Comparison: Multi-Agent Configurations",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="r", linestyle="--", alpha=0.3, label="50% baseline")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300, bbox_inches="tight")
    print(f"✅ Salvato: accuracy_comparison.png")


def plot_timing_breakdown(results_list, output_dir):
    """Grafico stacked bar: Breakdown tempi."""
    fig, ax = plt.subplots(figsize=(12, 7))

    configs = [r["config"] for r in results_list]

    # Calcola medie
    avg_generation = []
    avg_db = []
    avg_synthesis = []

    for result in results_list:
        n = result["summary"]["total"]
        gen = sum((t.get("time_generation") or 0) for t in result["results"]) / n
        db = sum((t.get("time_db") or 0) for t in result["results"]) / n
        syn = sum((t.get("time_synthesis") or 0) for t in result["results"]) / n

        avg_generation.append(gen)
        avg_db.append(db)
        avg_synthesis.append(syn)

    x = np.arange(len(configs))
    width = 0.6

    # Stacked bars
    p1 = ax.bar(x, avg_generation, width, label="Query Generation", color="#3498db")
    p2 = ax.bar(
        x, avg_db, width, bottom=avg_generation, label="DB Execution", color="#2ecc71"
    )
    p3 = ax.bar(
        x,
        avg_synthesis,
        width,
        bottom=np.array(avg_generation) + np.array(avg_db),
        label="Response Synthesis",
        color="#e74c3c",
    )

    # Totale sopra
    totals = [sum(x) for x in zip(avg_generation, avg_db, avg_synthesis)]
    for i, total in enumerate(totals):
        ax.text(
            i,
            total + 0.5,
            f"{total:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    ax.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Configuration", fontsize=12, fontweight="bold")
    ax.set_title(
        "Average Time Breakdown per Configuration",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.legend(loc="upper left", frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_dir / "timing_breakdown.png", dpi=300, bbox_inches="tight")
    print(f"✅ Salvato: timing_breakdown.png")


def plot_success_rate_by_test(results_list, output_dir):
    """Heatmap: Success rate per test per config."""

    # Prepara dati
    all_test_ids = sorted(
        set(test["test_id"] for result in results_list for test in result["results"])
    )

    data = []
    for result in results_list:
        config = result["config"]
        success_map = {
            test["test_id"]: int(test["success"]) for test in result["results"]
        }
        row = [success_map.get(tid, 0) for tid in all_test_ids]
        data.append(row)

    df = pd.DataFrame(
        data, index=[r["config"] for r in results_list], columns=all_test_ids
    )

    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        cbar_kws={"label": "Success"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        "Test Success Matrix: Which configs pass which tests?",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Test ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Configuration", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "success_heatmap.png", dpi=300, bbox_inches="tight")
    print(f"✅ Salvato: success_heatmap.png")


def plot_time_vs_accuracy(results_list, output_dir):
    """Scatter: Accuracy vs Tempo medio."""
    fig, ax = plt.subplots(figsize=(10, 7))

    configs = [r["config"] for r in results_list]
    accuracies = [r["summary"]["accuracy"] for r in results_list]
    avg_times = [r["summary"]["avg_time"] for r in results_list]

    colors = sns.color_palette("husl", len(configs))

    # Scatter con annotazioni
    for i, (config, acc, time) in enumerate(zip(configs, accuracies, avg_times)):
        ax.scatter(
            time,
            acc,
            s=500,
            color=colors[i],
            edgecolor="black",
            linewidth=2,
            alpha=0.7,
            zorder=3,
        )
        ax.annotate(
            config,
            (time, acc),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
        )

    ax.set_xlabel("Average Time per Query (seconds)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Trade-off: Accuracy vs Speed", fontsize=14, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_time.png", dpi=300, bbox_inches="tight")
    print(f"✅ Salvato: accuracy_vs_time.png")


def plot_jaccard_metrics(results_list, output_dir):
    """Grafico a barre raggruppate: medie Jaccard per config.

    Include tre varianti se disponibili nel JSON:
    - avg_jaccard (alias-aware)
    - avg_jaccard_value_only (ignora alias)
    - avg_jaccard_expected_projection (proiezione su colonne attese)
    """
    configs = [r.get("config") for r in results_list]
    avg_jac = [r.get("summary", {}).get("avg_jaccard") for r in results_list]
    avg_jac_val = [r.get("summary", {}).get("avg_jaccard_value_only") for r in results_list]
    avg_jac_proj = [r.get("summary", {}).get("avg_jaccard_expected_projection") for r in results_list]

    # Se nessuna metrica è presente, non generare il grafico
    if not any(x is not None for x in (avg_jac + avg_jac_val + avg_jac_proj)):
        return

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, [v if v is not None else np.nan for v in avg_jac], width, label="Jaccard", color="#4C78A8")
    bars2 = ax.bar(x, [v if v is not None else np.nan for v in avg_jac_val], width, label="Jaccard (value-only)", color="#F58518")
    bars3 = ax.bar(x + width, [v if v is not None else np.nan for v in avg_jac_proj], width, label="Jaccard (proj exp cols)", color="#54A24B")

    ax.set_ylabel("Average Jaccard", fontsize=12, fontweight="bold")
    ax.set_title("Average Jaccard Metrics by Configuration", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Annotazioni sopra le barre
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "jaccard_metrics.png", dpi=300, bbox_inches="tight")
    print(f"✅ Salvato: jaccard_metrics.png")


def plot_timing_distribution(df, output_dir):
    """Box plot: Distribuzione tempi per fase."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=df["config"].unique(),
            patch_artist=True,
            notch=True,
            showmeans=True,
        )

        # Colori
        colors = sns.color_palette("husl", len(data_to_plot))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(f"{title}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Time (seconds)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Time Distribution Analysis", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "timing_distribution.png", dpi=300, bbox_inches="tight")
    print(f"✅ Salvato: timing_distribution.png")


def generate_summary_report(results_list, output_dir):
    """Genera report testuale riassuntivo."""
    report_path = output_dir / "summary_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("📊 VALIDATION SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for result in results_list:
            f.write(f"\n{'='*60}\n")
            f.write(f"Configuration: {result['config']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Accuracy:      {result['summary']['accuracy']:.2f}%\n")
            f.write(
                f"Success:       {result['summary']['success']}/{result['summary']['total']}\n"
            )
            f.write(f"Avg Time:      {result['summary']['avg_time']:.2f}s\n")
            # Jaccard metrics (if present)
            avg_jac = result.get('summary', {}).get('avg_jaccard')
            avg_jac_val = result.get('summary', {}).get('avg_jaccard_value_only')
            avg_jac_proj = result.get('summary', {}).get('avg_jaccard_expected_projection')
            if avg_jac is not None:
                f.write(f"Avg Jaccard:   {avg_jac:.3f}\n")
            if avg_jac_val is not None:
                f.write(f"Jaccard (value-only): {avg_jac_val:.3f}\n")
            if avg_jac_proj is not None:
                f.write(f"Jaccard (proj exp cols): {avg_jac_proj:.3f}\n")

            # Failed tests
            failed = [t for t in result["results"] if not t["success"]]
            if failed:
                f.write(f"\nFailed Tests ({len(failed)}):\n")
                for test in failed:
                    f.write(f"  - {test['test_id']}: {test['error'][:80]}...\n")

        # Best config
        best = max(results_list, key=lambda x: x["summary"]["accuracy"])
        fastest = min(results_list, key=lambda x: x["summary"]["avg_time"])

        f.write(f"\n\n{'='*80}\n")
        f.write("🏆 BEST CONFIGURATIONS\n")
        f.write(f"{'='*80}\n")
        f.write(
            f"Best Accuracy:  {best['config']} ({best['summary']['accuracy']:.2f}%)\n"
        )
        f.write(
            f"Fastest:        {fastest['config']} ({fastest['summary']['avg_time']:.2f}s)\n"
        )

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
    output_dir = Path("charts")
    output_dir.mkdir(exist_ok=True)

    print(f"\n🎨 Generazione grafici...")
    print("=" * 60)

    # Genera tutti i grafici
    plot_accuracy_comparison(results, output_dir)
    plot_timing_breakdown(results, output_dir)
    plot_success_rate_by_test(results, output_dir)
    plot_time_vs_accuracy(results, output_dir)
    plot_jaccard_metrics(results, output_dir)

    # Timing distribution (serve DataFrame)
    df = extract_timing_data(results)
    plot_timing_distribution(df, output_dir)

    # Report testuale
    generate_summary_report(results, output_dir)

    print("\n" + "=" * 60)
    print(f"✅ Tutti i grafici salvati in: {output_dir}/")
    print("=" * 60)
    print("\nGrafici generati:")
    print("  📊 accuracy_comparison.png")
    print("  ⏱️  timing_breakdown.png")
    print("  📋 success_heatmap.png")
    print("  ⚖️  accuracy_vs_time.png")
    print("  📐 jaccard_metrics.png")
    print("  📊 timing_distribution.png")
    print("  📄 summary_report.txt")


if __name__ == "__main__":
    main()
