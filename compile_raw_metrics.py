import json
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_metrics(metrics_file: str) -> list:
    """
    Load metrics from a JSON file.
        metrics_file (str): Path to the metrics JSON file
        list: The loaded metrics
    """
    try:
        with open(metrics_file, "r") as f:
            data = json.load(f)
        return data.get("metrics", [])
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return []


def extract_overall_metrics(metrics: list) -> pd.DataFrame:
    """
    Extract overall metrics (not just fact-level) from the metrics data.
        metrics (list): List of metrics for all simulations
        pd.DataFrame: DataFrame containing overall metrics
    """
    rows = []

    for sim in metrics:
        if not sim.get("valid", False):
            continue

        rows.append(
            {
                "story_key": sim.get("story_key", "unknown"),
                "treatment_key": sim.get("treatment_key", "unknown"),
                "author_stance": sim.get("author_stance", "unknown"),
                "editor_stance": sim.get("editor_stance", "unknown"),
                "bertscore_precision": sim.get("bertscore_precision", 0),
                "bertscore_recall": sim.get("bertscore_recall", 0),
                "bertscore_f1": sim.get("bertscore_f1", 0),
                "fact_presence_avg": sim.get("fact_presence", {}).get(
                    "average_presence", 0
                ),
            }
        )

    return pd.DataFrame(rows)


def extract_fact_metrics(metrics: list) -> pd.DataFrame:
    """
    Extract fact-level metrics from the metrics data.
        metrics (list): List of metrics for all simulations
        pd.DataFrame: DataFrame containing fact-level metrics
    """
    rows = []

    for sim in metrics:
        if not sim.get("valid", False):
            continue

        story_key = sim.get("story_key", "unknown")
        treatment_key = sim.get("treatment_key", "unknown")
        author_stance = sim.get("author_stance", "unknown")
        editor_stance = sim.get("editor_stance", "unknown")
        bertscore_f1 = sim.get("bertscore_f1", 0)

        # Get fact presence scores
        fact_presence = sim.get("fact_presence", {})

        # Extract individual fact scores, excluding average_presence
        fact_keys = [k for k in fact_presence.keys() if k != "average_presence"]

        # Create a row for each fact
        for fact_key in fact_keys:
            fact_score = fact_presence.get(fact_key, 0)

            rows.append(
                {
                    "story_key": story_key,
                    "treatment_key": treatment_key,
                    "author_stance": author_stance,
                    "editor_stance": editor_stance,
                    "bertscore_f1": bertscore_f1,
                    "fact_key": fact_key,
                    "fact_score": fact_score,
                }
            )

    return pd.DataFrame(rows)


def summarize_bertscore_by_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize BERTScore metrics by treatment.
        df (pd.DataFrame): DataFrame with overall metrics
        pd.DataFrame: Summary of BERTScore by treatment
    """
    return (
        df.groupby("treatment_key")
        .agg(
            {
                "bertscore_precision": ["mean", "std", "count"],
                "bertscore_recall": ["mean", "std"],
                "bertscore_f1": ["mean", "std"],
                "fact_presence_avg": ["mean", "std"],
            }
        )
        .reset_index()
    )


def summarize_bertscore_by_story(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize BERTScore metrics by story (topic).
        df (pd.DataFrame): DataFrame with overall metrics
        pd.DataFrame: Summary of BERTScore by story
    """
    return (
        df.groupby("story_key")
        .agg(
            {
                "bertscore_precision": ["mean", "std", "count"],
                "bertscore_recall": ["mean", "std"],
                "bertscore_f1": ["mean", "std"],
                "fact_presence_avg": ["mean", "std"],
            }
        )
        .reset_index()
    )


def summarize_bertscore_by_treatment_story(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize BERTScore metrics by treatment and story.
        df (pd.DataFrame): DataFrame with overall metrics
        pd.DataFrame: Summary of BERTScore by treatment and story
    """
    return (
        df.groupby(["treatment_key", "story_key"])
        .agg(
            {
                "bertscore_precision": ["mean", "std", "count"],
                "bertscore_recall": ["mean", "std"],
                "bertscore_f1": ["mean", "std"],
                "fact_presence_avg": ["mean", "std"],
            }
        )
        .reset_index()
    )


def summarize_by_story_fact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize fact metrics by story and fact.
        df (pd.DataFrame): DataFrame with fact-level metrics
        pd.DataFrame: Summary by story and fact
    """
    return (
        df.groupby(["story_key", "fact_key"])
        .agg({"fact_score": ["mean", "std", "count"], "bertscore_f1": ["mean"]})
        .reset_index()
    )


def summarize_by_treatment_fact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize fact metrics by treatment and fact.
        df (pd.DataFrame): DataFrame with fact-level metrics
        pd.DataFrame: Summary by treatment and fact
    """
    return (
        df.groupby(["treatment_key", "fact_key"])
        .agg({"fact_score": ["mean", "std", "count"], "bertscore_f1": ["mean"]})
        .reset_index()
    )


def summarize_by_story_treatment_fact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize fact metrics by story, treatment, and fact.
        df (pd.DataFrame): DataFrame with fact-level metrics
        pd.DataFrame: Summary by story, treatment, and fact
    """
    return (
        df.groupby(["story_key", "treatment_key", "fact_key"])
        .agg({"fact_score": ["mean", "std", "count"], "bertscore_f1": ["mean"]})
        .reset_index()
    )


def plot_bertscore_heatmap(df: pd.DataFrame, output_dir: str = None):
    """
    Plot a heatmap of BERTScore F1 by story and treatment.
        df (pd.DataFrame): DataFrame with overall metrics
        output_dir (str, optional): Directory to save the plot to
    """
    # Create pivot table for the heatmap
    pivot_df = df.pivot_table(
        values="bertscore_f1",
        index="story_key",
        columns="treatment_key",
        aggfunc="mean",
    )

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title("BERTScore F1 by Story and Treatment")
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "bertscore_heatmap.png"))
        plt.close()
    else:
        plt.show()

    # Also create heatmap for precision
    pivot_df = df.pivot_table(
        values="bertscore_precision",
        index="story_key",
        columns="treatment_key",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title("BERTScore Precision by Story and Treatment")
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "bertscore_precision_heatmap.png"))
        plt.close()
    else:
        plt.show()

    # Also create heatmap for recall
    pivot_df = df.pivot_table(
        values="bertscore_recall",
        index="story_key",
        columns="treatment_key",
        aggfunc="mean",
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title("BERTScore Recall by Story and Treatment")
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "bertscore_recall_heatmap.png"))
        plt.close()
    else:
        plt.show()


def plot_bertscore_bars(df: pd.DataFrame, output_dir: str = None):
    """
    Plot bar charts comparing BERTScore metrics across treatments.
        df (pd.DataFrame): DataFrame with overall metrics
        output_dir (str, optional): Directory to save the plot to
    """
    # Aggregate by treatment
    treatment_summary = (
        df.groupby("treatment_key")
        .agg(
            {
                "bertscore_precision": "mean",
                "bertscore_recall": "mean",
                "bertscore_f1": "mean",
            }
        )
        .reset_index()
    )

    # Create bar chart
    plt.figure(figsize=(12, 6))

    # Create positions for the bars
    treatments = treatment_summary["treatment_key"].tolist()
    x = np.arange(len(treatments))
    width = 0.25

    # Create grouped bars
    plt.bar(
        x - width, treatment_summary["bertscore_precision"], width, label="Precision"
    )
    plt.bar(x, treatment_summary["bertscore_recall"], width, label="Recall")
    plt.bar(x + width, treatment_summary["bertscore_f1"], width, label="F1")

    plt.xlabel("Treatment")
    plt.ylabel("Score")
    plt.title("BERTScore Metrics by Treatment")
    plt.xticks(x, treatments, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "bertscore_by_treatment.png"))
        plt.close()
    else:
        plt.show()

    # Also create bar chart by story
    story_summary = (
        df.groupby("story_key")
        .agg(
            {
                "bertscore_precision": "mean",
                "bertscore_recall": "mean",
                "bertscore_f1": "mean",
            }
        )
        .reset_index()
    )

    plt.figure(figsize=(12, 6))

    # Create positions for the bars
    stories = story_summary["story_key"].tolist()
    x = np.arange(len(stories))
    width = 0.25

    # Create grouped bars
    plt.bar(x - width, story_summary["bertscore_precision"], width, label="Precision")
    plt.bar(x, story_summary["bertscore_recall"], width, label="Recall")
    plt.bar(x + width, story_summary["bertscore_f1"], width, label="F1")

    plt.xlabel("Story Topic")
    plt.ylabel("Score")
    plt.title("BERTScore Metrics by Story Topic")
    plt.xticks(x, stories, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "bertscore_by_story.png"))
        plt.close()
    else:
        plt.show()


def plot_fact_heatmap(df: pd.DataFrame, output_dir: str = None):
    """
    Plot a heatmap of fact scores by story and treatment.
        df (pd.DataFrame): DataFrame with fact-level metrics
        output_dir (str, optional): Directory to save the plot to
    """
    # Get average fact scores by story, treatment, and fact
    pivot_df = df.pivot_table(
        values="fact_score",
        index=["story_key", "fact_key"],
        columns="treatment_key",
        aggfunc="mean",
    )

    # Plot heatmap
    plt.figure(figsize=(12, max(8, len(pivot_df) * 0.4)))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", linewidths=0.5)
    plt.title("Fact Presence by Story, Fact, and Treatment")
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "fact_heatmap.png"))
        plt.close()
    else:
        plt.show()


def plot_fact_bars(df: pd.DataFrame, output_dir: str = None):
    """
    Plot bar charts of fact scores by treatment for each story.
        df (pd.DataFrame): DataFrame with fact-level metrics
        output_dir (str, optional): Directory to save the plots to
    """
    # Get unique stories and facts
    stories = df["story_key"].unique()

    for story in stories:
        # Filter for this story
        story_df = df[df["story_key"] == story]

        # Get average fact scores by fact and treatment
        pivot_df = story_df.pivot_table(
            values="fact_score",
            index="fact_key",
            columns="treatment_key",
            aggfunc="mean",
        )

        # Plot bar chart
        plt.figure(figsize=(12, 6))
        pivot_df.plot(kind="bar")
        plt.title(f"Fact Presence by Treatment for {story}")
        plt.xlabel("Fact")
        plt.ylabel("Average Fact Score")
        plt.legend(title="Treatment")
        plt.ylim(0, 1)
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir, f"fact_bars_{story}.png"))
            plt.close()
        else:
            plt.show()


def plot_bertscore_vs_fact_presence(df: pd.DataFrame, output_dir: str = None):
    """
    Plot scatter plot of BERTScore vs fact presence.
        df (pd.DataFrame): DataFrame with overall metrics
        output_dir (str, optional): Directory to save the plot to
    """
    plt.figure(figsize=(10, 8))

    # Plot scatter by treatment
    treatments = df["treatment_key"].unique()
    for treatment in treatments:
        treatment_df = df[df["treatment_key"] == treatment]
        plt.scatter(
            treatment_df["bertscore_f1"],
            treatment_df["fact_presence_avg"],
            label=treatment,
            alpha=0.7,
        )

    # Add trend line for all data
    z = np.polyfit(df["bertscore_f1"], df["fact_presence_avg"], 1)
    p = np.poly1d(z)
    plt.plot(
        sorted(df["bertscore_f1"]),
        p(sorted(df["bertscore_f1"])),
        "k--",
        alpha=0.5,
        label="Trend Line",
    )

    plt.xlabel("BERTScore F1")
    plt.ylabel("Average Fact Presence")
    plt.title("Relationship Between BERTScore and Fact Presence")
    plt.legend(title="Treatment")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "bertscore_vs_fact_presence.png"))
        plt.close()
    else:
        plt.show()


def compile_metrics(metrics_file: str, output_dir: str = None):
    """
    Compile and analyze metrics from the metrics file.
        metrics_file (str): Path to the metrics JSON file
        output_dir (str, optional): Directory to save the output to
    """
    print(f"Loading metrics from {metrics_file}...")
    metrics = load_metrics(metrics_file)

    if not metrics:
        print("No metrics found!")
        return

    print(f"Found {len(metrics)} simulations with metrics")

    # Extract metrics
    print("Extracting metrics...")
    overall_df = extract_overall_metrics(metrics)
    fact_df = extract_fact_metrics(metrics)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate overall summaries
    print("Generating BERTScore summaries...")
    bertscore_by_treatment = summarize_bertscore_by_treatment(overall_df)
    bertscore_by_story = summarize_bertscore_by_story(overall_df)
    bertscore_by_treatment_story = summarize_bertscore_by_treatment_story(overall_df)

    # Generate fact summaries
    print("Generating fact presence summaries...")
    story_fact_summary = summarize_by_story_fact(fact_df)
    treatment_fact_summary = summarize_by_treatment_fact(fact_df)
    full_summary = summarize_by_story_treatment_fact(fact_df)

    # Save summaries to CSV
    if output_dir:
        # Save overall metrics
        overall_df.to_csv(os.path.join(output_dir, "overall_metrics.csv"))

        # Save BERTScore summaries
        bertscore_by_treatment.to_csv(
            os.path.join(output_dir, "bertscore_by_treatment.csv")
        )
        bertscore_by_story.to_csv(os.path.join(output_dir, "bertscore_by_story.csv"))
        bertscore_by_treatment_story.to_csv(
            os.path.join(output_dir, "bertscore_by_treatment_story.csv")
        )

        # Save fact summaries
        story_fact_summary.to_csv(os.path.join(output_dir, "story_fact_summary.csv"))
        treatment_fact_summary.to_csv(
            os.path.join(output_dir, "treatment_fact_summary.csv")
        )
        full_summary.to_csv(os.path.join(output_dir, "full_summary.csv"))

        print(f"Saved summary CSVs to {output_dir}")

    # Generate visualizations
    print("Generating visualizations...")

    # BERTScore visualizations
    plot_bertscore_heatmap(overall_df, output_dir)
    plot_bertscore_bars(overall_df, output_dir)
    plot_bertscore_vs_fact_presence(overall_df, output_dir)

    # Fact presence visualizations
    plot_fact_heatmap(fact_df, output_dir)
    plot_fact_bars(fact_df, output_dir)

    # Generate a readable report
    report = []
    report.append(">>> Metrics summary report\n")

    # BERTScore summary
    report.append("BERTScore by Treatment:")
    for treatment, group in overall_df.groupby("treatment_key"):
        precision = group["bertscore_precision"].mean()
        recall = group["bertscore_recall"].mean()
        f1 = group["bertscore_f1"].mean()
        report.append(
            f"  {treatment}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

    report.append("\nBERTScore by Story:")
    for story, group in overall_df.groupby("story_key"):
        precision = group["bertscore_precision"].mean()
        recall = group["bertscore_recall"].mean()
        f1 = group["bertscore_f1"].mean()
        report.append(
            f"  {story}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

    # Fact presence summary
    report.append("\nFact Presence by Treatment:")
    for treatment, group in overall_df.groupby("treatment_key"):
        avg_score = group["fact_presence_avg"].mean()
        report.append(f"  {treatment}: {avg_score:.4f}")

    report.append("\nFact Presence by Story:")
    for story, group in overall_df.groupby("story_key"):
        avg_score = group["fact_presence_avg"].mean()
        report.append(f"  {story}: {avg_score:.4f}")

    # Get top 3 highest and lowest impact treatments on facts
    report.append("\nTop 3 treatments with highest fact presence:")
    treatment_avg = (
        overall_df.groupby("treatment_key")["fact_presence_avg"]
        .mean()
        .sort_values(ascending=False)
    )
    for i, (treatment, score) in enumerate(treatment_avg.head(3).items()):
        report.append(f"  {i+1}. {treatment}: {score:.4f}")

    report.append("\nTop 3 Treatments with Lowest Fact Presence:")
    for i, (treatment, score) in enumerate(treatment_avg.tail(3).items()):
        report.append(f"  {i+1}. {treatment}: {score:.4f}")

    # Fact breakdown by treatment
    report.append("\nFact Breakdown by Treatment:")
    grouped = (
        fact_df.groupby(["treatment_key", "fact_key"])["fact_score"]
        .mean()
        .reset_index()
    )
    for treatment, group in grouped.groupby("treatment_key"):
        report.append(f"\n  {treatment}:")
        for _, row in group.iterrows():
            report.append(f"    {row['fact_key']}: {row['fact_score']:.4f}")

    # Fact breakdown by story
    report.append("\nFact Breakdown by Story:")
    grouped = (
        fact_df.groupby(["story_key", "fact_key"])["fact_score"].mean().reset_index()
    )
    for story, group in grouped.groupby("story_key"):
        report.append(f"\n  {story}:")
        for _, row in group.iterrows():
            report.append(f"    {row['fact_key']}: {row['fact_score']:.4f}")

    report_text = "\n".join(report)

    if output_dir:
        with open(os.path.join(output_dir, "metrics_summary_report.txt"), "w") as f:
            f.write(report_text)
        print(f"Saved summary report to {output_dir}/metrics_summary_report.txt")
    else:
        print("\n" + report_text)

    print("\nMetrics analysis complete!")
    return overall_df, fact_df


def main():
    parser = argparse.ArgumentParser(
        description="Compile and analyze metrics from simulation results"
    )
    parser.add_argument("metrics_file", help="Path to the metrics JSON file")
    parser.add_argument("--output", "-o", help="Directory to save the output to")

    args = parser.parse_args()

    compile_metrics(args.metrics_file, args.output)


if __name__ == "__main__":
    main()
