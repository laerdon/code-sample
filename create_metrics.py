#!/usr/bin/env python
import json
import argparse
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from bert_score import score as bert_score

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def convert_numpy_types(obj):
    """
    Convert NumPy types to Python native types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj


def load_simulation_results(file_path: str) -> dict:
    """
    Load simulation results from a JSON file.
        file_path (str): Path to the JSON file containing simulation results
        dict: The loaded simulation results
    """
    try:
        with open(file_path, "r") as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading simulation results: {e}")
        return None


def extract_initial_story(simulation: dict, initial_stories: dict) -> str:
    """
    Extract the initial story for a simulation based on its story_key.
        simulation (dict): A single simulation result
        initial_stories (dict): Dictionary mapping story keys to initial stories
        Returns the initial story for this simulation
    """
    story_key = simulation.get("story_key", "tariffs")
    return initial_stories.get(story_key, "")


def extract_facts(simulation: dict, facts_library: dict) -> str:
    """
    Extract the facts for a simulation based on its facts_key.
        simulation (dict): A single simulation result
        facts_library (dict): Dictionary mapping keys to facts
        Returns the facts for this simulation, split into individual items
    """
    facts_key = simulation.get("facts_key", "tariffs")
    facts_text = facts_library.get(facts_key, "")

    # Split facts by bullet points and clean
    facts = []
    for fact in re.split(r"\n\s*\*\s*", facts_text):
        fact = fact.strip()
        if fact and not fact.startswith(
            "*"
        ):  # Skip empty lines and already processed lines
            facts.append(fact)

    return facts


def calculate_bertscore(initial_story: str, final_story: str) -> tuple:
    """
    Calculate BERTScore between initial and final stories.
        initial_story (str): The initial story
        final_story (str): The final story
        Returns the BERTScore precision, recall, and F1 scores
    """
    try:
        P, R, F1 = bert_score(
            [final_story], [initial_story], lang="en", rescale_with_baseline=True
        )
        return P.item(), R.item(), F1.item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return 0.0, 0.0, 0.0


def calculate_fact_presence(final_story: str, facts: list, model) -> dict:
    """
    Calculate the presence of each fact in the final story using sentence-level BERTScore.
        final_story (str): The final story
        facts (list): List of facts to check for presence
        model: The sentence transformer model
        Returns the maximum similarity score for each fact and average fact presence
    """
    # Tokenize the final story into sentences
    try:
        story_sentences = sent_tokenize(final_story)

        # If no sentences found, consider the whole story as one sentence
        if not story_sentences:
            story_sentences = [final_story]

        # Get embeddings for story sentences
        story_embeddings = model.encode(story_sentences)

        # Calculate presence for each fact
        fact_scores = {}
        for i, fact in enumerate(facts):
            # Get embedding for the fact
            fact_embedding = model.encode([fact])[0].reshape(1, -1)

            # Calculate similarity with each sentence in the story
            similarities = cosine_similarity(fact_embedding, story_embeddings)

            # Get the maximum similarity (best match)
            max_similarity = np.max(similarities)
            fact_scores[f"fact_{i+1}"] = max_similarity

        # Calculate average fact presence
        avg_presence = np.mean(list(fact_scores.values())) if fact_scores else 0
        fact_scores["average_presence"] = avg_presence

        return fact_scores
    except Exception as e:
        print(f"Error calculating fact presence: {e}")
        return {"average_presence": 0.0}


def analyze_simulation(
    simulation: dict, initial_stories: dict, facts_library: dict, model
) -> dict:
    """
    Analyze a single simulation for divergence metrics.
        simulation (dict): A single simulation result
        initial_stories (dict): Dictionary mapping story keys to initial stories
        facts_library (dict): Dictionary mapping keys to facts
        model: The sentence transformer model
        Returns the calculated metrics
    """
    # Extract data
    final_story = simulation.get("final_story", "")

    if not final_story:
        return {
            "story_key": simulation.get("story_key", ""),
            "treatment_key": simulation.get("treatment", {}).get("key", ""),
            "valid": False,
            "reason": "Empty final story",
        }

    # Get initial story
    initial_story = extract_initial_story(simulation, initial_stories)

    if not initial_story:
        return {
            "story_key": simulation.get("story_key", ""),
            "treatment_key": simulation.get("treatment", {}).get("key", ""),
            "valid": False,
            "reason": "Could not find initial story",
        }

    # Get facts
    facts = extract_facts(simulation, facts_library)

    # Calculate BERTScore
    precision, recall, f1 = calculate_bertscore(initial_story, final_story)

    # Calculate fact presence
    fact_scores = calculate_fact_presence(final_story, facts, model)

    # Return metrics
    return {
        "story_key": simulation.get("story_key", ""),
        "treatment_key": simulation.get("treatment", {}).get("key", ""),
        "author_stance": simulation.get("treatment", {}).get("author_stance", ""),
        "editor_stance": simulation.get("treatment", {}).get("editor_stance", ""),
        "valid": True,
        "bertscore_precision": precision,
        "bertscore_recall": recall,
        "bertscore_f1": f1,
        "fact_presence": fact_scores,
    }


def analyze_all_simulations(results_file: str, output_file: str = None) -> list:
    """
    Analyze all simulations in a results file and save metrics to a JSON file.
        results_file (str): Path to the JSON file containing simulation results
        output_file (str, optional): Path to save the metrics to. If None, will use results filename with _metrics suffix.
        Returns the calculated metrics for all simulations
    """
    print(f"Loading simulation results from {results_file}...")
    results = load_simulation_results(results_file)

    if not results:
        print("Error: Could not load simulation results")
        return []

    # Import INITIAL_STORIES and FACTS_LIBRARY from run_simulations
    try:
        # Try to import, but if it fails, use placeholders
        from run_simulations import INITIAL_STORIES, FACTS_LIBRARY

        print("Successfully imported INITIAL_STORIES and FACTS_LIBRARY")
    except ImportError:
        print(
            "Warning: Could not import INITIAL_STORIES and FACTS_LIBRARY, using placeholders"
        )
        INITIAL_STORIES = {}
        FACTS_LIBRARY = {}

    # Load the sentence transformer model
    print("Loading sentence transformer model...")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    # Analyze each simulation
    print("Analyzing simulations...")
    metrics = []
    skipped_count = 0
    for sim in tqdm(results.get("simulations", [])):
        # Skip simulations with a story_key of "ai regulation"
        if sim.get("story_key", "") == "ai regulation":
            skipped_count += 1
            continue

        sim_metrics = analyze_simulation(sim, INITIAL_STORIES, FACTS_LIBRARY, model)
        metrics.append(sim_metrics)

    if skipped_count > 0:
        print(f"Skipped {skipped_count} simulations with story_key 'ai regulation'")

    # Save metrics to a file
    if output_file is None:
        base_name = os.path.splitext(results_file)[0]
        output_file = f"{base_name}_metrics.json"

    # Convert NumPy types to Python native types before serialization
    serializable_metrics = convert_numpy_types(metrics)

    with open(output_file, "w") as f:
        json.dump({"metrics": serializable_metrics}, f, indent=2)

    print(f"Metrics saved to {output_file}")

    return metrics


def generate_summary_report(metrics: list, output_file: str = None) -> dict:
    """
    Generate a summary report from the metrics.
        metrics (list): List of metrics for all simulations
        output_file (str, optional): Path to save the report to. If None, will print to console.
        Returns the summary statistics
    """
    if not metrics:
        print("No metrics to summarize")
        return {}

    # Filter out invalid metrics
    valid_metrics = [m for m in metrics if m.get("valid", False)]

    if not valid_metrics:
        print("No valid metrics to summarize")
        return {}

    # Group by story_key and treatment_key
    by_story = defaultdict(list)
    by_treatment = defaultdict(list)
    by_combo = defaultdict(list)

    for m in valid_metrics:
        story_key = m.get("story_key", "unknown")
        treatment_key = m.get("treatment_key", "unknown")
        combo_key = f"{story_key}_{treatment_key}"

        by_story[story_key].append(m)
        by_treatment[treatment_key].append(m)
        by_combo[combo_key].append(m)

    # Calculate average metrics
    summary = {
        "overall": {
            "count": len(valid_metrics),
            "bertscore_f1_avg": np.mean(
                [m.get("bertscore_f1", 0) for m in valid_metrics]
            ),
            "fact_presence_avg": np.mean(
                [
                    m.get("fact_presence", {}).get("average_presence", 0)
                    for m in valid_metrics
                ]
            ),
        },
        "by_story": {},
        "by_treatment": {},
        "by_combination": {},
    }

    # Summary by story
    for story_key, story_metrics in by_story.items():
        summary["by_story"][story_key] = {
            "count": len(story_metrics),
            "bertscore_f1_avg": np.mean(
                [m.get("bertscore_f1", 0) for m in story_metrics]
            ),
            "fact_presence_avg": np.mean(
                [
                    m.get("fact_presence", {}).get("average_presence", 0)
                    for m in story_metrics
                ]
            ),
        }

    # Summary by treatment
    for treatment_key, treatment_metrics in by_treatment.items():
        summary["by_treatment"][treatment_key] = {
            "count": len(treatment_metrics),
            "bertscore_f1_avg": np.mean(
                [m.get("bertscore_f1", 0) for m in treatment_metrics]
            ),
            "fact_presence_avg": np.mean(
                [
                    m.get("fact_presence", {}).get("average_presence", 0)
                    for m in treatment_metrics
                ]
            ),
        }

    # Summary by combination
    for combo_key, combo_metrics in by_combo.items():
        summary["by_combination"][combo_key] = {
            "count": len(combo_metrics),
            "bertscore_f1_avg": np.mean(
                [m.get("bertscore_f1", 0) for m in combo_metrics]
            ),
            "fact_presence_avg": np.mean(
                [
                    m.get("fact_presence", {}).get("average_presence", 0)
                    for m in combo_metrics
                ]
            ),
        }

    # Convert NumPy types to Python native types before serialization
    serializable_summary = convert_numpy_types(summary)

    # Save or print report
    if output_file:
        with open(output_file, "w") as f:
            json.dump(serializable_summary, f, indent=2)
        print(f"Summary report saved to {output_file}")
    else:
        print("\n=== Summary Report ===")
        print(f"Total valid simulations: {summary['overall']['count']}")
        print(f"Overall BERTScore F1: {summary['overall']['bertscore_f1_avg']:.4f}")
        print(f"Overall Fact Presence: {summary['overall']['fact_presence_avg']:.4f}")

        print("\nBy Story:")
        for story_key, stats in summary["by_story"].items():
            print(
                f"  {story_key}: BERTScore F1={stats['bertscore_f1_avg']:.4f}, Fact Presence={stats['fact_presence_avg']:.4f}"
            )

        print("\nBy Treatment:")
        for treatment_key, stats in summary["by_treatment"].items():
            print(
                f"  {treatment_key}: BERTScore F1={stats['bertscore_f1_avg']:.4f}, Fact Presence={stats['fact_presence_avg']:.4f}"
            )

    return summary


def visualize_metrics(metrics: list, output_dir: str = None) -> None:
    """
    Generate visualizations from the metrics.
        metrics (list): List of metrics for all simulations
        output_dir (str, optional): Directory to save visualizations to. If None, will show plots.
        Returns None
    """
    if not metrics:
        print("No metrics to visualize")
        return

    # Filter out invalid metrics
    valid_metrics = [m for m in metrics if m.get("valid", False)]

    if not valid_metrics:
        print("No valid metrics to visualize")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(valid_metrics)
    df["fact_presence_avg"] = df["fact_presence"].apply(
        lambda x: x.get("average_presence", 0)
    )

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. Plot BERTScore F1 by story and treatment
    plt.figure(figsize=(12, 6))
    df_pivot = df.pivot_table(
        values="bertscore_f1",
        index="story_key",
        columns="treatment_key",
        aggfunc="mean",
    )
    ax = df_pivot.plot(kind="bar")
    plt.title("BERTScore F1 by Story and Treatment")
    plt.xlabel("Story")
    plt.ylabel("BERTScore F1")
    plt.legend(title="Treatment")
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "bertscore_by_story_treatment.png"))
        plt.close()
    else:
        plt.show()

    # 2. Plot Fact Presence by story and treatment
    plt.figure(figsize=(12, 6))
    df_pivot = df.pivot_table(
        values="fact_presence_avg",
        index="story_key",
        columns="treatment_key",
        aggfunc="mean",
    )
    ax = df_pivot.plot(kind="bar")
    plt.title("Fact Presence by Story and Treatment")
    plt.xlabel("Story")
    plt.ylabel("Average Fact Presence")
    plt.legend(title="Treatment")
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "fact_presence_by_story_treatment.png"))
        plt.close()
    else:
        plt.show()

    # 3. Scatter plot of BERTScore vs Fact Presence
    plt.figure(figsize=(10, 8))

    # Add colors by treatment
    treatments = df["treatment_key"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(treatments)))

    for i, treatment in enumerate(treatments):
        mask = df["treatment_key"] == treatment
        plt.scatter(
            df.loc[mask, "bertscore_f1"],
            df.loc[mask, "fact_presence_avg"],
            label=treatment,
            color=colors[i],
            alpha=0.7,
        )

    plt.title("BERTScore F1 vs Fact Presence")
    plt.xlabel("BERTScore F1")
    plt.ylabel("Average Fact Presence")
    plt.legend(title="Treatment")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, "bertscore_vs_fact_presence.png"))
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze divergence between initial and final stories"
    )
    parser.add_argument("results_file", help="Path to the simulation results JSON file")
    parser.add_argument("--output", "-o", help="Path to save the metrics to")
    parser.add_argument("--report", "-r", help="Path to save the summary report to")
    parser.add_argument("--visualize", "-v", help="Directory to save visualizations to")

    args = parser.parse_args()

    # Analyze simulations
    metrics = analyze_all_simulations(args.results_file, args.output)

    # Generate summary report
    generate_summary_report(metrics, args.report)

    # Visualize metrics
    if args.visualize:
        visualize_metrics(metrics, args.visualize)


if __name__ == "__main__":
    main()
