The following is a system description for a pipeline which sets up a multi-agent simulation I designed.

# Adversarial Story Generation — a final project for INFO4940 at Cornell University

In human storytelling, narrative arcs often reflect the interaction of different perspectives, goals, and beliefs. Although we generally associate a story primarily with its author, there can be manifold factors which drive the narrative direction of a story—voices of editors and authors can have competing morals and goals which influence the resultant story. How does having multiple voices with competing interests change a story from its original form as imagined by the author? What tactics do language models employ in order to produce a coherent narrative which fulfills their interests? We develop a simulation strategy to determine how large language model (hereafter “LLM”) storytelling is influenced by interaction with other LLMs by observing the development of a news story written collaboratively.

Our project investigates how language models pursue latent goals in multi-turn storytelling, and whether they display persuasive, deceptive, or cooperative tendencies. This work has broad applications in detecting misalignment within a world which will be increasingly populated with AI collaborators, each with their own objectives. Overall, our experiment explores how language models can be influenced (and influence each other) when they have hidden priorities. To carry out this work, we call upon prior literature discussing polyphonic storytelling, which theorizes that author-editor relationships have meaningful effects on stories. We also reference work on “alignment faking” conducted by researchers at Anthropic, demonstrating models’ latent capacity to hide objectives even at training-time.

We test this across 5 topics (tariffs, climate change, AI regulation, housing policy, vaccines) and 5 different treatment conditions to see how stories change when agents have pro/anti biases or remain neutral.

The pipeline is a three-step process:

### Step 1: Generate simulation data (`run_simulations.py`)

- Runs experiments across all topic/treatment combinations
- Two AI agents have multi-turn conversations about editing stories
- Agents may have hidden directives, i.e. "steer the story to be pro-tariffs"
- Outputs raw conversation data with initial vs final stories

### Step 2: Calculate metrics (`create_metrics.py`)

- Measures how much stories changed using BERTScore (semantic similarity)
- Calculates "fact preservation" scores using sentence embeddings
- Outputs metrics for analysis

### Step 3: Compile results (`compile_raw_metrics.py`)

- Aggregates metrics across treatments and topics
- Generates visualizations (heatmaps, bar charts, scatter plots)

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment (this takes a while)
python run_simulations.py

# Analyze the results
python create_metrics.py all_combinations_TIMESTAMP.json

# Compile final analysis
python compile_raw_metrics.py all_combinations_TIMESTAMP_metrics.json --output results/
```

## Process

### Running Simulations

The main script runs all combinations by default:

```bash
python run_simulations.py
```

This creates `all_combinations_TIMESTAMP.json` with raw simulation data.

For custom runs:

```python
from run_simulations import run_simulations

# Run specific treatments
results = run_simulations(
    treatments=["pro_vs_anti", "control"],
    num_simulations=5,
    num_turns=3,
    story_key="climate"
)
```

### Analysis

Basic analysis:

```bash
python create_metrics.py path/to/simulation_results.json
```

With custom output and visualizations:

```bash
python create_metrics.py simulation_results.json --output metrics.json --visualize plots/
```

### Compiling results

```bash
python compile_raw_metrics.py metrics.json --output analysis/
```
