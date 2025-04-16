# Reinforcement-Tuned Language Models for Graph Reasoning

### _(An Extension of NLGraph)_

This repository extends the original [NLGraph benchmark](https://arxiv.org/abs/2305.10037) introduced in the paper **"Can Language Models Solve Graph Problems in Natural Language?"** by Heng Wang et al., NeurIPS 2023. The original work evaluated GPT-based models on graph problems described in natural language.

This work evaluates modern **reasoning-tuned large language models (LLMs)** such as **DeepSeek R1**, **Claude 3.7 Sonnet**, and **OpenAI o3-mini** across several graph problems: **cycle detection**, **maximum flow**, and **GNN**. Only did those 3 due to cost measurements.

I gratefully acknowledge the NLGraph authors and build directly on their benchmark and codebase.

---

## Demo

[Click here to watch the demo video](https://drive.google.com/file/d/1TAYWiE-96qztf7lAogr77ci096IQF6Ia/view?usp=sharing)

---

## Environment Setup

Using Conda:

```bash
conda env create -f environment.yml
conda activate NLGraph
```

Or using `pip` if Conda is unavailable:

```bash
pip install -r requirements.txt
```

Or

```bash
pip install numpy networkx tqdm tenacity anthropic openai python-dotenv
```

---

## API Keys

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_key
DEEPSEEK_API_KEY=your_deepseek_key
ANTHROPIC_API_KEY=your_anthropic_key
```

---

## Evaluation

Run the evaluation for a specific task:

```bash
python evaluation/cycle.py --model <model_name> --provider <provider_name> --prompt <prompt_type> --mode <difficulty>
```

### Example Commands

```bash
# DeepSeek-R1 on Cycle Detection (easy)
python evaluation/cycle.py --model deepseek-reasoner --provider deepseek --prompt Algorithm --mode easy

# DeepSeek-R1 on Cycle Detection (hard)
python evaluation/cycle.py --model deepseek-reasoner --provider deepseek --prompt Instruct --mode hard

# DeepSeek-R1 on Cycle Detection (hard + CoT-SC)

python evaluation/cycle.py --model deepseek-reasoner --provider deepseek --prompt CoT --mode hard --SC 1 --SC_num 5

# o3-mini (OpenAI) on Max-Flow
python evaluation/flow.py --model o3-mini-2025-01-31 --provider openai --prompt none --mode easy

# Claude 3.7 Sonnet on GNN
python evaluation/gnn.py --model claude-3-7-sonnet-20250219 --provider anthropic --prompt CoT --mode hard
```

---

## Supported Prompting Modes

Available prompting techniques via the `--prompt` argument:

- `"none"` – Zero-Shot
- `"CoT"` – Chain-of-Thought
- `"0-CoT"` – Zero-Shot CoT
- `"k-shot"` – Few-Shot In-Context
- `"LTM"` – Least-to-Most Prompting
- `"PROGRAM"` – Code-based Reasoning
- `"Instruct"` – Instruction-Based Prompt
- `"Algorithm"` – Algorithmic Prompting
- `"Recitation"` – Subquestion Prompting
- `"medium-CoT"` and `"hard-CoT"` – Difficulty-specific CoT styles

---

## Self-Consistency Mode (--SC)

Enable self-consistency by passing the --SC flag along with the number of samples to aggregate using --SC_num. This runs the model multiple times and returns the most frequent answer (majority vote)

```bash
# DeepSeek-R1 on Cycle Detection (hard + CoT-SC)
python evaluation/cycle.py --model deepseek-reasoner --provider deepseek --prompt CoT --mode hard --SC 1 --SC_num 5
```

This command evaluates Cycle Detection on hard mode using DeepSeek R1 with Chain-of-Thought prompting and 5 self-consistency samples.

---

## Dataset Structure

Each task directory (`cycle/`, `flow/`, `GNN/`) contains:

- `main.json` — Ground-truth answers
- `graph/{easy,medium,hard}/` — Graph inputs grouped by difficulty

---

## Citation (Original NLGraph Paper)

If you use this repository or benchmark, please cite the original NLGraph paper:

```bibtex
@inproceedings{
wang2023can,
title={Can Language Models Solve Graph Problems in Natural Language?},
author={Heng Wang and Shangbin Feng and Tianxing He and Zhaoxuan Tan and Xiaochuang Han and Yulia Tsvetkov},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=UDqHhbqYJV}
}
```

---
