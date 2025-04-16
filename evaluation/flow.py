import os
import sys
import json
import re
import numpy as np
import networkx as nx
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_random_exponential

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.wrappers import (
    call_openai_chat,
    call_deepseek_chat,
    call_anthropic_claude,
)

# Argument parsing
import argparse
parser = argparse.ArgumentParser(description="maximum flow")
parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--provider', type=str, default="openai")
parser.add_argument('--mode', type=str, default="easy")
parser.add_argument('--prompt', type=str, default="none")
parser.add_argument('--T', type=int, default=0)
parser.add_argument('--token', type=int, default=400)
parser.add_argument('--SC', type=int, default=0)
parser.add_argument('--SC_num', type=int, default=5)
parser.add_argument('--dry_run', action='store_true')
args = parser.parse_args()

def extract_max_flow_value(answer_text: str) -> int:
    match = re.search(r"\bis\s+(\d+)\b", answer_text)
    if not match:
        raise ValueError(f"Could not extract flow value from: {answer_text}")
    return int(match.group(1))

def translate(G, q, args):
    edge = list(G.edges())
    n = G.number_of_nodes()
    Q = ''
    if args.prompt in ["CoT", "k-shot"]:
        with open("NLGraph/flow/prompt/" + args.prompt + "-prompt.txt", "r") as f:
            exemplar = f.read()
        Q += exemplar + "\n\n\n"
    Q += f"In a directed graph, the nodes are numbered from 0 to {n-1}, and the edges are:\n"
    for (u, v) in edge:
        cap = G[u][v]["capacity"]
        Q += f"an edge from node {u} to node {v} with capacity {cap},\n"
    Q = Q.rstrip(',\n') + ".\n"
    Q += f"Q: What is the maximum flow from node {q[0]} to node {q[1]}?\n"
    Q += f"A: Please answer at the start clearly with: 'The maximum flow from node {q[0]} to node {q[1]} is (a number goes here).'\n"

    match args.prompt:
        case "0-CoT": Q += "Let's think step by step:\n"
        case "LTM": Q += "Let's break down this problem:\n"
        case "PROGRAM": Q += "Let's solve the problem by a Python program and get the Final answer from there \n"
    return Q

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def predict_batch(Q_list):
    if args.dry_run:
        print("\n========== DRY RUN: Prompt Only ==========\n")
        for i, prompt in enumerate(Q_list):
            print(f"\n--- Prompt #{i+1} ---\n{prompt}\n")
        return ["(dry run - no API call)" for _ in Q_list], ["(dry run - no raw)" for _ in Q_list]

    answer_list = []
    raw_list = []
    for prompt in Q_list:
        if args.provider == "openai":
            response, _, raw = call_openai_chat(args.model, prompt, return_usage=True)
        elif args.provider == "anthropic":
            response, _, raw = call_anthropic_claude(args.model, prompt, return_usage=True)
        elif args.provider == "deepseek":
            response, _, raw = call_deepseek_chat(args.model, prompt, return_usage=True)
        else:
            raise ValueError(f"Unsupported provider: {args.provider}")
        answer_list.append(response)
        raw_list.append(raw)
    return answer_list, raw_list

def log(Q_list, res, answers, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    timestamp = bj_dt.strftime("%Y%m%d---%H-%M")
    folder = f"log/flow/{args.model}-{args.mode}-{timestamp}-{args.prompt}"
    if args.SC:
        folder += "+SC"
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "res.npy"), res)
    np.save(os.path.join(folder, "answer.npy"), answers)
    with open(os.path.join(folder, "prompt.txt"), "w") as f:
        for Q in Q_list:
            f.write(Q + "\n\n")
        f.write(f"Acc: {res.sum()}/{len(res)}\n")
        print(args, file=f)

def main():
    with open("NLGraph/flow/main.json", "r") as f:
        main_data = json.load(f)

    mode_index = {"easy": 0, "hard": 150}
    g_num = {"easy": 150, "hard": 200}[args.mode]
    # g_num = {"easy": 150, "hard": 200}[args.mode]
    base_index = mode_index[args.mode]

    res, answers = [], []
    batch_num = 20

    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        Q_list, correct_values = [], []
        for j in range(i * batch_num, min(g_num, (i + 1) * batch_num)):
            index = base_index + j
            with open(f"NLgraph/flow/graph/{args.mode}/standard/graph{j}.txt", "r") as f:
                n, m = map(int, f.readline().split())
                edge_data = [list(map(int, line.strip().split())) for line in f]
                edges, q, std = edge_data[:-2], edge_data[-2], edge_data[-1][0]
                G = nx.DiGraph()
                G.add_nodes_from(range(n))
                for u, v, cap in edges:
                    G.add_edge(u, v, capacity=cap)
                Q = translate(G, q, args)
                Q_list.append(Q)
                correct_ans = main_data[str(index)]["answer"]
                correct_values.append(extract_max_flow_value(correct_ans))

        sc = args.SC_num if args.SC else 1
        sc_votes = [0] * len(Q_list)

        for k in range(sc):
            print(f"Running call #{k+1}...")
            answer_list, _ = predict_batch(Q_list)
            print(f"Finished API call")
            for j, ans in enumerate(answer_list):
                answers.append(ans)
                try:
                    pattern = rf"The maximum flow from node \d+ to node \d+ is (\d+)"
                    match = re.search(pattern, ans, re.IGNORECASE)
                    if match:
                        pred_val = int(match.group(1))
                        if pred_val == correct_values[j]:
                            sc_votes[j] += 1
                    else:
                        print(f"Pattern not matched in answer:\n{ans}\n")
                except Exception as e:
                    print(f"Error parsing answer: {ans}\n{e}")

        for count in sc_votes:
            res.append(1 if count * 2 >= sc else 0)

    res = np.array(res)
    answers = np.array(answers)
    log(Q_list, res, answers, args)
    print("Final Accuracy:", res.sum(), "/", len(res))

if __name__ == "__main__":
    main()
