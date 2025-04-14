import os
import sys
import json
import numpy as np
import networkx as nx
import re
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from tenacity import retry, stop_after_attempt, wait_random_exponential

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "models")))

from wrappers import (
    call_openai_chat,
    call_deepseek_chat,
    call_anthropic_claude,
)

import argparse
parser = argparse.ArgumentParser(description="GNN")
parser.add_argument('--model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--provider', type=str, default="openai")
parser.add_argument('--mode', type=str, default="easy")
parser.add_argument('--prompt', type=str, default="none")
parser.add_argument('--T', type=int, default=0)
parser.add_argument('--token', type=int, default=2000)
parser.add_argument('--layer', type=int, default=1)
parser.add_argument('--SC', type=int, default=0)
parser.add_argument('--SC_num', type=int, default=5)
parser.add_argument('--dry_run', action='store_true')
args = parser.parse_args()

assert args.prompt in ["CoT", "none", "0-CoT", "LTM", "PROGRAM", "k-shot"]
assert args.layer in [1, 2]

def translate(G, embedding, args):
    edge = list(G.edges())
    n = G.number_of_nodes()
    Q = ''
    if args.prompt in ["CoT", "k-shot"]:
        folder = "NLGraph/GNN/prompt/" if args.layer == 2 else "NLGraph/GNN/one-prompt/"
        with open(folder + args.prompt + "-prompt.txt", "r") as f:
            Q += f.read() + "\n\n\n"
    Q += f"In an undirected graph, the nodes are numbered from 0 to {n-1}, and every node has an embedding. (i,j) means that node i and node j are connected with an undirected edge.\n"
    Q += "Embeddings:\n"
    for i in range(n):
        Q += f"node {i}: [" + ",".join(map(str, embedding[i])) + "]\n"
    Q += "The edges are:" + "".join(f" ({u},{v})" for u, v in edge) + "\n"
    Q += "In a simple graph convolution layer, each node's embedding is updated by the sum of its neighbors' embeddings.\n"
    layers = "two layers" if args.layer == 2 else "one layer"
    Q += f"Q: What's the embedding of each node after {layers} of simple graph convolution layer?\n"
    Q += "A: The answer is:\nnode 0: [x,x]\nnode 1: [x,x] ...\n"

    match args.prompt:
        case "0-CoT": Q += "Let's think step by step:\n"
        case "LTM": Q += "Let's break down this problem:\n"
        case "PROGRAM": Q += "Let's solve the problem by a Python program:\n"
    return Q

def extract_node_embeddings(text, n):
    try:
        parsed = re.findall(r"node\s+(\d+):\s*\[(.*?)\]", text)
        parsed = sorted(((int(i), [float(x.strip()) for x in vec.split(",")]) for i, vec in parsed), key=lambda x: x[0])
        return np.array([vec for _, vec in parsed])
    except:
        return None

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def predict(Q_list, args):
    answer_list, raw_list = [], []

    if args.dry_run:
        print("\n========== DRY RUN: Prompt Only ==========\n")
        for i, q in enumerate(Q_list):
            print(f"\n--- Prompt #{i+1} ---\n{q}\n")
        return ["(dry run - no API call)" for _ in Q_list], ["(dry run - no raw)" for _ in Q_list]

    for q in Q_list:
        if args.provider == "openai":
            content, _, raw = call_openai_chat(args.model, q, return_usage=True)
        elif args.provider == "anthropic":
            content, _, raw = call_anthropic_claude(args.model, q, return_usage=True)
        elif args.provider == "deepseek":
            content, _, raw = call_deepseek_chat(args.model, q, return_usage=True)
        else:
            raise ValueError(f"Unsupported provider {args.provider}")
        answer_list.append(content)
        raw_list.append(raw)
    return answer_list, raw_list

def save_logs(model, mode, prompt, SC, Q_list, res, answers, raw_list):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    timestamp = bj_dt.strftime("%Y%m%d---%H-%M")
    folder = f"log/GNN/{model}-{mode}-{timestamp}-{prompt}"
    if SC:
        folder += "+SC"
    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, "res.npy"), res)
    np.save(os.path.join(folder, "answer.npy"), answers)

    with open(os.path.join(folder, "prompt.txt"), "w") as f:
        for Q in Q_list:
            f.write(Q + "\n\n")
        f.write(f"Acc: {res.sum()}/{len(res)}\n")

    full_folder = f"log/GNN/fullresponses"
    os.makedirs(full_folder, exist_ok=True)
    with open(os.path.join(full_folder, f"{model}-{mode}-{timestamp}-{prompt}.txt"), "w", encoding="utf-8") as f:
        for raw in raw_list:
            f.write("=== RAW RESPONSE ===\n")
            f.write(raw + "\n\n")

def main():
    with open("NLGraph/GNN/main.json", "r") as f:
        main_data = json.load(f)

    mode_index = {"easy": 0, "hard": 100}
    g_num = {"easy": 10, "hard": 10}[args.mode]
    base_index = mode_index[args.mode]

    res, answer, raw_list = [], [], []
    batch_num = 10

    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        Q_list, std_list, G_list = [], [], []

        for j in range(i * batch_num, min(g_num, (i + 1) * batch_num)):
            idx = base_index + j
            with open(f"NLgraph/GNN/graph/{args.mode}/standard/graph{j}.txt", "r") as f:
                n, m, d = map(int, next(f).split())
                arr = [list(map(int, line.split())) for line in f]
                edge, emb = arr[:m], arr[m:m+n]
                G = nx.Graph()
                G.add_edges_from((u, v) for u, v in edge)
                emb = np.array(emb)

                layer1 = np.zeros((n, d))
                for node in G.nodes:
                    neighbors = list(G[node])
                    if neighbors:
                        layer1[node] = np.sum([emb[nei] for nei in neighbors], axis=0)

                std = layer1
                if args.layer == 2:
                    std = np.zeros((n, d))
                    for node in G.nodes:
                        neighbors = list(G[node])
                        if neighbors:
                            std[node] = np.sum([layer1[nei] for nei in neighbors], axis=0)

                Q = translate(G, emb, args)
                Q_list.append(Q)
                std_list.append(std)
                G_list.append(G)

        sc = args.SC_num if args.SC else 1
        sc_votes = [0] * len(Q_list)

        for k in range(sc):
            print(f"Running call #{k + 1}...")
            answer_list, raw_batch = predict(Q_list, args)
            for j, (ans, raw) in enumerate(zip(answer_list, raw_batch)):
                answer.append(ans)
                raw_list.append(raw)
                parsed = extract_node_embeddings(ans, G_list[j].number_of_nodes())
                if parsed is not None and parsed.shape == std_list[j].shape:
                    if np.allclose(parsed, std_list[j], atol=0.01):
                        sc_votes[j] += 1

        for v in sc_votes:
            res.append(1 if v * 2 >= sc else 0)

    res = np.array(res)
    answer = np.array(answer)
    save_logs(args.model, args.mode, args.prompt, args.SC, Q_list, res, answer, raw_list)
    print("Final Accuracy:", res.sum(), "/", len(res))

if __name__ == "__main__":
    main()
