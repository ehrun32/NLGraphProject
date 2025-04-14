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

def extract_node_embeddings(text, n):
    try:
        parsed = re.findall(r"node\s+(\d+):\s*\[(.*?)\]", text)
        parsed = sorted(((int(i), [float(x.strip()) for x in vec.split(",")]) for i, vec in parsed), key=lambda x: x[0])
        return np.array([vec for _, vec in parsed])
    except:
        return None

def translate(G, embedding, args):
    edge = list(G.edges())
    n = G.number_of_nodes()
    Q = ''

    if args.prompt in ["CoT", "k-shot"]:
        folder = "one-prompt" if args.layer == 1 else "prompt"
        with open(f"NLGraph/GNN/{folder}/{args.prompt}-prompt.txt", "r") as f:
            exemplar = f.read()
        Q += exemplar + "\n\n\n"

    Q += "In an undirected graph, the nodes are numbered from 0 to " + str(n-1) + ", and every node has an embedding. (i,j) means that node i and node j are connected with an undirected edge.\n"
    Q += "Embeddings:\n"
    for i in range(n):
        Q += f"node {i}: [{','.join(str(x) for x in embedding[i])}]\n"
    Q += "The edges are:" + ''.join(f" ({u},{v})" for u, v in edge) + "\n"
    Q += "In a simple graph convolution layer, each node's embedding is updated by the sum of its neighbors' embeddings.\n"
    layer_phrase = "two layers" if args.layer == 2 else "one layer"
    Q += f"Q: What's the embedding of each node after {layer_phrase} of simple graph convolution layer?\n"
    Q += "A: Please return the answer in the following format:\nThe answer is:\n"
    for i in range(n):
        Q += f"node {i}: [x,x]\n"
    match args.prompt:
        case "0-CoT": Q += "Let's think step by step:\n"
        case "LTM": Q += "Let's break down this problem:\n"
        case "PROGRAM": Q += "Let's solve the problem by a Python program:\n"
    return Q

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
def predict(Q_list, args):
    answer_list, raw_list = [], []

    if args.dry_run:
        print("\n========== DRY RUN: Prompt Only ==========\n")
        for i, q in enumerate(Q_list):
            print(f"\n--- Prompt #{i + 1} ---\n{q}\n")
        return ["(dry run - no API call)" for _ in Q_list], ["(dry run - no raw)" for _ in Q_list]

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

def log(Q_list, res, answer, raw_list, args):
    utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    timestamp = bj_dt.strftime("%Y%m%d---%H-%M")
    folder = f"log/GNN/{args.model}-{args.mode}-{timestamp}-{args.prompt}"
    if args.SC:
        folder += "+SC"
    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, "res.npy"), res)
    np.save(os.path.join(folder, "answer.npy"), answer)

    with open(os.path.join(folder, "prompt.txt"), "w") as f:
        for q in Q_list:
            f.write(q + "\n\n")
        f.write(f"Acc: {res.sum()}/{len(res)}\n")
        print(args, file=f)

    full_folder = "log/GNN/fullresponses"
    os.makedirs(full_folder, exist_ok=True)
    with open(os.path.join(full_folder, f"{args.model}-{args.mode}-{timestamp}-{args.prompt}.txt"), "w", encoding="utf-8") as f:
        for raw in raw_list:
            f.write("=== RAW RESPONSE ===\n")
            if hasattr(raw, "choices") and hasattr(raw.choices[0].message, "content"):
                f.write(raw.choices[0].message.content + "\n\n")
            else:
                f.write(str(raw) + "\n\n")


def main():
    with open("NLGraph/GNN/main.json", "r") as f:
        main_data = json.load(f)

    mode_index = {"easy": 0, "hard": 100}
    g_num = {"easy": 2, "hard": 1}[args.mode]
    base_idx = mode_index[args.mode]

    res, answer, raw_all = [], [], []
    batch_num = 10

    for i in tqdm(range((g_num + batch_num - 1) // batch_num)):
        Q_list, GT_list, G_list = [], [], []
        for j in range(i * batch_num, min(g_num, (i + 1) * batch_num)):
            idx = base_idx + j
            with open(f"NLgraph/GNN/graph/{args.mode}/standard/graph{j}.txt", "r") as f:
                n, m, d = map(int, f.readline().split())
                lines = [list(map(int, line.split())) for line in f]
                edges, embedding = lines[:m], lines[m:m+n]
                G = nx.Graph()
                G.add_nodes_from(range(n))
                for u, v in edges:
                    G.add_edge(u, v)

                GT_array = extract_node_embeddings(main_data[str(idx)]["answer"], n)
                if args.layer == 2:
                    emb = embedding
                    for _ in range(2):
                        emb = [np.sum([emb[n] for n in G[i]], axis=0) for i in range(n)]
                else:
                    emb = [np.sum([embedding[n] for n in G[i]], axis=0) for i in range(n)]

                Q = translate(G, embedding, args)
                Q_list.append(Q)
                GT_list.append(GT_array)
                G_list.append(G)

        sc = args.SC_num if args.SC else 1
        sc_votes = [0] * len(Q_list)

        for k in range(sc):
            print(f"Running call #{k + 1}...")
            ans_list, raw_list = predict(Q_list, args)
            for j in range(len(Q_list)):
                ans = ans_list[j]
                answer.append(ans)
                raw_all.append(raw_list[j])
                pred = extract_node_embeddings(ans, n=GT_list[j].shape[0])
                if pred is not None and pred.shape == GT_list[j].shape and np.allclose(pred, GT_list[j], atol=0.01):
                    sc_votes[j] += 1

        for v in sc_votes:
            res.append(1 if v * 2 >= sc else 0)

    res = np.array(res)
    answer = np.array(answer)
    log(Q_list, res, answer, raw_all, args)
    print("Final Accuracy:", res.sum(), "/", len(res))

if __name__ == "__main__":
    main()
