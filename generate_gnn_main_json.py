import os
import json
import numpy as np
import networkx as nx

def load_graph_and_embeddings(filepath):
    with open(filepath, "r") as f:
        n, m, d = map(int, f.readline().split())
        lines = [list(map(int, line.strip().split())) for line in f.readlines()]
        edges = lines[:m]
        embeddings = np.array(lines[m:m+n])
    return n, m, d, edges, embeddings

def generate_question(n, edges, embeddings, num_layers):
    q = f"In an undirected graph, the nodes are numbered from 0 to {n-1}, and every node has an embedding. (i,j) means that node i and node j are connected with an undirected edge.\n"
    q += "Embeddings:\n"
    for i in range(n):
        q += f"node {i}: [{','.join(map(str, embeddings[i]))}]\n"
    q += "The edges are:" + ''.join(f" ({u},{v})" for u, v in edges) + "\n"
    q += "In a simple graph convolution layer, each node's embedding is updated by the sum of its neighbors' embeddings.\n"
    layer_desc = "two layers" if num_layers == 2 else "one layer"
    q += f"Q: What's the embedding of each node after {layer_desc} of simple graph convolution layer?\n"
    q += "A: Please return the answer in the following format:\nThe answer is:\n"
    for i in range(n):
        q += f"node {i}: [x,x]\n"
    return q

def gnn_convolution(G, embeddings, layers):
    emb = embeddings.copy()
    for _ in range(layers):
        new_emb = np.zeros_like(emb)
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                new_emb[node] = np.sum([emb[neigh] for neigh in neighbors], axis=0)
        emb = new_emb
    return emb

def format_answer(embeddings):
    answer = "The answer is:\n"
    for i, vec in enumerate(embeddings):
        vec_str = ','.join(str(int(x)) if x == int(x) else f"{x:.4f}".rstrip('0').rstrip('.') for x in vec)
        answer += f"node {i}: [{vec_str}]\n"
    return answer

def process_mode(mode, count, layer_count, base_idx, output):
    for j in range(count):
        filepath = f"NLgraph/GNN/graph/{mode}/standard/graph{j}.txt"
        n, m, d, edges, embeddings = load_graph_and_embeddings(filepath)

        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)

        question = generate_question(n, edges, embeddings, num_layers=layer_count)
        final_emb = gnn_convolution(G, embeddings, layers=layer_count)
        answer = format_answer(final_emb)

        output[str(base_idx + j)] = {
            "question": question,
            "answer": answer,
            "difficulty": mode
        }

def main():
    output = {}

    process_mode("easy", count=100, layer_count=1, base_idx=0, output=output)

    process_mode("hard", count=140, layer_count=2, base_idx=100, output=output)

    with open("NLgraph/GNN/main.json", "w") as f:
        json.dump(output, f, indent=2)

    print("main.json generated at: NLgraph/GNN/main.json")

if __name__ == "__main__":
    main()
