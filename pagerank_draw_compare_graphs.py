import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------
def build_transition_matrix(G, nodes):
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    P = np.zeros((N, N), dtype=float)

    for src in nodes:
        out_neighbors = list(G.successors(src))
        if len(out_neighbors) == 0:
            P[idx[src], :] = 1.0 / N  # dangling node handling
        else:
            prob = 1.0 / len(out_neighbors)
            for dst in out_neighbors:
                P[idx[src], idx[dst]] = prob
    return P

def pagerank_power_iteration(P, c, eps=1e-12, max_iter=5000):
    """
    Returns:
      p_final (np.array),
      iterations_to_converge (int),
      diffs (list of L1 diffs per iteration),
      history (list of p vectors, optional use)
    """
    N = P.shape[0]
    v = np.ones(N) / N

    # initial distribution: all ones normalized
    p = np.ones(N, dtype=float)
    p = p / p.sum()

    diffs = []
    history = [p.copy()]

    for t in range(1, max_iter + 1):
        p_next = c * (p @ P) + (1 - c) * v
        diff = np.linalg.norm(p_next - p, ord=1)
        diffs.append(diff)
        history.append(p_next.copy())
        p = p_next

        if diff < eps:
            return p, t, diffs, history

    return p, max_iter, diffs, history

def draw_graph(G, title):
    plt.figure()
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, arrows=True)
    plt.title(title)
    plt.show()

# -----------------------------
# Graph definitions (same nodes)
# -----------------------------
nodes = ["A", "B", "C", "D"]

# Graph-1 (your original)
G1 = nx.DiGraph()
G1.add_edges_from([
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("C", "A"),
    ("D", "C")
])

# Graph-2 (same nodes, different topology)
G2 = nx.DiGraph()
G2.add_edges_from([
    ("A", "B"),
    ("A", "C"),
    ("B", "A"),
    ("B", "C"),
    ("C", "A"),
    ("C", "D"),
    ("D", "C")
])

# Build transition matrices
P1 = build_transition_matrix(G1, nodes)
P2 = build_transition_matrix(G2, nodes)

# =========================================================
# PART 1: Graph-1, compare different c values + plots
# =========================================================
cs = [0.50, 0.85, 0.95]
eps = 1e-12
max_iter = 5000

results_g1 = []
convergence_g1 = {}  # c -> diffs

for c in cs:
    p_final, iters, diffs, _ = pagerank_power_iteration(P1, c=c, eps=eps, max_iter=max_iter)
    convergence_g1[c] = diffs

    final = {nodes[i]: float(p_final[i]) for i in range(len(nodes))}
    ordered = sorted(final.items(), key=lambda x: x[1], reverse=True)
    rank_order = " > ".join([n for n, _ in ordered])

    results_g1.append({
        "graph": "G1",
        "c": c,
        "iterations_to_converge": iters,
        "A": final["A"],
        "B": final["B"],
        "C": final["C"],
        "D": final["D"],
        "rank_order": rank_order
    })

df_g1 = pd.DataFrame(results_g1).sort_values("c").reset_index(drop=True)

print("\n========================")
print("PART 1: Graph-1 (G1) different c values")
print("========================")
print(df_g1)

# Draw Graph-1 structure
draw_graph(G1, "Graph-1 structure (G1)")

# Convergence plot for Graph-1 across c
plt.figure()
for c in cs:
    diffs = convergence_g1[c]
    plt.plot(range(1, len(diffs) + 1), diffs, label=f"c = {c}")
plt.yscale("log")
plt.xlabel("Iteration t")
plt.ylabel("L1 difference ||p(t) - p(t-1)||_1 (log scale)")
plt.title("Graph-1 (G1): Convergence for different c values")
plt.legend()
plt.show()

# =========================================================
# PART 2: Compare Graph-1 vs Graph-2 at c = 0.85 + plots
# =========================================================
c_fixed = 0.85

p1_final, it1, diffs1, _ = pagerank_power_iteration(P1, c=c_fixed, eps=eps, max_iter=max_iter)
p2_final, it2, diffs2, _ = pagerank_power_iteration(P2, c=c_fixed, eps=eps, max_iter=max_iter)

final1 = {nodes[i]: float(p1_final[i]) for i in range(len(nodes))}
final2 = {nodes[i]: float(p2_final[i]) for i in range(len(nodes))}

ordered1 = " > ".join([n for n, _ in sorted(final1.items(), key=lambda x: x[1], reverse=True)])
ordered2 = " > ".join([n for n, _ in sorted(final2.items(), key=lambda x: x[1], reverse=True)])

df_compare = pd.DataFrame([
    {
        "graph": "G1",
        "c": c_fixed,
        "iterations_to_converge": it1,
        "A": final1["A"], "B": final1["B"], "C": final1["C"], "D": final1["D"],
        "rank_order": ordered1
    },
    {
        "graph": "G2",
        "c": c_fixed,
        "iterations_to_converge": it2,
        "A": final2["A"], "B": final2["B"], "C": final2["C"], "D": final2["D"],
        "rank_order": ordered2
    }
])

print("\n========================")
print("PART 2: Compare Graph-1 (G1) vs Graph-2 (G2) at c = 0.85")
print("========================")
print(df_compare)

# Draw both graph structures
draw_graph(G1, "Graph-1 structure (G1)")
draw_graph(G2, "Graph-2 structure (G2)")

# Convergence plot: G1 vs G2 at c=0.85
plt.figure()
plt.plot(range(1, len(diffs1) + 1), diffs1, label="G1 (c=0.85)")
plt.plot(range(1, len(diffs2) + 1), diffs2, label="G2 (c=0.85)")
plt.yscale("log")
plt.xlabel("Iteration t")
plt.ylabel("L1 difference ||p(t) - p(t-1)||_1 (log scale)")
plt.title("Convergence comparison at c=0.85: Graph-1 vs Graph-2")
plt.legend()
plt.show()
