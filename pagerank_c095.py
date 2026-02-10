import numpy as np
import networkx as nx

# -----------------------------
# 1) Build the directed graph
# -----------------------------
G = nx.DiGraph()
G.add_edges_from([
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("C", "A"),
    ("D", "C")
])

nodes = ["A", "B", "C", "D"]
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

# -----------------------------
# 2) Parameters (c = 0.95)
# -----------------------------
c = 0.95
eps = 1e-12
max_iter = 100

v = np.ones(N) / N  # teleportation vector

# -----------------------------
# 3) Transition matrix P (row-stochastic)
# -----------------------------
P = np.zeros((N, N))

for src in nodes:
    out_neighbors = list(G.successors(src))
    prob = 1.0 / len(out_neighbors)
    for dst in out_neighbors:
        P[idx[src], idx[dst]] = prob

print("Transition matrix P:")
print(P)

# -----------------------------
# 4) Initial distribution (all ones, normalized)
# -----------------------------
p = np.ones(N)
p = p / p.sum()

print("\nInitial p0:")
print({nodes[i]: float(p[i]) for i in range(N)})

# -----------------------------
# 5) Iterate until convergence
# -----------------------------
print("\nIterations:")
print("t=0:", {nodes[i]: float(p[i]) for i in range(N)})

for t in range(1, max_iter + 1):
    p_next = c * (p @ P) + (1 - c) * v
    diff = np.linalg.norm(p_next - p, ord=1)

    print(f"t={t}:", {nodes[i]: float(p_next[i]) for i in range(N)}, f"L1 diff={diff:.3e}")

    if diff < eps:
        print(f"\n✅ Converged at t={t}")
        p = p_next
        break

    p = p_next

final_ranks = {nodes[i]: float(p[i]) for i in range(N)}
print("\nFinal converged PageRank:")
print(final_ranks)

# Compare with NetworkX
nx_ranks = nx.pagerank(G, alpha=c, tol=1e-12)
print("\nNetworkX PageRank:")
print(nx_ranks)
