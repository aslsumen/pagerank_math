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
# 2) Parameters (c = 0.50)
# -----------------------------
c = 0.50
eps = 1e-12          # convergence tolerance (L1 difference)
max_iter = 100    # safety cap

# Teleportation vector (uniform)
v = np.ones(N) / N

# -----------------------------
# 3) Transition matrix P (row-stochastic)
#    rows=source, cols=destination
# -----------------------------
P = np.zeros((N, N), dtype=float)

for src in nodes:
    out_neighbors = list(G.successors(src))
    if len(out_neighbors) == 0:
        # dangling node: distribute uniformly
        P[idx[src], :] = 1.0 / N
    else:
        prob = 1.0 / len(out_neighbors)
        for dst in out_neighbors:
            P[idx[src], idx[dst]] = prob

print("Transition matrix P (rows=source, cols=destination) order [A,B,C,D]:")
print(P)

# -----------------------------
# 4) Initial distribution: all ones, then normalize
# -----------------------------
p = np.ones(N, dtype=float)
p = p / p.sum()

print("\nInitial p0 (all ones, normalized):")
print({nodes[i]: float(p[i]) for i in range(N)})

# -----------------------------
# 5) Iterate and print ALL t values until convergence
# -----------------------------
print("\nIterations:")
print("t=0:", {nodes[i]: float(p[i]) for i in range(N)})

for t in range(1, max_iter + 1):
    p_next = c * (p @ P) + (1 - c) * v
    diff = np.linalg.norm(p_next - p, ord=1)  # L1 norm

    print(f"t={t}:", {nodes[i]: float(p_next[i]) for i in range(N)}, f"  L1 diff={diff:.3e}")

    if diff < eps:
        print(f"\n✅ Converged at t={t} with L1 diff={diff:.3e}")
        p = p_next
        break

    p = p_next
else:
    print(f"\n⚠️ Did not converge within {max_iter} iterations. Last L1 diff={diff:.3e}")

final_ranks = {nodes[i]: float(p[i]) for i in range(N)}
print("\nFinal converged PageRank (our iteration):")
print(final_ranks)

# Optional: compare with NetworkX's converged PageRank
nx_ranks = nx.pagerank(G, alpha=c, tol=1e-12, max_iter=1000)
print("\nNetworkX PageRank (comparison):")
print(nx_ranks)
