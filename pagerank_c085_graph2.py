import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1) Build the directed graph (GRAPH-2)
#    Same nodes (A,B,C,D) => page count is controlled
# -------------------------------------------------
G = nx.DiGraph()
G.add_edges_from([
    ("A", "B"),
    ("A", "C"),
    ("B", "A"),   # changed vs Graph-1: B links back to A
    ("B", "C"),
    ("C", "A"),   # keeps a strong component among A,B,C
    ("C", "D"),   # changed vs Graph-1: C now links to D
    ("D", "C")    # changed vs Graph-1: D and C form a 2-cycle
])

nodes = ["A", "B", "C", "D"]
idx = {n: i for i, n in enumerate(nodes)}
N = len(nodes)

# -------------------------------------------------
# 2) Parameters (c = 0.85)
# -------------------------------------------------
c = 0.85                  # damping factor
eps = 1e-12               # convergence tolerance
max_iter = 100            # safety cap

# Teleportation vector
v = np.ones(N) / N

# -------------------------------------------------
# 3) Transition matrix P (row-stochastic)
#    rows = source, cols = destination
# -------------------------------------------------
P = np.zeros((N, N))

for src in nodes:
    out_neighbors = list(G.successors(src))
    if len(out_neighbors) == 0:
        P[idx[src], :] = 1.0 / N
    else:
        prob = 1.0 / len(out_neighbors)
        for dst in out_neighbors:
            P[idx[src], idx[dst]] = prob

print("Transition matrix P (rows=source, cols=destination):")
print(P)

# -------------------------------------------------
# 4) Initial distribution: all ones → normalized
# -------------------------------------------------
p = np.ones(N)
p = p / p.sum()

print("\nInitial distribution p0:")
print({nodes[i]: float(p[i]) for i in range(N)})

# -------------------------------------------------
# 5) Power iteration (print ALL t values)
# -------------------------------------------------
print("\nIterations:")
print(f"t=0: { {nodes[i]: float(p[i]) for i in range(N)} }")

history = [p.copy()]

for t in range(1, max_iter + 1):
    p_next = c * (p @ P) + (1 - c) * v
    diff = np.linalg.norm(p_next - p, ord=1)

    print(
        f"t={t}: "
        f"{ {nodes[i]: float(p_next[i]) for i in range(N)} } "
        f"L1 diff = {diff:.3e}"
    )

    history.append(p_next.copy())
    p = p_next

    if diff < eps:
        print(f"\n✅ Converged at iteration {t}")
        break
else:
    print("\n⚠️ Did not converge within max_iter")

# -------------------------------------------------
# 6) Final PageRank results
# -------------------------------------------------
final_ranks = {nodes[i]: float(p[i]) for i in range(N)}
ordered = sorted(final_ranks.items(), key=lambda x: x[1], reverse=True)

print("\nFinal converged PageRank (manual iteration):")
print(final_ranks)

print("\nRanking order:")
for n, score in ordered:
    print(f"{n}: {score:.12f}")

# -------------------------------------------------
# 7) Compare with NetworkX PageRank
# -------------------------------------------------
nx_ranks = nx.pagerank(G, alpha=c, tol=1e-12)
print("\nNetworkX PageRank:")
print(nx_ranks)

print("\nAbsolute differences |manual − networkx|:")
print({n: abs(final_ranks[n] - nx_ranks[n]) for n in nodes})

# -------------------------------------------------
# 8) Draw the graph (node size ∝ PageRank)
# -------------------------------------------------
pos = nx.spring_layout(G, seed=42)
node_sizes = [8000 * final_ranks[n] for n in nodes]

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=node_sizes,
    node_color="lightblue",
    arrows=True,
    arrowsize=20
)

plt.title("PageRank graph2 (c = 0.85)")
plt.show()
