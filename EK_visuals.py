import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from EdmondKarp import EdmondsKarp

# ── утилиты ───────────────────────────────────────────────────────────────────
OUT_DIR = "visuals"
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=300)
    plt.close()

def build_nx_graph(mat, attr="capacity"):
    G = nx.DiGraph()
    n = mat.shape[0]
    for u in range(n):
        for v in range(n):
            if mat[u, v] > 0:
                G.add_edge(u, v, **{attr: int(mat[u, v])})
    return G

def draw_capacity(G, fname):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=700)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(u, v): G[u][v]['capacity'] for u, v in G.edges},
        font_size=9)
    plt.title("Исходные пропускные способности")
    plt.axis("off")
    save_fig(fname)

# ── один шаг Эдмондса-Карпа ───────────────────────────────────────────────────
def run_single_iteration(ek):
    parent = np.full(ek.vertices, -1)
    if not ek.bfs(ek.source, ek.sink, parent):
        return None                # путей больше нет
    path_flow = float('inf')
    v = ek.sink
    while v != ek.source:
        u = parent[v]
        path_flow = min(path_flow, ek.residual_capacity[u, v])
        v = u
    v = ek.sink
    while v != ek.source:
        u = parent[v]
        ek.residual_capacity[u, v] -= path_flow
        ek.residual_capacity[v, u] += path_flow
        v = u
    return path_flow               # величина добавленного потока

def draw_flow(ek, fname, title):
    """Рисует подписи вида flow/capacity по положительным capacity."""
    flow_mat = ek.capacity - ek.residual_capacity
    G = build_nx_graph(ek.capacity)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=700)
    edge_lbl = {(u, v): f"{flow_mat[u, v]}/{ek.capacity[u, v]}"
                for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_lbl, font_size=9)
    plt.title(title)
    plt.axis("off")
    save_fig(fname)

def draw_residual(ek, fname):
    G = build_nx_graph(ek.residual_capacity, attr="res")
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos,
            with_labels=True,
            node_color="lightgrey",
            node_size=700)
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels={(u, v): G[u][v]['res'] for u, v in G.edges},
        font_size=9)
    plt.title("Остаточная сеть")
    plt.axis("off")
    save_fig(fname)

# ── фабрика экземпляров ───────────────────────────────────────────────────────
def make_ek(cap, s=0, t=5):
    ek = EdmondsKarp(cap.shape[0])
    ek.capacity = cap.copy()
    ek.residual_capacity = cap.copy()
    ek.source, ek.sink = s, t
    return ek

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = np.array([
        [0, 10, 10, 0, 0, 0],
        [0, 0, 0, 4, 8, 0],
        [0, 0, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 10],
        [0, 0, 0, 0, 0, 10],
        [0, 0, 0, 0, 0, 0],
    ])

    # Рисунок 1
    draw_capacity(build_nx_graph(cap), "fig1_initial_capacity.png")

    # Рисунок 2
    ek_first = make_ek(cap)
    run_single_iteration(ek_first)
    draw_flow(ek_first, "fig2_after_step1.png",
              "После первой итерации")

    # Рисунки 3 и 4
    ek_full = make_ek(cap)
    max_flow = ek_full.max_flow(ek_full.source, ek_full.sink)
    print("Максимальный поток:", max_flow)

    draw_flow(ek_full, "fig3_final_flow.png",
              "Финальное распределение потоков")
    draw_residual(ek_full, "fig4_residual.png")