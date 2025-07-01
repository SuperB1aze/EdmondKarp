import numpy as np
import matplotlib.pyplot as plt


class EdmondsKarp:
    def __init__(self, vertices):
        self.vertices = vertices
        self.capacity = np.zeros((vertices, vertices), dtype=int)
        self.residual_capacity = self.capacity.copy()

    def add_edge(self, u, v, capacity):
        """Добавляет ребро в граф с заданной пропускной способностью"""
        self.capacity[u][v] += capacity
        self.residual_capacity[u][v] += capacity

    def bfs(self, source, sink, parent):
        """Поиск в ширину для нахождения увеличивающего пути"""
        visited = np.zeros(self.vertices, dtype=bool)
        queue = [source]
        visited[source] = True

        while queue:
            u = queue.pop(0)
            for v in range(self.vertices):
                if not visited[v] and self.residual_capacity[u][v] > 0:
                    parent[v] = u
                    queue.append(v)
                    visited[v] = True
                    if v == sink:
                        return True
        return False

    def max_flow(self, source, sink):
        """Вычисление максимального потока с использованием алгоритма Эдмондса-Карпа"""
        self.residual_capacity = np.copy(self.capacity)  # Матрица остаточной пропускной способности
        parent = np.full(self.vertices, -1, dtype=int)
        max_flow = 0

        while self.bfs(source, sink, parent):
            # Находим минимальную остаточную пропускную способность на пути
            path_flow = float('Inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, self.residual_capacity[u][v])
                v = u

            # Обновляем остаточную пропускную способность
            v = sink
            while v != source:
                u = parent[v]
                self.residual_capacity[u][v] -= path_flow
                self.residual_capacity[v][u] += path_flow
                v = u

            # Увеличиваем общий поток
            max_flow += path_flow

        return max_flow

    def visualize_graph(self):
        """Визуализация графа"""
        fig, ax = plt.subplots(figsize=(8, 6))
        for u in range(self.vertices):
            for v in range(self.vertices):
                if self.capacity[u][v] > 0:
                    ax.arrow(u, 0, v - u, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
                    ax.text((u + v) / 2, 0.1, f'{self.capacity[u][v]}', color='red', fontsize=12)
        ax.set_xlim(-1, self.vertices)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        plt.show()


# Пример использования
ek = EdmondsKarp(6)
ek.add_edge(0, 1, 16)
ek.add_edge(0, 2, 13)
ek.add_edge(1, 2, 10)
ek.add_edge(1, 3, 12)
ek.add_edge(2, 4, 14)
ek.add_edge(3, 5, 20)
ek.add_edge(4, 3, 7)
ek.add_edge(4, 5, 4)

ek.visualize_graph()