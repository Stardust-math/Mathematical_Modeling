# Copyright 2026, Yumeng Liu @ USTC

"""
地铁网络算法模块 —— 数据加载、图构建、Dijkstra 求解
"""

import csv
import heapq
from pathlib import Path

import numpy as np


# ============================================================
# Graph 数据结构
# ============================================================

class Graph:
    """
    简单的无向加权图。

    需要实现的接口
    -------------
    - add_node(node_id, **attrs) : 添加节点
    - add_edge(u, v, weight)     : 添加无向边
    - neighbors(node_id)         : 返回邻居字典 {neighbor_id: weight}
    - number_of_nodes()          : 返回节点数
    - number_of_edges()          : 返回边数
    - edges()                    : 返回所有边列表 [(u, v, weight), ...]

    属性
    ----
    nodes : dict[int, dict]
        节点字典，{node_id: {"name": str, ...}}。
        GUI 会读取此属性来获取节点信息，请确保 add_node 时正确填充。

    提示
    ----
    你可以自由选择底层数据结构（邻接表、邻接矩阵、边列表等）。
    """

    def __init__(self):
        self.nodes = {}
        self.adj = {}

    def add_node(self, node_id, **attrs):
        """
        添加节点。

        Parameters
        ----------
        node_id : int
            节点编号。
        **attrs
            节点属性，例如 name="StationA"。
        """
        self.nodes[node_id] = dict(attrs)
        if node_id not in self.adj:
            self.adj[node_id] = {}

    def add_edge(self, u, v, weight=1.0):
        """
        添加无向边 (u, v)，权重为 weight。
        """
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)

        self.adj[u][v] = float(weight)
        self.adj[v][u] = float(weight)

    def neighbors(self, node_id):
        """
        返回 node_id 的邻居字典 {neighbor_id: weight}。

        若节点不存在或无邻居，返回空字典。
        """
        return self.adj.get(node_id, {})

    def number_of_nodes(self):
        """返回图中节点数量。"""
        return len(self.nodes)

    def number_of_edges(self):
        """返回图中边的数量（每条无向边只计一次）。"""
        total = sum(len(neigh) for neigh in self.adj.values())
        return total // 2

    def edges(self):
        """
        返回所有边的列表 [(u, v, weight), ...]，每条边只出现一次。

        GUI 的绘图函数会调用此方法来绘制网络边。
        """
        edge_list = []
        seen = set()

        for u, neighbors in self.adj.items():
            for v, weight in neighbors.items():
                key = tuple(sorted((u, v)))
                if key not in seen:
                    seen.add(key)
                    edge_list.append((u, v, weight))

        return edge_list


# ============================================================
# 数据加载
# ============================================================

def load_station_map(tsv_path: str) -> dict[int, str]:
    """读取 station-id-map.tsv，返回 {id: name} 映射。"""
    stations: dict[int, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            stations[int(row["id"])] = row["name"]
    return stations


def load_adjacency_matrix(csv_path: str) -> np.ndarray:
    """读取 adjacency-distance.csv，返回 N×N numpy 矩阵。"""
    return np.loadtxt(csv_path, delimiter=",")


def load_station_lines(tsv_path: str) -> dict[str, set[str]]:
    """读取 station-lines.txt，返回 {station_name: {line1, line2, ...}}。"""
    station_lines: dict[str, set[str]] = {}
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="	")
        for row in reader:
            name = row["station"]
            lines = {x.strip() for x in row["lines"].split(",") if x.strip()}
            if name not in station_lines:
                station_lines[name] = set()
            station_lines[name].update(lines)
    return station_lines


def build_graph(stations: dict[int, str], adj: np.ndarray) -> Graph:
    """
    根据站点映射和邻接矩阵构建加权图。

    Parameters
    ----------
    stations : dict[int, str]
        站点 id → 名称映射（id 从 1 开始）。
    adj : np.ndarray
        N×N 邻接距离矩阵，adj[i,j] > 0 表示站点 i+1 与 j+1 之间有边。

    Returns
    -------
    Graph
        带权无向图，节点属性 name 为站名，边权 weight 为距离。

    提示
    ----
    - 使用 Graph.add_node(node_id, name=...) 添加节点
    - 使用 Graph.add_edge(u, v, weight=...) 添加边
    - 矩阵下标从 0 开始，站点 id 从 1 开始
    """
    G = Graph()

    for sid, name in stations.items():
        G.add_node(sid, name=name)

    N = adj.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] > 0:
                u = i + 1
                v = j + 1
                G.add_edge(u, v, weight=adj[i, j])

    return G


# ============================================================
# Dijkstra 最短路径
# ============================================================

def dijkstra(G: Graph, src: int, dst: int) -> tuple[float, list[int]]:
    """
    实现 Dijkstra 求 src → dst 最短路径。

    Parameters
    ----------
    G : Graph
        带权图。
    src : int
        起点站点 id。
    dst : int
        终点站点 id。

    Returns
    -------
    (cost, path) : (float, list[int])
        cost 为最短距离，path 为站点 id 序列（含起终点）。
        若不可达，返回 (float("inf"), [])。

    提示
    ----
    - 使用 G.neighbors(u) 获取邻居字典 {neighbor_id: weight}
    - 使用 heapq 实现最小堆
    - 使用前驱字典 prev 回溯路径
    """
    if src not in G.nodes or dst not in G.nodes:
        return float("inf"), []

    dist = {node: float("inf") for node in G.nodes}
    prev = {node: None for node in G.nodes}
    dist[src] = 0.0

    pq = [(0.0, src)]
    visited = set()

    while pq:
        cur_dist, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u == dst:
            break

        for v, w in G.neighbors(u).items():
            if v in visited:
                continue

            new_dist = cur_dist + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    if dist[dst] == float("inf"):
        return float("inf"), []

    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return dist[dst], path

def dijkstra_with_transfer(
    G: Graph,
    src: int,
    dst: int,
    station_lines: dict[int, set[str]],
    transfer_time: float,
) -> tuple[float, list[int]]:
    """
    带换乘代价的 Dijkstra。

    状态定义为 (station_id, current_line)，表示当前到达某站时所在的线路。
    走一条边时，需要从两站共享的线路中选一条；若该线路与 current_line
    不同，则额外增加 transfer_time。
    """
    if src not in G.nodes or dst not in G.nodes:
        return float("inf"), []

    src_lines = station_lines.get(src, set())
    dst_lines = station_lines.get(dst, set())
    if not src_lines or not dst_lines:
        return dijkstra(G, src, dst)

    dist: dict[tuple[int, str], float] = {}
    prev: dict[tuple[int, str], tuple[int, str] | None] = {}
    pq: list[tuple[float, int, str]] = []

    for line in src_lines:
        state = (src, line)
        dist[state] = 0.0
        prev[state] = None
        heapq.heappush(pq, (0.0, src, line))

    best_dst_state = None
    best_dst_cost = float("inf")

    while pq:
        cur_dist, u, cur_line = heapq.heappop(pq)
        state = (u, cur_line)
        if cur_dist > dist.get(state, float("inf")):
            continue

        if u == dst:
            best_dst_state = state
            best_dst_cost = cur_dist
            break

        for v, w in G.neighbors(u).items():
            common_lines = station_lines.get(u, set()) & station_lines.get(v, set())
            if not common_lines:
                continue

            for next_line in common_lines:
                penalty = 0.0 if next_line == cur_line else float(transfer_time)
                new_dist = cur_dist + float(w) + penalty
                next_state = (v, next_line)
                if new_dist < dist.get(next_state, float("inf")):
                    dist[next_state] = new_dist
                    prev[next_state] = state
                    heapq.heappush(pq, (new_dist, v, next_line))

    if best_dst_state is None:
        return float("inf"), []

    state_path = []
    cur_state = best_dst_state
    while cur_state is not None:
        state_path.append(cur_state)
        cur_state = prev[cur_state]
    state_path.reverse()

    path = [state_path[0][0]]
    for station_id, _line in state_path[1:]:
        if station_id != path[-1]:
            path.append(station_id)

    return best_dst_cost, path


# ============================================================
# MetroSystem 高层封装
# ============================================================

class MetroSystem:
    """封装单个城市的地铁系统：加载数据、构建图、求解路径。"""

    def __init__(self, data_dir: str | Path):
        data_dir = Path(data_dir)
        self.city = data_dir.name

        tsv = next(data_dir.glob("*station-id-map.tsv"))
        csv_f = next(data_dir.glob("*adjacency-distance.csv"))

        self.stations = load_station_map(str(tsv))
        adj = load_adjacency_matrix(str(csv_f))
        self.graph = build_graph(self.stations, adj)

        self.name_to_id: dict[str, int] = {
            name: sid for sid, name in self.stations.items()
        }

        self.station_lines_by_id: dict[int, set[str]] = {}
        line_file = next(data_dir.glob("*station-lines.txt"), None)
        if line_file is not None:
            station_lines_by_name = load_station_lines(str(line_file))
            for name, sid in self.name_to_id.items():
                self.station_lines_by_id[sid] = station_lines_by_name.get(name, set())

    def sorted_station_names(self) -> list[str]:
        """返回按字母排序的站名列表。"""
        return sorted(self.stations.values())

    def shortest_path(
        self,
        src_name: str,
        dst_name: str,
        use_transfer_time: bool = False,
        transfer_time: float = 5.0,
    ) -> tuple[float, list[int]]:
        """
        求两站之间的最短路径。

        Parameters
        ----------
        src_name : str
            起点站名。
        dst_name : str
            终点站名。
        use_transfer_time : bool
            是否考虑换乘代价。
        transfer_time : float
            每次换乘增加的固定代价。

        Returns
        -------
        (cost, path) : (float, list[int])
            cost 为路径总代价，path 为站点 id 序列。
        """
        src_id = self.name_to_id[src_name]
        dst_id = self.name_to_id[dst_name]

        if use_transfer_time and self.station_lines_by_id:
            return dijkstra_with_transfer(
                self.graph, src_id, dst_id, self.station_lines_by_id, transfer_time
            )

        return dijkstra(self.graph, src_id, dst_id)


def detect_cities(data_root: str | Path) -> list[str]:
    """扫描 data_root 下所有包含数据文件的城市子目录。"""
    data_root = Path(data_root)
    cities: list[str] = []
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and list(d.glob("*adjacency-distance.csv")):
            cities.append(d.name)
    return cities
