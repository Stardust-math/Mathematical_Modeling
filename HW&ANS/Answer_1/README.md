# Metro Route Planning Project

## 1. Project Overview

This project implements an interactive **metro route planning system** based on **weighted graph modeling** and incremental system enhancement.
The repository is organized into multiple versions, including a **Basic** version, a **GUI_advanced** version, and an **Optional** version with transfer-aware routing.

The core workflow is:

- load metro station metadata and adjacency-distance matrices,
- build a weighted graph for the selected city,
- compute the shortest path between two stations,
- visualize the metro network and the resulting route in a GUI.

Compared with the Basic version, the advanced versions further improve user interaction and route modeling.
The `GUI_advanced` version adds zooming, dragging, and fit-view operations, while the `Optional` version adds **transfer-time-aware shortest path planning** for datasets that include line information.

---

## 2. Mathematical Models

### 2.1 Weighted Graph Shortest Path

For each city, the metro system is modeled as a weighted graph:

\[
G = (V, E, w),
\]

where:

- \(V\) is the set of stations,
- \(E\) is the set of connections between stations,
- \(w(u,v) > 0\) is the edge weight, usually the distance between two adjacent stations.

Given a source station \(s\) and a destination station \(t\), the basic routing problem is:

\[
\min_{P:s \to t} \sum_{(u,v) \in P} w(u,v).
\]

This problem is solved by **Dijkstra's algorithm**.

### 2.2 Transfer-Aware Shortest Path

In the optional version, the route cost can also include a transfer penalty.
The algorithm uses an expanded state representation:

\[
(station, current\_line).
\]

If a route changes from one metro line to another, a fixed transfer cost \(\tau\) is added.
Thus, the route objective becomes:

\[
\min_{P:s \to t}
\left(
\sum_{(u,v) \in P} w(u,v)
+
\tau \cdot N_{\text{transfer}}(P)
\right),
\]

where \(N_{\text{transfer}}(P)\) is the number of line changes along the route.

This formulation is implemented only when line metadata is available in the dataset.

---

## 3. Project Structure

```text
project/
├── Basic/
│   ├── code/
│   │   ├── main.py
│   │   ├── metro_algorithm.py
│   │   └── gui.py
│   └── data/
│       ├── Barcelona/
│       ├── Beijing/
│       ├── Berlin/
│       ├── New_tokyo/
│       ├── Osaka/
│       ├── Paris/
│       ├── Shanghai/
│       ├── london-tube/
│       ├── README.md
│       └── summary.tsv
├── GUI_advanced/
│   ├── code/
│   │   ├── main.py
│   │   ├── metro_algorithm.py
│   │   └── gui.py
│   └── data/
├── Optional/
│   ├── code/
│   │   ├── main.py
│   │   ├── metro_algorithm.py
│   │   └── gui.py
│   └── data/
├── README.md
└── environment.yml
```

### File description

- `main.py`
  Entry point of each version.

- `metro_algorithm.py`
  Core graph-related implementation, including data loading, graph construction, Dijkstra shortest path, and transfer-aware routing.

- `gui.py`
  Graphical user interface for city selection, station selection, route solving, network drawing, and interaction.

- `data/`
  Metro network datasets for different cities.

- `environment.yml`
  Conda environment configuration file.

- `README.md`
  Project documentation.

---

## 4. Environment Configuration

This project uses Python together with a small set of numerical, visualization, and GUI libraries.

### Recommended environment

- Python 3.12.2
- NumPy
- Matplotlib
- Tk

### Create the environment

Use Conda to create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate metro-route-planning
```

---

## 5. How to Run

Each version has its own `main.py` entry file in the corresponding `code/` directory.
The program automatically locates the sibling `data/` directory.

### Run the Basic version

```bash
cd Basic/code
python main.py
```

### Run the GUI_advanced version

```bash
cd GUI_advanced/code
python main.py
```

### Run the Optional version

```bash
cd Optional/code
python main.py
```

---
