## Project Overview

This project develops a metro route planning system under a weighted-graph framework and presents the full implementation process from the original teaching template to a more complete and user-friendly system.

## Repository Structure

- **Basic**: completes the baseline route-planning functions, including graph construction, station indexing, shortest-path search, and route output.
- **GUI_advanced**: extends the baseline system with a more interactive graphical interface and clearer route visualization.
- **Optional**: further introduces transfer-aware route planning so that the algorithm can consider both travel distance and transfer overhead.
- **Images** and **Videos**: provide visual demonstrations for different stages of the project.

## Core Modeling Idea

The metro network is modeled as a weighted undirected graph:

- each **station** is treated as a node,
- each **connection between adjacent stations** is treated as an edge,
- each edge weight represents the corresponding travel cost.

Based on this formulation, the shortest route between an origin and a destination is computed using a hand-written **Dijkstra algorithm**.

## Stage-by-Stage Development

### 1. Basic Stage

The first stage focuses on completing the essential route-planning pipeline.  
This includes defining the graph data structure, building the station connection graph, supporting station-name queries, and computing the shortest path between any two stations.

### 2. GUI Advanced Stage

The second stage improves usability.  
Instead of only returning textual results, the system provides a more intuitive interactive interface with clearer route presentation, making the program easier to demonstrate and use.

### 3. Optional Stage

The final stage introduces a transfer-aware formulation.  
Instead of searching only in the original station space, the system expands the state representation so that transfer penalties can be incorporated into the route cost. This allows the algorithm to balance pure travel distance and transfer convenience.

## Main Highlights

- A complete weighted-graph metro route planning framework.
- A hand-written shortest-path solver rather than relying on external graph libraries.
- A clearer and more interactive GUI for route display.
- An optional transfer-aware extension for more realistic route recommendation.
- A project page that integrates paper, code, images, video, and poster presentation.

## Notes for Visitors

This page emphasizes the final integrated presentation of the project, while the repository preserves the incremental development path from the original template to the advanced and optional versions.  
Readers can therefore compare how the system evolves from a basic shortest-path solver into a more complete metro route planning application.
