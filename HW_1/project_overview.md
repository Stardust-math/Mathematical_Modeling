### Core Modeling Idea

For each city, the metro network is represented as a weighted graph $G=(V,E,w)$, where stations are vertices, adjacent connections are edges, and edge weights denote travel distances.

The basic route-planning task is

$$
\min_{P:s\to t}\sum_{(u,v)\in P} w(u,v),
$$

which is solved by a hand-written Dijkstra algorithm.

### What Was Completed

- **Basic version:** completed graph construction, station-based query, and shortest-path computation.
- **GUI_advanced version:** improved usability with **zooming**, **dragging**, and **fit-view** operations, making the network easier to inspect and interact with.
- **Optional version:** extended the planner from pure distance minimization to **transfer-aware routing** when line metadata is available.

### Transfer-Aware Extension

To make the recommended route more realistic for passengers, the optional model adds a transfer penalty:

$$
\min_{P:s\to t}\left(
\sum_{(u,v)\in P} w(u,v) + \tau\, N_{\text{transfer}}(P)
\right),
$$

where $\tau$ is the transfer penalty and $N_{\text{transfer}}(P)$ is the number of line changes along the route.

This means the system can balance **distance efficiency** and **transfer convenience**, rather than optimizing only one criterion.

### Demonstration Focus

A representative Beijing case compares the ordinary shortest path and the transfer-aware route from **Beigongmen** to **Guomao** under the setting $\tau = 1.0$.

On this page, the project is shown from three complementary perspectives:

- **Abstract:** the overall problem setting and project stages
- **This overview:** a concise explanation of the model and improvements
- **Image / video demos and poster:** visual evidence of the interface and the final outcome

### Why the Project Is Meaningful

This homework develops from a classical shortest-path exercise into a more complete interactive system. It not only solves the mathematical routing problem, but also improves how users **see**, **compare**, and **use** the solution in practice.
