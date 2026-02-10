# PageRank Algorithm Implementation

This repository contains an implementation of the **PageRank algorithm** developed as part of the  
**MAA507 – Mathematics of Internet** course.

The aim of this project is to understand and apply the mathematical foundations behind PageRank,
including graph representation, transition matrices, and iterative rank computation.

---

## 📌 Project Overview

PageRank is an algorithm used to rank nodes in a directed graph based on their importance.
In the context of the web, nodes represent web pages and edges represent hyperlinks.

In this project, we implemented:
- A directed graph model
- The PageRank iterative formula
- Rank convergence through repeated iterations
- Handling of damping factor

The implementation focuses on clarity and correctness rather than optimisation.

---

## 🧮 Mathematical Background

PageRank is defined as:

PR(i) = (1 - d)/N + d · Σ ( PR(j) / outdegree(j) )

Where:
- **d** is the damping factor (typically 0.85)
- **N** is the total number of nodes
- **PR(i)** is the PageRank of node *i*
- The sum is over all nodes *j* that link to *i*

The algorithm iterates until PageRank values converge.

---

## ⚙️ Implementation Details

- The graph is represented as an adjacency structure
- Initial PageRank values are assigned equally
- PageRank values are updated iteratively
- The algorithm runs for a fixed number of iterations or until convergence

The code is written to be easy to follow and directly reflects the mathematical model discussed in the course.

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/aslsumen/pagerank_math.git

## Project Information

Authors: Ayat Mannaa & Asli Sümen  
Course: MAA507 – Mathematics of Internet  
University: Mälardalen University  
Semester: Spring 2026  
