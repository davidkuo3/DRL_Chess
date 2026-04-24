# DRL_Chess: Analyzing Self-Play Evolution

## Project Introduction
This project implements a simplified Deep Reinforcement Learning (DRL) architecture to analyze the evolution of self-play strategies in Chess. It focuses on the impact of different exploration strategies (MCTS vs. epsilon-greedy) on behavioral evolution.

## Project Structure
- `openspec/`: Contains the project specifications and change management.
    - `specs/drl_chess.md`: The core system specification.
- `DRL_Chess_Blueprint.pdf`: The original project proposal and architecture blueprint.

## Getting Started
This project uses the **OpenSpec** framework for specification-driven development.
To see the full specification, refer to [openspec/specs/drl_chess.md](./openspec/specs/drl_chess.md).

## Core Architecture
1. **Environment**: 8x8 Board simulation.
2. **Networks**: Policy (p) and Value (v) networks.
3. **Action Selection**: MCTS or Epsilon-Greedy.
4. **Self-Play Loop**: Autonomous training data generation.
