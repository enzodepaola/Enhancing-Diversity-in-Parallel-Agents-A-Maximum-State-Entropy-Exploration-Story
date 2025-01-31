# Enhancing-Diversity-in-Parallel-Agents-A-Maximum-State-Entropy-Exploration-Story

Repository for Maximum State Entropy Exploration Via Parallel Agents.

## Requirements

Before running the project, ensure you have the following dependencies installed:

- Python 3.x
- Required Python libraries are indicated in the requirement.txt file

You can install them using:
```bash
pip install -r requirements.txt

-You can train PGSPE with:
python3 main.py

-You can generatean offline replay buffer to use for Batch RL learning with:
python3 replay_buffer_creation.py

-You can train an offline q-learning version from the dataset with:
python3 train_offline_rl.py