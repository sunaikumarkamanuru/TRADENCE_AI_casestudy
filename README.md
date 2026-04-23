# Self-Pruning Neural Network — Tredence AI Internship

A feed-forward neural network that **learns to remove its own unnecessary
weights during training** using learnable sigmoid gates + L1 sparsity
regularisation on CIFAR-10.

## Files
| File | Purpose |
|---|---|
| `self_pruning_network.py` | Complete solution — run this |
| `requirements.txt` | Dependencies |
| `gate_distribution.png` | Generated after training |

## Run
```bash
pip install -r requirements.txt
python self_pruning_network.py
```
