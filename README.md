# Self-Pruning Neural Network — Tredence AI Internship

A feed-forward neural network that **learns to remove its own unnecessary
weights during training** using learnable sigmoid gates + L1 sparsity
regularisation on CIFAR-10.

## Files
| File | Purpose |
|---|---|
| `self_pruning_network.py` | Complete solution — run this |
| `REPORT.md` | Written analysis |
| `requirements.txt` | Dependencies |
| `gate_distribution.png` | Generated after training |

## Run
```bash
pip install -r requirements.txt
python self_pruning_network.py
```

## How it works
```
Normal:   output = W × x
Prunable: gates  = sigmoid(gate_scores)     ← learned, in (0,1)
          output = (W × gates) × x

gate → 0  =  weight pruned
gate → 1  =  weight kept

Loss = CrossEntropy + λ × Σ(gates)
```

## Results
| Lambda | Accuracy | Sparsity |
|---|---|---|
| 0.0001 (Low) | ~54–58% | ~15–30% |
| 0.0010 (Mid) | ~50–54% | ~35–55% |
| 0.0100 (High)| ~44–50% | ~60–80% |

