# MuonAdamW

A hybrid optimizer that combines [Muon](https://github.com/KellerJordan/Muon) and AdamW into a single drop-in optimizer, with automatic parameter routing based on model architecture.

> ⚠️ **Beta — not yet stable.** Known issues are tracked below.

---

## Optimizer definition
```python

from MuonAdamW import MuonAdamW, MuonArgs, AdamwArgs

adamw_args = AdamwArgs(
    weight_decay = 0.05,    # Change any default args of respective optimizer
    eps = 1e-8
)
muon_args = MuonArgs(
    ns_steps = 6,
    momentum = 0.8
)
optimizer = MuonAdamW(
    model,                  # Pass the model, not the iterator
    lr = learning_rate,
    mode = "transformer",   # Choose the mode for parameter split
    adamw_args = adamw_args,
    muon_args = muon_args
)

```

## Motivation

Muon excels at optimizing weight matrices via orthogonalized Nesterov momentum, while AdamW remains the go-to for embeddings, norms, and biases. Splitting parameters manually every run is tedious. MuonAdamW does it for you.

---

## Routing Modes

| Mode | AdamW gets | Muon gets |
|---|---|---|
|`general` | params with dim = 1 | params with dim > 1 |
| `transformer` | Embeddings, LayerNorms, biases, final projection heads | All other `dim > 1` weight matrices |p
| `custom` | Explicitly passed list | Explicitly passed list |

---

## `muon_lr_multiplier`

Controls how Muon's effective LR is set relative to the base `lr`:

| Value | Behavior |
|---|---|
| `"original"` | Uses Keller's implemetation |
| `"match_rms_adamw"` | Scales to match RMS of AdamW updates, Moonshot's implementation |
| `float` | `muon_lr = lr × multiplier` (not recommended) |

---

## Checkpointing

```python
# Save
torch.save(optimizer.state_dict(), "checkpoint.pt")

# Load
optimizer.load_state_dict(torch.load("checkpoint.pt"))
```

---

## Known Limitations (Beta)

- **LR schedulers** — passing `MuonAdamW` to `torch.optim.lr_scheduler.*` will not update sub-optimizer LRs. Workaround: step schedulers on `optimizer.adamw` and `optimizer.muon` directly.
- **`optimizer.state`** — always empty; state lives in the sub-optimizers. Use `state_dict()` for inspection.
- Duplicate parameter deduplication fix in progress for `transformer` and `cnn` modes.

---

## Roadmap
- [x] state_dict/load_state_dict functionality added and tested
- [x] docs added
- [x] General mode tested and benchmarked
- [x] Fix parameter deduplication in `transformer` 
- [x] LR scheduler compatibility shim
- [ ] PyPI release


---

## License

MIT
