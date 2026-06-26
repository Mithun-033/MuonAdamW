# MuonAdamW

A hybrid optimizer that combines [Muon](https://github.com/KellerJordan/Muon) and AdamW into a single drop-in optimizer, with automatic parameter routing based on model architecture.

> ⚠️ **Beta — not yet stable.** Known issues are tracked below.

---

## Motivation

Muon excels at optimizing weight matrices via orthogonalized Nesterov momentum, while AdamW remains the go-to for embeddings, norms, and biases. Splitting parameters manually every run is tedious. MuonAdamW does it for you.

---

## Routing Modes

| Mode | AdamW gets | Muon gets |
|---|---|---|
|`general` | params with dim = 1 | params with dim > 1 |
| `transformer` | Embeddings, LayerNorms, biases, final projection heads | All other `dim > 1` weight matrices |
| `cnn` | First conv layer, heads, biases | All other `dim > 1` weight matrices |
| `custom` | Explicitly passed list | Explicitly passed list |

---

## `muon_lr_multiplier`

Controls how Muon's effective LR is set relative to the base `lr`:

| Value | Behavior |
|---|---|
| `"original"` | Uses Keller's implemetation |
| `"match_rms_adamw"` | Scales to match RMS of AdamW updates, Moonshot's implementation |
| `float` | `muon_lr = lr × multiplier` |

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
- [ ] Fix parameter deduplication in `transformer` / `cnn` modes
- [x] LR scheduler compatibility shim
- [ ] PyPI release


---

## License

MIT
