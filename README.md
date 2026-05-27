# Koopman Operator Methods: DMD & EDMD

Companion code for the seminar paper "Dynamic Mode Decomposition
and Its Connection to the Koopman Operator
for Nonlinear Dynamical System" by Lukas Kramer and Matteo Mämpel.

---

## Structure

```
.
├── src/
│   ├── dmd.py          # Core: DMD, EDMD, KoopmanVisualizer classes
│   ├── exp1.py         # Experiment 1 — Linear baseline (3D oscillator + decay)
│   ├── exp2.py         # Experiment 2 — High-dimensional noise & SVD truncation
│   └── exp3.py         # Experiment 3 — Nonlinear 2D system, DMD vs. EDMD
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

All scripts must be run from the **project root** as modules:

```bash
python3 -m src.exp1   # Linear system: DMD exact recovery
python3 -m src.exp2   # Noise truncation: full-rank vs. truncated SVD
python3 -m src.exp3   # Nonlinear system: DMD vs. EDMD comparison
```

---

## Experiments Overview

| Experiment | System | Key Concept |
|---|---|---|
| `exp1` | 3D linear (rotation + decay) | Exact DMD recovery on a linear system |
| `exp2` | 3D linear embedded in 20D + noise | SVD truncation for noise filtering |
| `exp3` | 2D nonlinear (quintic coupling) | EDMD with lifted observables vs. classical DMD |

---

## Core Classes (`src/dmd.py`)

### `DMD`
Classical exact DMD via SVD-based projection.

```python
dmd = DMD(energy_threshold=0.999)   # or pass target_rank=r
dmd.fit(snapshots)                  # shape: (n_states, n_timesteps)
x_pred = dmd.predict(n_steps=50)    # shape: (n_states, n_steps)
Key attributes after fitting: `dmd.Lambda` (eigenvalues), `dmd.Phi` (modes), `dmd.Sigma` (singular values).
```
### `EDMD`
Extended DMD with user-defined observable dictionary.

```python
def my_observables(state):
    x, y = state
    return np.array([x, y, x**2, x*y])  # lift into dictionary space

edmd = EDMD(observables_fn=my_observables)
edmd.fit(snapshots_X, snapshots_Y)       # shifted snapshot pairs
x_pred = edmd.predict(snapshots_X, n_steps=50)
```

### `KoopmanVisualizer`
Static plotting utility

---

## Experiment 3: Open vs. Closed Observable Set

`exp3.py` has a flag at the top:

```python
OPEN = False  # True → includes x^(1/4) in observables (open system)
              # False → closed observable set [x, y, x^5]
```

Toggling this demonstrates the effect of an **incomplete observable basis** on EDMD accuracy.
