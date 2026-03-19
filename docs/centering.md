# Centering API

This document describes the centering behavior for `AssemblyPart` and `Assembly`.

## Why This Update

Centering may not reach very strict thresholds (for example `1e-5`) when frame estimation depends on fitted parametric models.
To make convergence criteria explicit and tunable, all `center()` methods now support separate tolerances for rotation and translation.

## API

### AssemblyPart.center

```python
def center(
    self,
    max_try=10,
    atol_rot: float = 1e-5,
    atol_trans: float = 1e-5,
    verbose: bool = False,
)
```

Parameters:
- `max_try`: Maximum centering iterations.
- `atol_rot`: Rotation convergence tolerance in radians, checked against Euler xyz residuals.
- `atol_trans`: Translation convergence tolerance, checked against centroid translation residual.
- `verbose`: Print iteration-by-iteration progress.

Convergence rule:
- Rotation residual is within `atol_rot` **and** translation residual is within `atol_trans`.

### Assembly.center

```python
def center(
    self,
    part_index,
    max_try=10,
    atol_rot: float = 1e-5,
    atol_trans: float = 1e-5,
    verbose: bool = False,
)
```

Behavior:
- Uses `parts[part_index]` as the reference to compute centering transforms.
- Applies the same rigid transform to every part in the assembly.
- Uses the same convergence criteria (`atol_rot`, `atol_trans`) as `AssemblyPart.center()`.

## Recommended Settings

For parametric/fitted models (for example CCCP bundle workflows), start with:

```python
part.center(max_try=30, atol_rot=1e-5, atol_trans=1e-4, verbose=True)
```

If centering is still not converged:
- Increase `max_try` (for example `50`).
- Relax `atol_trans` to `2e-4` or `5e-4` depending on task sensitivity.

## Example

```python
part = CCCPHelixBundle.from_param(helix_num=2)
part.structure = structure
part.mask = {...}

part.center(max_try=30, atol_rot=1e-5, atol_trans=1e-4, verbose=True)
part.fit(verbose=True)
```
