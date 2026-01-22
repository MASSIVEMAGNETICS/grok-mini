# Predictive Autograd Engine (PAE) Manual
**Version:** v1.0.0-PREDICTIVE-AUTOGRAD-GODCORE  
**Backend:** NumPy (CPU)  
**Paradigm:** Reverse-mode autodiff (dynamic computation graphs)

## 0. Design Goals
PAE is engineered for:
- **Correctness first** (broadcasting gradients, stable softmax/xent, numerical checks)
- **Debuggability** (graph capture, anomaly detection, gradient health)
- **Extensibility** (clear op registry, single-file drop-in, backend abstraction points)
- **Determinism** (seed control and explicit graph lifetimes)

## 1. Core Types

### 1.1 `Tensor`
A `Tensor` wraps:
- `data: np.ndarray`
- `grad: Optional[np.ndarray]`
- `requires_grad: bool`
- graph metadata: `_ctx` (operation context), `_parents`, `_op`

#### Construction
- `Tensor(data, requires_grad=False, name=None, dtype=None)`
- `Tensor.zeros(shape, requires_grad=False)`
- `Tensor.ones(shape, requires_grad=False)`
- `Tensor.randn(shape, requires_grad=False, seed=None, scale=1.0)`
- `Tensor.uniform(shape, low=0.0, high=1.0, requires_grad=False, seed=None)`

#### Properties
- `t.shape`, `t.ndim`, `t.dtype`
- `t.T` transpose (2D shortcut)

#### Graph control
- `t.detach()` returns a Tensor not connected to the graph
- `Tensor.no_grad()` context manager disables graph building

### 1.2 Predictive planning / graph introspection
- `t.graph_summary()` prints node counts and op histogram
- `Tensor.set_anomaly_mode(True/False)` enables NaN/Inf checks with op context
- `Tensor.set_predictive_mode(True/False)` enables op-stat collection & fusion hints
- `Tensor.get_predictive_report()` returns a report of common patterns and suggestions

## 2. Autograd Semantics

### 2.1 Backprop
- `t.backward(grad=None, retain_graph=False)`
Rules:
- If `t` is scalar and `grad` is None, uses `1.0`.
- Accumulates gradients into `leaf.grad` (NumPy arrays).
- Topological traversal over recorded parents.

### 2.2 Broadcasting gradients
PAE follows NumPy broadcasting:
- For an operation `z = x + y` where `y` is broadcast, PAE reduces `dz/dy` along broadcast axes.

## 3. Supported Operations

### 3.1 Elementwise ops
- `+ - * / **`
- `neg()`
- `exp(), log()`
- `tanh(), sigmoid(), relu(), gelu()`

### 3.2 Reductions
- `sum(axis=None, keepdims=False)`
- `mean(axis=None, keepdims=False)`
- `var(axis=None, keepdims=False, unbiased=False)`
- `std(axis=None, keepdims=False, unbiased=False)`

### 3.3 Shape ops
- `reshape(*shape)`
- `transpose(*axes)` / `permute(*axes)`
- `squeeze(axis=None)`
- `unsqueeze(axis)`
- slicing: `t[a:b, ...]`
- `concat([t1,t2,...], axis=...)`

### 3.4 Linear algebra
- `matmul` via `@`
- `softmax(axis=-1)` (stable)
- `log_softmax(axis=-1)` (stable)
- `cross_entropy(logits, targets, reduction='mean')` (stable; fused path recommended)
- `layer_norm(normalized_shape, eps=1e-5)`

### 3.5 Regularization
- `dropout(p=0.5, training=True, seed=None)` (deterministic with seed)

## 4. Utilities

### 4.1 Optimizer utilities (minimal)
- `Tensor.zero_grads(params)`
- `Tensor.clip_grad_norm(params, max_norm, eps=1e-12)`

(You can implement Adam/SGD externally; this engine keeps gradients correct and stable.)

## 5. Troubleshooting

### 5.1 "My grads are None"
- Ensure `requires_grad=True` on leaves you want gradients for.
- Ensure you didn't call `.detach()` or run under `Tensor.no_grad()`.

### 5.2 "NaN/Inf in gradients"
- Turn on anomaly mode:
  - `Tensor.set_anomaly_mode(True)`
- Use gradient clipping:
  - `Tensor.clip_grad_norm(params, 1.0)`
- Prefer stable losses:
  - use `Tensor.cross_entropy` (it uses log-sum-exp stabilization)

### 5.3 "Broadcasting gradients look wrong"
- Verify shapes. If the forward uses implicit broadcasting, the backward will reduce over broadcast axes.
- If you reshaped or summed, confirm `axis` and `keepdims` usage.

## 6. SAVE3 Scaffold (embedded)
This library embeds a minimal SAVE3-compatible scaffold (TrustModelBeta + HandoverEnvelope) to comply with your file standards. It is inert unless you call it. It exists to standardize provenance and handover metadata when you later split engine responsibilities across agents/modules.

## 7. Validation
Run:
```bash
python predictive_autograd_engine.py
```

It performs:
- Numerical gradient checks (finite differences) on key ops
- Softmax + cross-entropy sanity checks
- Broadcasting gradient checks
- Matmul gradient checks

If a test fails, it prints the op context and the offending node.
