#!/usr/bin/env python3
"""Generate a Phantora performance database on a GPU, WITHOUT building Phantora.

A Phantora perf DB is just `(op, shape) -> time`. The op+shape *keys* are
GPU-independent; only the *times* are GPU-specific. So this tool takes the keys
from an existing reference DB (e.g. tests/perfdb/l40s/), re-profiles each kernel
on the local GPU with **stock PyTorch**, and writes a fresh DB in the same CSV
format. No C stubs, no Rust simulator, no patched PyTorch -- just `torch`.

It mirrors phantora/src/torch_estimate.rs: the same op dispatch, the same
allocation aliasing (operands of equal size+dtype share one buffer within an op),
and kernel-only device timing via torch.profiler (matching Phantora's CUPTI
sum). Verified to reproduce a Phantora-recorded DB: per-op times match to ~1%
median and a replayed preset's simulated iteration time is identical. Pass
--wall to use CUDA-event wall time instead (includes launch overhead).

Usage:
    python3 bench.py --ref tests/perfdb/l40s --out tests/perfdb/my_gpu
    python3 bench.py --ref tests/perfdb/l40s            # --out defaults to the GPU name

Replay the result with Phantora: `config_gen.py ... --perf-db <out-name>`.
"""
import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

# tch::Kind serialized name -> torch dtype
DTYPE = {
    "Uint8": torch.uint8, "Int8": torch.int8, "Int16": torch.int16,
    "Int": torch.int32, "Int64": torch.int64, "Half": torch.float16,
    "Float": torch.float32, "Double": torch.float64, "Bool": torch.bool,
    "BFloat16": torch.bfloat16,
    # complex tensors appear in Generic ops (e.g. TorchTitan's RoPE: view_as_real, _conj)
    "ComplexFloat": torch.complex64, "ComplexDouble": torch.complex128, "ComplexHalf": torch.complex32,
}

# Compact-tensor dtype code -> Kind name. Mirrors DTYPE_CODES in
# phantora/phantora/src/perf_db.rs; keep the two in sync.
CODE_TO_KIND = {
    "f32": "Float", "f64": "Double", "f16": "Half", "bf16": "BFloat16",
    "b8": "Bool", "i32": "Int", "i64": "Int64", "i16": "Int16",
    "i8": "Int8", "u8": "Uint8",
}


def expand(v):
    """Inverse of perf_db.rs `compact_value`: a compact tensor `[code,[dims]]`
    -> `{shape,dtype}`, and `{"R":[k,[period]]}` -> the period repeated k times.
    Leaves the legacy verbose `{shape,dtype}` form untouched."""
    if isinstance(v, dict):
        if len(v) == 1 and isinstance(v.get("R"), list) and len(v["R"]) == 2:
            k, period = v["R"]
            out = []
            for _ in range(k):
                out.extend(expand(e) for e in period)
            return out
        return {kk: expand(val) for kk, val in v.items()}
    if isinstance(v, list):
        if len(v) == 2 and isinstance(v[0], str) and v[0] in CODE_TO_KIND and isinstance(v[1], list):
            return {"shape": v[1], "dtype": CODE_TO_KIND[v[0]]}
        return [expand(e) for e in v]
    return v


# Operand buffers. Upstream Phantora's record mode (torch_estimate.rs `allocate`/
# `tensor_cache`) lets operands of the same (numel, dtype) share one buffer. Real
# training never aliases them, and for memory-bound multi-operand ops (Adam's
# `_foreach_addcdiv_`/`_addcmul_`/`_lerp_`, grad `add_`) sharing cuts DRAM traffic by
# up to half: measured on A100 (cluster-sim tools/accuracy), it made those ops 30-50%
# too fast. PERFDB_ALIAS=0 makes operands distinct (an op whose distinct operands do
# not fit in memory falls back to shared buffers). The default stays 1 (shared) so
# that shapes merged into an existing *_v1 DB match the timings already in it;
# record a whole new DB generation with PERFDB_ALIAS=0.
ALIAS = os.environ.get("PERFDB_ALIAS", "1") == "1"
_ACACHE = {}


def _make(shape, dt, device):
    if dt.is_floating_point or dt.is_complex:
        return torch.randn(shape, dtype=dt, device=device)
    if dt == torch.bool:
        return torch.randint(0, 2, shape, device=device).to(torch.bool)
    return torch.randint(0, 128, shape, dtype=dt, device=device)


def alloc(ti, device="cuda", requires_grad=False, share=None):
    """Allocate a tensor matching a TensorInfo {shape, dtype}, reusing a cached
    buffer of the same (numel, dtype) like Phantora does (autograd leaves excepted)."""
    shape = ti["shape"]
    dt = DTYPE.get(ti["dtype"], torch.float32)
    if requires_grad:  # autograd leaves must be distinct, not aliased views
        return _make(shape, dt, device).detach().requires_grad_(True)
    if shape == [] and dt in (torch.float64, torch.int64):
        # a 0-dim double/long operand of a typed op is a wrapped Python scalar
        # (x.mul_(0.5), x + 1): PyTorch keeps those on the CPU, which selects the
        # scalar kernel; on the GPU it would take a slower tensor-tensor path
        return _make(shape, dt, "cpu")
    if not (ALIAS if share is None else share):
        try:
            return _make(shape, dt, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()  # too big to keep distinct: share as upstream does
    numel = 1
    for d in shape:
        numel *= d
    key = (numel, dt)
    buf = _ACACHE.get(key)
    if buf is None:
        buf = _make([numel], dt, device)
        _ACACHE[key] = buf
    return buf.view(shape)


def is_ti(x):
    return isinstance(x, dict) and "shape" in x and "dtype" in x


# op name -> function(payload) returning a zero-arg thunk that runs the op.
# payload is the JSON value under the variant name (dict for newtype/struct,
# list for tuple/Vec). Tensors are allocated up front (outside timing).
def build(op, payload):
    P = payload
    _ACACHE.clear()  # fresh per op; aliases same-(numel,dtype) operands within this op
    if op == "Generic":
        return build_generic(P)

    def two():            # tuple of two TensorInfo
        return alloc(P[0]), alloc(P[1])

    if op == "MM":
        a, b = two(); return lambda: torch.mm(a, b)
    if op == "MatMul":
        a, b = two(); return lambda: torch.matmul(a, b)
    if op == "BMM":
        a, b = two(); return lambda: torch.bmm(a, b)
    if op == "Linear":
        a = alloc(P[0]); w = alloc(P[1]); bias = alloc(P[2]) if P[2] else None
        return lambda: F.linear(a, w, bias)
    if op == "AddMM":
        a, b, c = alloc(P[0]), alloc(P[1]), alloc(P[2]); return lambda: torch.addmm(a, b, c)
    if op == "BAddBMM":
        a, b, c = alloc(P[0]), alloc(P[1]), alloc(P[2]); return lambda: torch.baddbmm(a, b, c)
    if op in ("Mul", "Add", "Div"):
        a, b = two()
        return {"Mul": lambda: a * b, "Add": lambda: a + b, "Div": lambda: a / b}[op]
    if op == "Mul_":
        a, b = two(); return lambda: a.mul_(b)
    if op == "Add_":
        a, b = two(); return lambda: a.add_(b)
    if op == "MulScalar":
        a = alloc(P); return lambda: a * 2
    if op == "MulScalar_":
        a = alloc(P); return lambda: a.mul_(2)
    if op == "DivScalar":
        a = alloc(P); return lambda: a / 2
    if op == "Pow":
        a = alloc(P); return lambda: a.pow(2)
    if op == "Sqrt":
        a = alloc(P); return lambda: a.sqrt()
    if op == "Sum":
        a = alloc(P); dt = DTYPE.get(P["dtype"], torch.float32); return lambda: a.sum(dtype=dt)
    if op == "ZerosLike":
        a = alloc(P); return lambda: torch.zeros_like(a)
    if op == "ConvDType":
        a = alloc(P[0]); dt = DTYPE.get(P[1], torch.float32); return lambda: a.to(dt)
    if op == "Gelu":
        a = alloc(P); return lambda: F.gelu(a)
    if op == "Softmax":
        a = alloc(P[0]); dim = P[1]; dt = DTYPE.get(P[0]["dtype"], torch.float32)
        return lambda: torch.softmax(a, dim, dtype=dt)
    if op == "SoftmaxBackward":
        g = alloc(P[0]); o = alloc(P[1]); dim = P[2]; dt = DTYPE.get(P[0]["dtype"], torch.float32)
        return lambda: torch.ops.aten._softmax_backward_data(g, o, dim, dt)
    if op in ("AddCMul_", "AddCDiv_"):
        a, b, c = alloc(P[0]), alloc(P[1]), alloc(P[2])
        return (lambda: a.addcmul_(b, c)) if op == "AddCMul_" else (lambda: a.addcdiv_(b, c))
    if op == "ForeachMulScalar_":
        ts = [alloc(ti) for ti in P]; return lambda: torch._foreach_mul_(ts, 2)
    if op == "ForeachMul_":
        l1 = [alloc(ti) for ti in P[0]]; l2 = [alloc(ti) for ti in P[1]]
        return lambda: torch._foreach_mul_(l1, l2)
    if op in ("ForeachAddCMul_", "ForeachAddCDiv_"):
        l1 = [alloc(ti) for ti in P[0]]; l2 = [alloc(ti) for ti in P[1]]; l3 = [alloc(ti) for ti in P[2]]
        fn = torch._foreach_addcmul_ if op == "ForeachAddCMul_" else torch._foreach_addcdiv_
        return lambda: fn(l1, l2, l3, 2)
    if op in ("MaskedFill", "MaskedFill_"):
        a = alloc(P[0]); m = alloc(P[1]).to(torch.bool)
        return (lambda: a.masked_fill(m, -10000.0)) if op == "MaskedFill" else (lambda: a.masked_fill_(m, -10000.0))
    if op in ("Dropout", "NativeDropout"):
        a = alloc(P[0]); p = P[1] / 1_000_000.0; train = P[2]
        return (lambda: F.dropout(a, p, train)) if op == "Dropout" else (lambda: torch.native_dropout(a, p, train))
    if op == "FusedDropout":
        a = alloc(P[0]); p = P[1] / 1_000_000.0
        return lambda: torch.ops.aten._fused_dropout(a, p)
    if op == "GeluBackward":
        g = alloc(P[0]); a = alloc(P[1])
        return lambda: torch.ops.aten.gelu_backward(g, a, approximate="none")
    if op == "Conv2d":
        i = alloc(P["input"]); w = alloc(P["weight"])
        b = alloc(P["bias"]) if P.get("bias") else None
        return lambda: F.conv2d(i, w, b, P["stride"], P["padding"], P["dilation"], P["groups"])
    if op == "Conv2dBackward":
        # torch_estimate.rs times aten::_slow_conv2d_backward; use the functional
        # (output_mask) overload rather than the out= variant it calls -- same
        # kernels, and it doesn't need pre-allocated grad buffers.
        g = alloc(P["grad_output"]); i = alloc(P["input"]); w = alloc(P["weight"])
        return lambda: torch.ops.aten._slow_conv2d_backward(
            g, i, w, P["kernel"], P["stride"], P["padding"], [True, True, True])
    if op == "NativeDropoutBackward":
        g = alloc(P[0]); m = alloc(P[1]); scale = P[2] / 1_000_000.0
        return lambda: torch.ops.aten.native_dropout_backward(g, m, scale)
    if op == "Where":
        c = alloc(P[0]).to(torch.bool); a = alloc(P[1]); b = alloc(P[2])
        return lambda: torch.where(c, a, b)
    if op == "WhereScalar":
        c = alloc(P[0]).to(torch.bool); a = alloc(P[1])
        return lambda: torch.where(c, a, 1)
    if op == "SDPA":
        # `mask` is absent from keys recorded before the materialized-mask fix;
        # missing == None, matching serde's Option decoding of an old DB row.
        q, k, v = alloc(P["q"]), alloc(P["k"]), alloc(P["v"])
        mask = alloc(P["mask"]) if P.get("mask") else None
        return lambda: F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=P["causal"], enable_gqa=P["gqa"])
    if op == "SDPAEfficientBackward":
        # Mirrors torch_estimate.rs: build the forward graph via autograd, then
        # time only the backward (torch.autograd.grad == Tensor::run_backward
        # with explicit inputs: returns grads, keeps the graph, no create_graph).
        q = alloc(P["q"], requires_grad=True); k = alloc(P["k"], requires_grad=True); v = alloc(P["v"], requires_grad=True)
        bias = alloc(P["bias"]) if P.get("bias") else None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=P["causal"]).sum(dtype=torch.float32)
        return lambda: torch.autograd.grad(out, [q, k, v], retain_graph=True)
    if op == "SDPABackward":
        # Approximate the internal flash-attn backward via autograd: build the
        # forward graph once, time the backward.
        q = alloc(P["q"], requires_grad=True); k = alloc(P["k"], requires_grad=True); v = alloc(P["v"], requires_grad=True)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=P["causal"])
        g = alloc(P["grad"])
        return lambda: out.backward(g, retain_graph=True)
    return None  # unknown op



# ---------------------------------------------------------------------------
# Generic ops: TorchCallInfo::Generic {name, args} -- any aten op the simulator has
# no dedicated variant for. Run it through torch.ops.aten.<name>. The function tracer
# omits None arguments, so the traced values are aligned to the op schema: a schema
# argument that the next traced value does not fit is skipped (None / its default).
# ---------------------------------------------------------------------------

class _GT:  # a traced tensor argument, materialised later
    def __init__(self, ti, cpu):
        self.ti, self.cpu = ti, cpu


def _g_unwrap(a):
    (k, v), = a.items()
    if k == "T":
        return _GT(v, False)
    if k == "CpuT":
        return _GT(v, True)
    if k == "TL":
        return [_GT(t, False) for t in v]
    if k == "I":
        return int(v)
    if k == "F":
        # "f" = a float other than 0/1 (keys drop the value: LRs change every step).
        # 2.0 rather than an arbitrary value: for norms it is the ord training uses
        # (grad clipping), and 0.5 sent linalg_vector_norm down a 2-3x slower path.
        return 2.0 if v == "f" else float(v)
    if k == "B":
        return bool(v)
    if k == "S":
        return str(v)
    if k == "L":
        return [_g_unwrap(x) for x in v]
    if k == "Dev":
        return torch.device("cpu" if v.lower().startswith("cpu") else "cuda")
    raise ValueError(f"unknown generic arg {k}")


def _base_type(ty):
    """Schema type as Python prints it ('Tensor', 'List[Tensor]', 'Optional[List[int]]',
    'number' for Scalar, 'int' also for ScalarType/MemoryFormat) -> (inner, optional)."""
    if ty.startswith("Optional[") and ty.endswith("]"):
        return ty[len("Optional["):-1], True
    return ty, False


def _fits(ty, v):
    t, _ = _base_type(ty)
    if isinstance(v, _GT):
        return t == "Tensor"
    if isinstance(v, list):
        if not v:  # an empty list (e.g. fused AdamW's max_exp_avg_sqs) fits any list type
            return t.startswith("List[")
        if all(isinstance(x, _GT) for x in v):
            return t in ("List[Tensor]", "List[Optional[Tensor]]")
        return t.startswith("List[") and "Tensor" not in t
    if isinstance(v, bool):
        return t == "bool"
    if isinstance(v, int):
        return t in ("int", "SymInt", "number", "float", "List[int]", "List[SymInt]")
    if isinstance(v, float):
        return t in ("float", "number")
    if isinstance(v, str):
        return t == "str"
    if isinstance(v, torch.device):
        return t == "Device"
    return False


def _align(schema, vals):
    """Assign traced values to schema arguments in order. -> (args, kwargs) or None."""
    pos, kw, i = [], {}, 0
    for arg in schema.arguments:
        ty = str(arg.type)
        if i < len(vals) and _fits(ty, vals[i]):
            v = vals[i]
            i += 1
        elif arg.has_default_value():
            v = arg.default_value
        elif _base_type(ty)[1]:
            v = None
        else:
            return None  # a required argument we cannot fill: wrong overload
        if arg.kwarg_only:
            kw[arg.name] = v
        else:
            pos.append(v)
    return (pos, kw) if i == len(vals) else None


def _index_bound(vals):
    """Integer tensors (indices, targets) get values in [0, bound): the smallest
    dimension > 1 among ALL tensors and int lists of the call (an index tensor can
    index an integer tensor too, e.g. scatter into int32), so every index is valid
    for whatever it indexes. Dimensions of 1 are skipped so e.g. an embedding lookup
    with ids of shape [1, 1024] still spreads over 1024 rows."""
    dims = []

    def walk(v):
        if isinstance(v, _GT):
            dims.extend(d for d in v.ti["shape"] if d > 1)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, int) and not isinstance(x, bool):
                    if x > 1:
                        dims.append(x)
                else:
                    walk(x)
    for v in vals:
        walk(v)
    return max(1, min(dims + [128]))


def _materialise(v, bound, alias=True):
    if isinstance(v, _GT):
        dt = DTYPE.get(v.ti["dtype"], torch.float32)
        shape = v.ti["shape"]
        if v.cpu:
            return torch.ones(shape, dtype=dt)
        if dt.is_floating_point or dt.is_complex:
            return alloc(v.ti, share=True) if alias else _make(shape, dt, "cuda")
        if dt == torch.bool:
            return torch.randint(0, 2, shape, device="cuda").to(torch.bool)
        return torch.randint(0, bound, shape, dtype=dt, device="cuda")
    if isinstance(v, list):
        return [_materialise(x, bound, alias) for x in v]
    return v


def build_generic(P):
    name = P["name"]
    if not name.startswith("aten::"):
        raise NotImplementedError(name)
    packet = getattr(torch.ops.aten, name.split("::", 1)[1])
    vals = [_g_unwrap(a) for a in P["args"]]
    bound = _index_bound(vals)
    errors = []
    for ov in packet.overloads():
        fn = getattr(packet, ov)
        al = _align(fn._schema, vals)
        if al is None:
            continue
        pos, kw = al
        # Distinct operands first (see ALIAS); shared buffers as the fallback when
        # distinct ones do not fit (or, with PERFDB_ALIAS=1, first).
        for alias in ((True, False) if ALIAS else (False, True)):
            try:
                _ACACHE.clear()
                pos_t = [_materialise(v, bound, alias) for v in pos]
                kw_t = {k: _materialise(v, bound, alias) for k, v in kw.items()}
                fn(*pos_t, **kw_t)  # dry run: proves this overload accepts the arguments
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001 -- retry unshared, then the next overload
                if "device-side assert" in str(e) or type(e).__name__ == "AcceleratorError":
                    raise  # the CUDA context is poisoned: let the worker restart
                errors.append(f"{ov}{'' if alias else '/unshared'}: {type(e).__name__}: {str(e)[:60]}")
                continue
            return lambda: fn(*pos_t, **kw_t)
    raise NotImplementedError(f"{name}: no overload fits ({'; '.join(errors)[:200] or 'none aligned'})")


def time_kernel_compute(thunk, warmup, iters):
    """time_kernel without memcpy/memset activity: device copies inside a generic op
    (copy_, clone, to) are already charged on Phantora's memcpy path."""
    for _ in range(warmup):
        thunk()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            thunk()
        torch.cuda.synchronize()
    us = sum(
        (getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0))
        for e in prof.key_averages()
        if not (e.key.startswith("Memcpy") or e.key.startswith("Memset"))
    )
    return int(us * 1000 / iters)


def generic_worker(inp, outp, warmup, iters):
    """Time Generic keys one by one (`--generic-worker IN OUT`). Each result line is
    flushed, so the parent can restart after a crash (e.g. a device-side assert,
    which poisons the CUDA context) and skip the key that caused it."""
    with open(inp) as f, open(outp, "a") as g:
        for line in f:
            key = json.loads(line)["key"]
            obj = json.loads(key)
            try:
                _ACACHE.clear()
                thunk = build_generic(expand(obj["Generic"]))
                res = {"key": key, "nanos": time_kernel_compute(thunk, warmup, iters)}
            except Exception as e:  # noqa: BLE001
                res = {"key": key, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                if "device-side assert" in str(e) or type(e).__name__ == "AcceleratorError":
                    # The CUDA context is now unusable: record this key, then exit so the
                    # parent restarts a fresh worker for the remaining keys.
                    g.write(json.dumps(res) + "\n")
                    g.flush()
                    os._exit(3)
            g.write(json.dumps(res) + "\n")
            g.flush()
            torch.cuda.empty_cache()


def time_generic_isolated(keys, warmup, iters):
    """Run generic_worker in a subprocess; restart after a crash, skipping the key
    that was running. -> {key: nanos or error string}."""
    import subprocess
    import tempfile
    done, todo = {}, list(keys)
    tmp = tempfile.mkdtemp(prefix="bench_generic_")
    while todo:
        inp, outp = os.path.join(tmp, "in.jsonl"), os.path.join(tmp, "out.jsonl")
        with open(inp, "w") as f:
            for k in todo:
                f.write(json.dumps({"key": k}) + "\n")
        if os.path.exists(outp):
            os.remove(outp)
        rc = subprocess.run([sys.executable, "-I", os.path.abspath(__file__), "--generic-worker",
                             inp, outp, str(warmup), str(iters)]).returncode
        got = [json.loads(l) for l in open(outp)] if os.path.exists(outp) else []
        for r in got:
            done[r["key"]] = r.get("nanos", "ERR " + r.get("error", ""))
        todo = todo[len(got):]
        if todo and rc == 3:  # a key poisoned the CUDA context; it is recorded, restart
            continue
        if todo and rc != 0:  # the next key crashed the worker
            done[todo[0]] = f"ERR worker crashed (exit {rc})"
            todo = todo[1:]
        elif todo:
            break
    return done


def time_kernel(thunk, warmup, iters):
    """Kernel-only device time, mirroring Phantora's CUPTI sum: profile `iters`
    runs and sum each kernel's self device time (excludes launch/CPU overhead)."""
    for _ in range(warmup):
        thunk()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            thunk()
        torch.cuda.synchronize()
    us = sum(
        getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        for e in prof.key_averages()
    )
    return int(us * 1000 / iters)  # us total over iters -> ns per iter


def time_wall(thunk, warmup, iters):
    """Wall GPU time via CUDA events (includes launch overhead); --wall option."""
    for _ in range(warmup):
        thunk()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(iters):
        start.record()
        thunk()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e6)  # ms -> ns
    times.sort()
    return int(times[len(times) // 2])  # median ns


def round_multiple(x, m):
    return (x + m - 1) // m * m


_FLASH = None


def flash_mod():
    """The same extension module Phantora profiles against (from flash-attn).
    Imported lazily: a DB with no flash_attn table needs only stock PyTorch."""
    global _FLASH
    if _FLASH is None:
        import flash_attn_2_cuda
        _FLASH = flash_attn_2_cuda
    return _FLASH


FLASH_COLS = ["is_fwd", "is_bf16", "batch", "seqlen_q", "seqlen_k", "num_heads",
              "num_heads_k", "head_size", "win_left", "win_right", "is_causal"]


def flash_thunk(row):
    """Mirror CudaEstimator::flash_attn (cuda_estimate.rs): same operand shapes
    (head_size rounded up to a multiple of 8 where it rounds), same softmax scale,
    same fwd/bwd argument lists."""
    fa = flash_mod()
    r = dict(zip(FLASH_COLS, row))
    b, sq, sk = int(r["batch"]), int(r["seqlen_q"]), int(r["seqlen_k"])
    nh, nhk, hs = int(r["num_heads"]), int(r["num_heads_k"]), int(r["head_size"])
    wl, wr = int(r["win_left"]), int(r["win_right"])
    causal = r["is_causal"] == "true"
    dt = torch.bfloat16 if r["is_bf16"] == "true" else torch.float16
    hs8 = round_multiple(hs, 8)
    scale = float(hs) ** -0.5
    rnd = lambda *shape, dtype=dt: torch.randn(shape, dtype=dtype, device="cuda")

    if r["is_fwd"] == "true":
        q, k, v = rnd(b, sq, nh, hs8), rnd(b, sk, nhk, hs8), rnd(b, sk, nhk, hs8)
        return lambda: fa.fwd(q, k, v, None, None, 0.0, scale, causal, wl, wr, 0.0, False, None)
    dout = rnd(b, sq, nh, hs8)
    q, k, v = rnd(b, sq, nh, hs), rnd(b, sk, nhk, hs), rnd(b, sk, nhk, hs)
    o = rnd(b, sq, nh, hs)
    lse = rnd(b, nh, sq, dtype=torch.float32)
    return lambda: fa.bwd(dout, q, k, v, o, lse, None, None, None, None, 0.0, scale,
                          causal, wl, wr, 0.0, False, None, None)


def memcpy_thunk(kind, size):
    n = max(size, 1)
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    def buf(dev, pin=False):
        t = torch.empty(n, dtype=torch.uint8, device=dev)
        return t.pin_memory() if pin and dev == cpu else t
    if kind == "DeviceToDevice":
        s, d = buf(cuda), buf(cuda)
    elif kind == "HostToDevice":
        s, d = buf(cpu), buf(cuda)
    elif kind == "PinnedHostToDevice":
        s, d = buf(cpu, pin=True), buf(cuda)
    elif kind == "DeviceToHost":
        s, d = buf(cuda), buf(cpu)
    elif kind == "DeviceToPinnedHost":
        s, d = buf(cuda), buf(cpu, pin=True)
    else:  # HostToHost
        s, d = buf(cpu), buf(cpu)
    return lambda: d.copy_(s)


def read_manifest_gpu(db_dir):
    """GPU name recorded in a DB's manifest.md, or None. Mirrors the
    `- **GPU:** ` prefix parse in perf_db.rs's PerfDb::load."""
    path = os.path.join(db_dir, "manifest.md")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            if line.startswith("- **GPU:** "):
                return line[len("- **GPU:** "):].strip() or None
    return None


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--generic-worker":
        _, _, inp, outp, warmup, iters = sys.argv
        return generic_worker(inp, outp, int(warmup), int(iters))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference DB dir to take keys from (e.g. tests/perfdb/l40s)")
    ap.add_argument("--out", default=None, help="output DB dir (default: tests/perfdb/<gpu-name>)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--wall", action="store_true",
                    help="use CUDA-event wall time instead of kernel-only (default) timing")
    ap.add_argument("--merge", action="store_true",
                    help="merge profiled keys into an existing --out DB (e.g. to fill a "
                         "<db>.missing manifest into <db>) instead of overwriting it")
    ap.add_argument("--force", action="store_true",
                    help="allow --merge into a DB recorded on a different GPU (produces a DB "
                         "whose timings come from two GPUs; almost never what you want)")
    args = ap.parse_args()
    timer = time_wall if args.wall else time_kernel

    if not torch.cuda.is_available():
        sys.exit("error: a CUDA GPU is required to record timings")
    gpu = torch.cuda.get_device_name(0)
    out = args.out or os.path.join(os.path.dirname(args.ref.rstrip("/")), gpu.replace(" ", "_"))
    os.makedirs(out, exist_ok=True)

    # A DB's timings are only meaningful for one GPU. Merging fills the *missing*
    # keys from this GPU while the keys already present keep the original GPU's
    # times, and the manifest below is rewritten with this GPU's name -- so a
    # cross-GPU merge silently yields one DB blending two GPUs. Refuse it: the
    # documented recovery loop ("on any GPU machine, run bench.py ... --merge")
    # otherwise corrupts the target DB.
    if args.merge:
        target_gpu = read_manifest_gpu(out)
        if target_gpu and target_gpu != gpu and not args.force:
            sys.exit(
                f"error: {out} was recorded on '{target_gpu}' but this machine is '{gpu}'.\n"
                f"Merging would mix timings from two GPUs into one database. Either run this on "
                f"an {target_gpu}, or record a separate DB for this GPU:\n"
                f"    python3 {sys.argv[0]} --ref {args.ref} --out tests/perfdb/{gpu.replace(' ', '_').lower()}\n"
                f"(pass --force to merge anyway.)"
            )
    print(f"GPU: {gpu}\nref: {args.ref}\nout: {out}")

    def load_existing(path, nkey):
        """Load an existing CSV into {tuple(first nkey cols): full row}."""
        rows = {}
        if os.path.exists(path):
            with open(path) as f:
                rd = csv.reader(f)
                next(rd, None)
                for row in rd:
                    if row:
                        rows[tuple(row[:nkey])] = row
        return rows

    # compute.csv (columns: key, nanos; key = compacted TorchCallInfo JSON, op is
    # its single outer key). With --merge, start from the existing out DB and
    # overlay the profiled keys; otherwise write a fresh table. The original
    # (compact) key string is carried through verbatim, so no Python compaction
    # is needed -- only expand() to reconstruct tensors for profiling.
    compute_out = os.path.join(out, "compute.csv")
    rows = load_existing(compute_out, 1) if args.merge else {}
    n_ok = n_skip = n_drop = 0
    with open(os.path.join(args.ref, "compute.csv")) as f:
        ref_rows = [row for row in csv.reader(f)][1:]
    # Generic ops run in a crash-isolated subprocess (see time_generic_isolated).
    gkeys = [row[0] for row in ref_rows if row and json.loads(row[0]).keys() == {"Generic"}]
    generic = time_generic_isolated(gkeys, args.warmup, args.iters) if gkeys and not args.wall else {}
    if gkeys:
        g_err = sum(isinstance(v, str) for v in generic.values())
        print(f"generic ops: {len(gkeys) - g_err} timed, {g_err} failed")
    if True:
        for row in ref_rows:
            if not row:
                continue
            key, ref_nanos = row[0], row[1]
            obj = json.loads(key)
            op = next(iter(obj))
            try:
                if op == "Generic" and key in generic:
                    if isinstance(generic[key], str):
                        raise RuntimeError(generic[key])
                    nanos = generic[key]
                    n_ok += 1
                    rows[(key,)] = [key, nanos]
                    continue
                global ALIAS
                try:
                    thunk = build(op, expand(obj[op]))
                    if thunk is None:
                        raise NotImplementedError(op)
                    nanos = timer(thunk, args.warmup, args.iters)
                except torch.cuda.OutOfMemoryError:
                    if ALIAS:
                        raise
                    # distinct operands (and outputs) do not fit: time with shared buffers
                    thunk = None
                    torch.cuda.empty_cache()
                    ALIAS = True
                    try:
                        thunk = build(op, expand(obj[op]))
                        nanos = timer(thunk, args.warmup, args.iters)
                    finally:
                        ALIAS = False
                        thunk = None
                        _ACACHE.clear()
                        torch.cuda.empty_cache()
                    print(f"  note {op}: too large for distinct operands; timed with shared buffers",
                          file=sys.stderr)
                n_ok += 1
            except Exception as e:  # keep the DB complete; carry the reference time
                nanos = int(ref_nanos)
                n_skip += 1
                if nanos == 0:
                    # The ref time is a zero placeholder (a `<db>.missing` row), so
                    # there is nothing to carry over. Writing it would bake a 0 ns
                    # entry into the DB that replay then treats as a hit -- charging
                    # the op no time and never reporting it missing again. Drop the
                    # row instead: it stays missing and gets rediscovered.
                    n_drop += 1
                    print(f"  WARN {op}: {type(e).__name__}: {str(e)[:80]} -> DROPPED "
                          f"(unprofilable, no reference time)", file=sys.stderr)
                    continue
                print(f"  WARN {op}: {type(e).__name__}: {str(e)[:80]} -> kept reference time", file=sys.stderr)
            rows[(key,)] = [key, nanos]
    with open(compute_out, "w", newline="") as g:
        w = csv.writer(g); w.writerow(["key", "nanos"])
        for row in sorted(rows.values()):
            w.writerow(row)
    print(f"compute.csv: {n_ok} profiled, {n_skip - n_drop} carried-over, "
          f"{n_drop} dropped, {len(rows)} total")
    if n_drop:
        print(f"  {n_drop} op(s) could not be profiled and had no reference time; they are "
              f"NOT in the DB and will be reported missing again on the next replay.",
              file=sys.stderr)

    # memcpy.csv (key columns: kind, size_bytes)
    memcpy_out = os.path.join(out, "memcpy.csv")
    mrows = load_existing(memcpy_out, 2) if args.merge else {}
    m_ok = m_drop = 0
    with open(os.path.join(args.ref, "memcpy.csv")) as f:
        r = csv.reader(f); next(r)
        for kind, size, ref_nanos in r:
            try:
                nanos = timer(memcpy_thunk(kind, int(size)), args.warmup, args.iters)
                m_ok += 1
            except Exception as e:
                nanos = int(ref_nanos)
                if nanos == 0:  # a .missing placeholder: see the compute path
                    m_drop += 1
                    print(f"  WARN memcpy {kind}/{size}: {e} -> DROPPED (no reference time)",
                          file=sys.stderr)
                    continue
                print(f"  WARN memcpy {kind}/{size}: {e} -> kept reference time", file=sys.stderr)
            mrows[(kind, size)] = [kind, size, nanos]
    with open(memcpy_out, "w", newline="") as g:
        w = csv.writer(g); w.writerow(["kind", "size_bytes", "nanos"])
        for row in sorted(mrows.values(), key=lambda x: (x[0], int(x[1]))):
            w.writerow(row)
    print(f"memcpy.csv: {m_ok} profiled, {m_drop} dropped, {len(mrows)} total")

    # flash_attn.csv (key columns: the 11 FlashAttnKey fields). Profiled like any
    # other table -- it used to be byte-copied from the ref, which meant a
    # `<db>.missing` flash table (all 0 ns) was copied in verbatim and then read
    # back by replay as real hits, silently charging attention no time forever.
    flash_ref = os.path.join(args.ref, "flash_attn.csv")
    flash_out = os.path.join(out, "flash_attn.csv")
    if os.path.exists(flash_ref):
        frows = load_existing(flash_out, len(FLASH_COLS)) if args.merge else {}
        f_ok = f_carry = f_drop = 0
        with open(flash_ref) as f:
            r = csv.reader(f); next(r)
            for row in r:
                if not row:
                    continue
                key, ref_nanos = tuple(row[:len(FLASH_COLS)]), int(row[len(FLASH_COLS)])
                try:
                    nanos = timer(flash_thunk(key), args.warmup, args.iters)
                    f_ok += 1
                except Exception as e:
                    if ref_nanos == 0:  # a .missing placeholder: see the compute path
                        f_drop += 1
                        print(f"  WARN flash_attn {key}: {type(e).__name__}: {str(e)[:60]} -> "
                              f"DROPPED (unprofilable, no reference time)", file=sys.stderr)
                        continue
                    nanos = ref_nanos
                    f_carry += 1
                    print(f"  WARN flash_attn {key}: {type(e).__name__}: {str(e)[:60]} -> "
                          f"kept reference time", file=sys.stderr)
                frows[key] = list(key) + [nanos]
        if frows:
            with open(flash_out, "w", newline="") as g:
                w = csv.writer(g); w.writerow(FLASH_COLS + ["nanos"])
                for row in sorted(frows.values()):
                    w.writerow(row)
        print(f"flash_attn.csv: {f_ok} profiled, {f_carry} carried-over, "
              f"{f_drop} dropped, {len(frows)} total")
        if f_drop or f_carry:
            print("  flash-attn rows need the `flash_attn_2_cuda` module (pip install flash-attn) "
                  "-- the same extension Phantora profiles against.", file=sys.stderr)

    # sequence.csv: hash-keyed groups of calls, not (op, shape) -- they cannot be
    # re-profiled from the DB alone. Both perf-db modes force single-op timing, so
    # this table is empty in practice; if a ref ever carries one, skip it rather
    # than copy another GPU's timings into this GPU's database.
    seq_ref = os.path.join(args.ref, "sequence.csv")
    if os.path.exists(seq_ref):
        with open(seq_ref) as f:
            n_seq = max(0, sum(1 for _ in f) - 1)
        if n_seq:
            print(f"  WARN sequence.csv: {n_seq} row(s) in {args.ref} cannot be re-profiled "
                  f"(hash-keyed sequences); not copied into {out}.", file=sys.stderr)

    mpath = os.path.join(out, "manifest.md")
    if args.merge and read_manifest_gpu(out) == gpu:
        # keep the DB's provenance; note the merge (the ref path is often a temp dir)
        with open(mpath) as f:
            text = f.read()
        note = "- Later shapes were added with `bench.py --merge` on the same GPU type.\n"
        if note not in text:
            with open(mpath, "a") as f:
                f.write(note)
    else:
        with open(mpath, "w") as f:
            f.write(f"# Phantora performance database\n\n- **GPU:** {gpu}\n- **Schema version:** 1\n"
                    f"- Recorded by tests/perfdb/bench.py (stock PyTorch, no Phantora build) "
                    f"from keys in `{args.ref}`.\n")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
