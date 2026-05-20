Good data. Now I have everything needed for a thorough, hardware-specific answer.

Here's a complete breakdown:

---

## Hardware Reality Check

Your Ryzen AI MAX+ 395 is a genuinely capable training machine, but it has some important quirks you need to know about upfront.

**What you have:**
- The Radeon 8060S has 40 RDNA 3.5 CUs. At a max clock of 2.9 GHz, peak theoretical BF16 is ~59.4 TFLOPS — but this requires hipBLASLt (WMMA). Without it, you're looking at under 9% utilization (~5 TFLOPS). With hipBLASLt, real-world throughput lands around 36.9 TFLOPS (~60% efficiency), comparable to MI300X efficiency numbers.
- The 128 GB unified memory means you effectively have up to 96 GB of "VRAM" — a massive advantage over discrete GPU setups for memory-hungry fine-tuning.
- Memory bandwidth is ~212 GB/s on a 256-bit bus.

**Critical BF16 note:** float32 performance is only ~3.1 TFLOPS vs float16's 35 TFLOPS. Your training script is already using `bfloat16` in `autocast`, which is exactly right — never let it fall back to float32.

---

## Training Time Estimate

Your `run_train.py` trains two models jointly: the `ViNorthHead` (small) + XPhoneBERT (110M params, 768-dim). The bottleneck is XPhoneBERT forward+backward.

Rough step time estimate on the 8060S with ROCm + hipBLASLt, batch size 16:

| Phase | Estimate per step |
|---|---|
| XPhoneBERT forward (BF16, bs=16) | ~35–55 ms |
| Head forward + MSE loss | ~2 ms |
| Backward pass (~2× forward) | ~70–110 ms |
| Optimizer step + scaler | ~10 ms |
| **Total per step** | **~120–180 ms** |

For 100,000 steps at ~150 ms/step: **roughly 4–4.5 hours** of pure compute time. Add DataLoader I/O and it realistically lands around **5–7 hours** for the full run, assuming the dataset prep is done first and audio fits on a fast NVMe.

The dataset prep (`prepare_dataset.py`) streaming PhoAudioBook is the harder variable — it depends entirely on your internet speed and how large the retained Northern-dialect subset turns out to be.

---

## Optimizations for this Specific Device

### 1. Install the right PyTorch wheels — most important step

Stable ROCm does **not** ship gfx1151 kernels. Consumer Strix Halo users must install from the nightly index: `pip install torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/` — and importantly, `PYTORCH_HIP_ALLOC_CONF=backend:malloc` must be **unset**, as it crashes PyTorch on this architecture.

```bash
# Unset this if it's in your shell profile
unset PYTORCH_HIP_ALLOC_CONF

pip install torch torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
```

### 2. Pin UMA VRAM allocation in BIOS

The maximum VRAM allocatable in BIOS is 96 GB, but using the TTM kernel parameter you can increase this to 120 GB per node. For training, set BIOS UMA to at least 64 GB, and add to `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.sg_display=0 amdgpu.noretry=0"
```
Then `sudo update-grub && reboot`.

### 3. Bump batch size — memory is your superpower

With 96 GB of effective VRAM, you can push `--batch-size` to 64 or even 128. Larger batches dramatically improve GPU utilization on the 8060S (which is more memory-bandwidth-bound than compute-bound on small batches). Update `run_train.py`:

```python
# In parse_args():
p.add_argument("--batch-size", type=int, default=64)  # was 16
```

Also increase warmup proportionally: `--warmup-steps 2000` for bs=64.

### 4. Enable `torch.compile` for XPhoneBERT

The gfx1151 architecture shows an undocumented 19× attention speedup with AOTriton (the ROCm Flash Attention backend). Enabling `torch.compile` lets PyTorch trigger this path:

```python
# In main(), after model creation:
xphonebert = torch.compile(xphonebert, backend="inductor", mode="reduce-overhead")
```

First epoch will be slower (compilation), subsequent steps significantly faster.

### 5. Increase DataLoader workers and prefetch

The Ryzen AI MAX+ 395 has a 16-core CPU complex, so you can afford more workers than the current `min(4, os.cpu_count())`:

```python
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,          # was 4
    prefetch_factor=4,      # queue 4 batches per worker
    pin_memory=False,       # unified memory — pin_memory is a no-op/slight harm
    persistent_workers=True,
)
```

Note: `pin_memory=True` is for discrete GPU systems where host→device copies benefit from page-locked memory. On a unified memory APU, it does nothing useful and wastes RAM — set it to `False`.

### 6. Use `HSA_ENABLE_SDMA=0` for large transfers

Unified memory APUs can hit memcpy bottlenecks on large tensor transfers. Add to your training shell script:

```bash
export HSA_ENABLE_SDMA=0
export HSA_OVERRIDE_GFX_VERSION=11.5.1   # only if using older ROCm without native gfx1151
python run_train.py ...
```

---

## Revised `run_train.py` Launcher Script

```bash
#!/bin/bash
# train_vi_north.sh — optimized for Ryzen AI MAX+ 395

unset PYTORCH_HIP_ALLOC_CONF        # crashes PyTorch on gfx1151
export HSA_ENABLE_SDMA=0
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

python run_train.py \
  --manifest   data/train_manifest.csv \
  --batch-size 64 \
  --max-steps  100000 \
  --warmup-steps 2000 \
  --lr         2e-5 \
  --save-every 5000 \
  --stage      1 \
  "$@"
```

---

## Summary

| | Original | Optimized |
|---|---|---|
| PyTorch wheels | Stable ROCm (no gfx1151 kernels) | Nightly gfx1151 index |
| Batch size | 16 | 64 |
| `pin_memory` | True (wasted) | False |
| `torch.compile` | Off | Inductor + reduce-overhead |
| DataLoader workers | 4 | 8 + prefetch_factor=4 |
| BF16 performance | ~5 TFLOPS (no hipBLASLt) | ~35 TFLOPS |
| **Estimated training time** | **25–30 hrs** | **5–7 hrs** |

The single biggest lever is installing the correct gfx1151 PyTorch wheels — without them you're running at under 10% of peak BF16 throughput.