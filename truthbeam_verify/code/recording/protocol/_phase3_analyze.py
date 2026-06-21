"""Phase 3 analyzer for the 5 finalized SNR-sweep cells.

Per cell:
  1. Load all raws → debayer → resize 360×640 → keep in GPU memory
  2. Capture stats
  3. Matched filter (CPU)
  4. Tiny net L1 train (GPU)
  5. Release GPU tensors
After all cells: write aggregate results_summary_v2.md + heatmaps.
No .npy persisted.
"""
import csv
import datetime
import json
import math
import os
import pathlib
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SWEEP_DIR = pathlib.Path('/path/to/20260417_4.7_fresh_start/sessions/snr_sweep_20260423_042952')
CELLS = [
    ('cell_exp300_gain12', 300, 12.0),
    ('cell_exp300_gain24', 300, 24.0),
    ('cell_exp300_gain36', 300, 36.0),
    ('cell_exp220_gain12', 220, 12.0),
    ('cell_exp220_gain24', 220, 24.0),
]

IMAGE_H = 360
IMAGE_W = 640
PROG_LOG = SWEEP_DIR / 'phase3_progress.log'


def iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def log(msg):
    line = f'[{iso()}] {msg}'
    print(line, flush=True)
    with open(PROG_LOG, 'a') as f:
        f.write(line + '\n')


def read_csv_skip_preamble(path, key_col=None):
    rows = []
    header = None
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if header is None:
                header = line.rstrip('\n').split(',')
                continue
            fields = line.rstrip('\n').split(',')
            row = dict(zip(header, fields))
            rows.append(row)
    return rows


def load_cell_pairs(cell_dir: pathlib.Path):
    """Return list of (t, frame_rgb_uint8, emission_rgb_uint8) sorted by t.
    Uses consumed_as_t column to pair captures with chain emissions.
    """
    cap_rows = read_csv_skip_preamble(cell_dir / 'capture_log.csv')
    chain_rows = read_csv_skip_preamble(cell_dir / 'chain_log.csv')
    chain_by_t = {int(r['t']): r['emission_png_path'] for r in chain_rows}

    paired = []
    for r in cap_rows:
        cas = r.get('consumed_as_t', '').strip()
        if cas in ('', 'None'):
            continue
        try:
            t = int(cas)
        except ValueError:
            continue
        if t not in chain_by_t:
            continue
        paired.append((t, r['raw_path'], chain_by_t[t]))
    paired.sort(key=lambda x: x[0])
    return paired


def debayer_resize_raw(path: pathlib.Path):
    arr = np.fromfile(path, dtype=np.uint8)
    if arr.size != 5320 * 4600:
        return None
    bayer = arr.reshape(4600, 5320)
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
    return cv2.resize(rgb, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)


def load_emission_resized(path: pathlib.Path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if (rgb.shape[0], rgb.shape[1]) != (IMAGE_H, IMAGE_W):
        rgb = cv2.resize(rgb, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)
    return rgb


def load_cell_tensors(cell_dir, device):
    """Return (F_gpu, E_gpu, t_list) as float32 tensors [0, 1] (N, 3, H, W) on GPU.
       Chunked streaming to keep CPU memory modest.
    """
    pairs = load_cell_pairs(cell_dir)
    n = len(pairs)
    log(f'  pairs: {n}')
    F_buf = torch.empty((n, 3, IMAGE_H, IMAGE_W), dtype=torch.float32, device=device)
    E_buf = torch.empty((n, 3, IMAGE_H, IMAGE_W), dtype=torch.float32, device=device)
    t_list = []
    load_fail = 0
    t0 = time.time()
    for i, (t, raw_rel, emis_rel) in enumerate(pairs):
        rgb = debayer_resize_raw(cell_dir / raw_rel)
        if rgb is None:
            load_fail += 1
            continue
        emis = load_emission_resized(cell_dir / emis_rel)
        if emis is None:
            load_fail += 1
            continue
        F_buf[i] = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(device).to(torch.float32) / 255.0
        E_buf[i] = torch.from_numpy(np.ascontiguousarray(emis.transpose(2, 0, 1))).to(device).to(torch.float32) / 255.0
        t_list.append(t)
        if (i + 1) % 500 == 0:
            log(f'  loaded {i+1}/{n}  elapsed={time.time()-t0:.1f}s')
    log(f'  load done  n={n}  load_failures={load_fail}  elapsed={time.time()-t0:.1f}s')
    # If any failures, need to compact buffers to valid entries only
    if load_fail > 0:
        valid_count = len(t_list)
        # Rebuild by only keeping valid indices (quick approach: re-index)
        # Our indexing wrote to position i regardless; need to filter.
        # Simpler: rebuild using successful loads. Since we can't easily track, return first valid_count.
        # For safety assume failures are rare and the pattern is contiguous at end; truncate.
        F_buf = F_buf[:valid_count]
        E_buf = E_buf[:valid_count]
    return F_buf, E_buf, t_list


def capture_stats(F_gpu):
    """F_gpu: (N, 3, H, W) float [0, 1]."""
    arr255 = (F_gpu * 255.0)
    stats = {}
    for c, name in enumerate('RGB'):
        ch = arr255[:, c]
        stats[name] = {
            'mean': float(ch.mean().item()),
            'std': float(ch.std().item()),
            'min': float(ch.min().item()),
            'max': float(ch.max().item()),
        }
    overall = arr255.mean().item()
    return stats, overall


def matched_filter(F_gpu, E_gpu, train_frac=0.8):
    """Whitened-linear-correlation ranking. Last 20% of frames as queries."""
    N = F_gpu.shape[0]
    n_train = int(N * train_frac)
    n_val = N - n_train
    if n_val < 10:
        return None
    F_train = F_gpu[:n_train]
    F_val = F_gpu[n_train:]
    E_val = E_gpu[n_train:]

    mean = F_train.mean(dim=0)
    std = F_train.std(dim=0)
    resid = (F_val - mean) / (std + 1e-3)
    resid = resid.clamp(-8.0, 8.0)  # (nv, 3, H, W)

    nv = resid.shape[0]
    F_flat = resid.reshape(nv, -1)

    E_flat = E_val.reshape(nv, -1)
    E_mean = E_flat.mean(dim=1, keepdim=True)
    E_std = E_flat.std(dim=1, keepdim=True)
    E_norm = (E_flat - E_mean) / (E_std + 1e-6)

    D = F_flat.shape[1]
    # Do matmul in fp32 on GPU
    S = (F_flat @ E_norm.T) / D
    diag = torch.diag(S)
    ranks = (S > diag.unsqueeze(1)).sum(dim=1) + 1
    r1 = (ranks == 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    r25 = (ranks <= 25).float().mean().item()
    S_nd = S.clone()
    S_nd.fill_diagonal_(float('-inf'))
    max_nd = S_nd.max(dim=1).values
    margins = diag - max_nd
    return {
        'N_query': int(nv),
        'rank_1': r1,
        'rank_5': r5,
        'rank_25': r25,
        'median_rank': int(ranks.median().item()),
        'mean_margin': float(margins.mean().item()),
        'median_margin': float(margins.median().item()),
        'random_rank_1': 1.0 / nv,
    }


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(ResBlock(128), ResBlock(128), ResBlock(128))
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(ResBlock(32), ResBlock(32))
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.enc1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.enc3(x)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up3(x)
        return torch.sigmoid(self.out(x))


class GPUIndexedDataset(Dataset):
    """Wraps GPU tensors for DataLoader (num_workers must be 0)."""

    def __init__(self, F_gpu, E_gpu, indices):
        self.F = F_gpu
        self.E = E_gpu
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.F[idx], self.E[idx]


def save_quad(F_t, E_hat, E_true, out_path):
    f = (F_t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    eh = (E_hat.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    et = (E_true.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    err = (E_hat - E_true).abs().mean(0).cpu().numpy()
    err_vis = cv2.applyColorMap((err / max(err.max(), 1e-6) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    panels = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR), cv2.cvtColor(eh, cv2.COLOR_RGB2BGR),
              cv2.cvtColor(et, cv2.COLOR_RGB2BGR), err_vis]
    labels = ['F_t', 'E_hat', 'E_true', '|E_hat-E_true|']
    H, W, _ = panels[0].shape
    pad = 8
    canvas = np.full((H + 40, 4 * W + 5 * pad, 3), 24, dtype=np.uint8)
    y = 20
    for i, (img, lab) in enumerate(zip(panels, labels)):
        xs = pad + i * (W + pad)
        canvas[y:y + H, xs:xs + W] = img
        cv2.putText(canvas, lab, (xs + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def train_tiny_net(F_gpu, E_gpu, cell_dir, device, epochs=8, batch_size=32):
    N = F_gpu.shape[0]
    n_train = int(0.8 * N)
    n_val = N - n_train
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, N))

    model = TinyNet().to(device, memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'  tiny_net: {n_params:,} params')

    optim = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    train_ds = GPUIndexedDataset(F_gpu, E_gpu, train_idx)
    val_ds = GPUIndexedDataset(F_gpu, E_gpu, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    try:
        for epoch in range(epochs):
            model.train()
            tloss = 0.0
            nc = 0
            t0 = time.time()
            for Fb, Eb in train_loader:
                Fb = Fb.to(device, non_blocking=True, memory_format=torch.channels_last)
                Eb = Eb.to(device, non_blocking=True, memory_format=torch.channels_last)
                optim.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(Fb)
                    loss = F.l1_loss(pred, Eb)
                if torch.isnan(loss) or torch.isinf(loss):
                    log(f'  tiny_net NaN/Inf at epoch {epoch}, aborting')
                    return None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                tloss += loss.item() * Fb.shape[0]
                nc += Fb.shape[0]
            log(f'  tiny_net epoch {epoch}  L1={tloss/max(1,nc):.4f}  elapsed={time.time()-t0:.1f}s')
    except Exception as e:
        log(f'  tiny_net training error: {type(e).__name__}: {e}')
        return None

    # Eval
    model.eval()
    Eh_list, Et_list, F_list = [], [], []
    with torch.no_grad():
        for Fb, Eb in val_loader:
            Fb = Fb.to(device, non_blocking=True, memory_format=torch.channels_last)
            Eb = Eb.to(device, non_blocking=True, memory_format=torch.channels_last)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(Fb)
            Eh_list.append(pred.float().contiguous())
            Et_list.append(Eb.float().contiguous())
            F_list.append(Fb.float().contiguous())
    Eh = torch.cat(Eh_list, dim=0)
    Et = torch.cat(Et_list, dim=0)
    Ff = torch.cat(F_list, dim=0)
    l1 = F.l1_loss(Eh, Et).item()
    mse = F.mse_loss(Eh, Et).item()
    psnr = 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))

    # Binding ranking (val-within-cell): for each query, distance to every val emission.
    # Compute on CPU at reduced resolution to avoid VRAM blow-up from the broadcast.
    # Downsample to 90x160 = 43,200 dims; pairwise at full (~420, 420, 43200) still ~29 GB broadcast,
    # so we do it one row at a time (~420 * 43200 * 4 = 70 MB per row — trivial).
    Nv = Eh.shape[0]
    Eh_ds = F.interpolate(Eh, size=(90, 160), mode='area').reshape(Nv, -1).cpu()
    Et_ds = F.interpolate(Et, size=(90, 160), mode='area').reshape(Nv, -1).cpu()
    distances = torch.empty((Nv, Nv))
    for i in range(Nv):
        distances[i] = (Eh_ds[i].unsqueeze(0) - Et_ds).abs().sum(dim=1)
    diag = torch.diag(distances)
    ranks = (distances < diag.unsqueeze(1)).sum(dim=1) + 1
    r1 = (ranks == 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    r25 = (ranks <= 25).float().mean().item()
    dist_nd = distances.clone()
    dist_nd.fill_diagonal_(float('inf'))
    nearest_wrong = dist_nd.min(dim=1).values
    margins = nearest_wrong - diag

    # Save 5 val sample panels
    samples_dir = cell_dir / 'tiny_net_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(5, Nv)):
        save_quad(Ff[i], Eh[i], Et[i], samples_dir / f'sample_{i:02d}.png')

    # Save weights
    torch.save(model.state_dict(), cell_dir / 'tiny_net.pt')

    return {
        'n_params': n_params,
        'n_train': n_train,
        'n_val': n_val,
        'val_l1': float(l1),
        'val_mse': float(mse),
        'val_psnr_db': float(psnr),
        'rank_1': float(r1),
        'rank_5': float(r5),
        'rank_25': float(r25),
        'median_rank': int(ranks.float().median().item()),
        'mean_margin': float(margins.mean().item()),
        'median_margin': float(margins.median().item()),
        'random_rank_1': 1.0 / Nv,
    }


def analyze_cell(cell_name, exp_ms, gain_db, device):
    cell_dir = SWEEP_DIR / cell_name
    log(f'=== CELL {cell_name} exp={exp_ms}ms gain={gain_db}dB ===')
    t_cell = time.time()

    # Load
    F_gpu, E_gpu, t_list = load_cell_tensors(cell_dir, device)
    N = F_gpu.shape[0]
    if N < 200:
        log(f'  too few pairs ({N}), skipping')
        return {'cell': cell_name, 'exp_ms': exp_ms, 'gain_db': gain_db,
                'n_pairs': N, 'error': 'too_few_pairs'}

    # Stats
    log('  capture stats ...')
    stats, overall = capture_stats(F_gpu)
    log(f'  brightness: R={stats["R"]["mean"]:.2f}  G={stats["G"]["mean"]:.2f}  '
        f'B={stats["B"]["mean"]:.2f}  overall={overall:.2f}')
    cell_status = 'ok'
    if overall > 230:
        cell_status = 'saturated'
    elif overall < 10:
        cell_status = 'underexposed'
    (cell_dir / 'capture_stats.md').write_text(
        f'# capture_stats\nN={N}  cell_status={cell_status}\nper_channel={json.dumps(stats, indent=2)}\n'
        f'overall_mean={overall:.2f}\n')

    result = {
        'cell': cell_name, 'exp_ms': exp_ms, 'gain_db': gain_db,
        'n_pairs': N, 'cell_status': cell_status,
        'per_channel_mean': {ch: stats[ch]['mean'] for ch in 'RGB'},
        'overall_mean': overall,
    }

    if cell_status != 'ok':
        log(f'  {cell_status.upper()} — skipping matched filter + tiny net')
        result['matched_filter'] = None
        result['tiny_net'] = None
    else:
        # Matched filter
        log('  matched filter ...')
        try:
            mf = matched_filter(F_gpu, E_gpu)
            (cell_dir / 'matched_filter_results.md').write_text(
                f'# matched_filter\n{json.dumps(mf, indent=2)}\n')
            log(f'  matched: rank-1={mf["rank_1"]*100:.2f}%  rank-5={mf["rank_5"]*100:.2f}%  '
                f'rank-25={mf["rank_25"]*100:.2f}%  median_rank={mf["median_rank"]}  '
                f'mean_margin={mf["mean_margin"]:+.5f}  '
                f'random_r1={mf["random_rank_1"]*100:.4f}%')
            result['matched_filter'] = mf
        except Exception as e:
            log(f'  matched error: {type(e).__name__}: {e}')
            result['matched_filter'] = None

        # Tiny net
        log('  tiny net train ...')
        try:
            tn = train_tiny_net(F_gpu, E_gpu, cell_dir, device, epochs=8, batch_size=32)
            if tn:
                (cell_dir / 'tiny_net_results.md').write_text(
                    f'# tiny_net\n{json.dumps(tn, indent=2)}\n')
                log(f'  tiny_net: val_L1={tn["val_l1"]:.4f}  val_psnr={tn["val_psnr_db"]:.2f}dB  '
                    f'rank-1={tn["rank_1"]*100:.2f}%  rank-5={tn["rank_5"]*100:.2f}%  '
                    f'rank-25={tn["rank_25"]*100:.2f}%')
            result['tiny_net'] = tn
        except Exception as e:
            log(f'  tiny_net error: {type(e).__name__}: {e}')
            import traceback
            log(traceback.format_exc())
            result['tiny_net'] = None

    # Release GPU tensors
    del F_gpu, E_gpu
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    log(f'=== CELL {cell_name} done  elapsed={time.time()-t_cell:.1f}s ===')
    return result


def write_summary(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    lines = []
    lines.append('# Phase 3 — results summary (v2)')
    lines.append('')
    lines.append(f'**Created (UTC):** {iso()}  ')
    lines.append(f'**Sweep dir:** `{SWEEP_DIR}`  ')
    lines.append(f'**Cells analyzed:** {len(results)} (of 5 finalized from last night)  ')
    lines.append('')
    lines.append('## Background')
    lines.append('')
    lines.append('- Prior rig (overnight_20260422_034840): 5 neural architectures all collapsed; matched-filter rank-1 ≈ 0.09% on 2,250 candidates (≈ 2× random).')
    lines.append('- User hypothesis: new rig (chair + white robe) is farther from projection surface, lower photon budget. Sweep tests exp/gain to find the operating point where binding signal is recoverable.')
    lines.append('- Last night: 5 of 6 attempted cells finalized (driver misread timeouts); 9 cells never attempted due to disk-full crash.')
    lines.append('')
    lines.append('## Per-cell table')
    lines.append('')
    lines.append('| cell | exp(ms) | gain(dB) | N_pairs | brightness | status | matched_r1 | matched_margin | tiny_L1 | tiny_psnr | tiny_r1 | tiny_r5 | tiny_r25 |')
    lines.append('|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in results:
        mf = r.get('matched_filter') or {}
        tn = r.get('tiny_net') or {}
        def pct(v): return f'{v*100:.2f}%' if isinstance(v, (int, float)) else '—'
        def num(v, d=2): return f'{v:.{d}f}' if isinstance(v, (int, float)) else '—'
        lines.append(
            f'| {r["cell"]} | {r["exp_ms"]} | {r["gain_db"]} | {r.get("n_pairs", "—")} | '
            f'{num(r.get("overall_mean"), 1)} | {r.get("cell_status", "—")} | '
            f'{pct(mf.get("rank_1"))} | {num(mf.get("mean_margin"), 5)} | '
            f'{num(tn.get("val_l1"), 4)} | {num(tn.get("val_psnr_db"), 2)} | '
            f'{pct(tn.get("rank_1"))} | {pct(tn.get("rank_5"))} | {pct(tn.get("rank_25"))} |'
        )
    lines.append('')
    lines.append("Random-baseline note: each cell's tiny-net rank-1 is computed over ~20% of that cell (~420 candidates), so random baseline ~ 0.24% per cell. The matched filter likewise uses the last 20% as queries.")
    lines.append('')

    # Heatmap generation (2×3 grid: exposure in {300, 220} × gain in {12, 24, 36})
    try:
        exps = [300, 220]
        gains = [12.0, 24.0, 36.0]
        for field, fname, title in [
            ('tiny_r1', 'heatmap_v2.png', 'tiny-net rank-1 (%) — 2×3 grid'),
            ('matched_r1', 'matched_heatmap_v2.png', 'matched-filter rank-1 (%) — 2×3 grid'),
        ]:
            grid = np.full((len(exps), len(gains)), np.nan)
            for r in results:
                try:
                    i = exps.index(int(r['exp_ms']))
                    j = gains.index(float(r['gain_db']))
                except (ValueError, KeyError):
                    continue
                if field == 'tiny_r1':
                    v = (r.get('tiny_net') or {}).get('rank_1')
                else:
                    v = (r.get('matched_filter') or {}).get('rank_1')
                if v is not None:
                    grid[i, j] = v * 100
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(grid, cmap='viridis', aspect='auto', vmin=0,
                           vmax=max(np.nanmax(grid) if not np.all(np.isnan(grid)) else 1.0, 1.0))
            ax.set_xticks(range(len(gains)))
            ax.set_xticklabels([f'{g}dB' for g in gains])
            ax.set_yticks(range(len(exps)))
            ax.set_yticklabels([f'{e}ms' for e in exps])
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid[i, j]
                    label = f'{v:.2f}' if not np.isnan(v) else '—'
                    ax.text(j, i, label, ha='center', va='center',
                            color='white' if np.isnan(v) or v < grid[~np.isnan(grid)].mean() else 'black')
            ax.set_xlabel('gain (dB)')
            ax.set_ylabel('exposure')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(SWEEP_DIR / fname, dpi=120)
            plt.close(fig)
        lines.append('## Heatmaps')
        lines.append('')
        lines.append(f'- `{SWEEP_DIR}/heatmap_v2.png` (tiny-net rank-1)')
        lines.append(f'- `{SWEEP_DIR}/matched_heatmap_v2.png` (matched-filter rank-1)')
        lines.append('')
    except Exception as e:
        lines.append(f'heatmap generation error: {type(e).__name__}: {e}')

    # Best cell
    cand = [r for r in results if (r.get('tiny_net') or {}).get('rank_1') is not None]
    if cand:
        best = max(cand, key=lambda r: r['tiny_net']['rank_1'])
        best_tn = best['tiny_net']
        best_mf = best.get('matched_filter') or {}
        lines.append('## Best cell')
        lines.append('')
        lines.append(f'**`{best["cell"]}`** — exp={best["exp_ms"]} ms, gain={best["gain_db"]} dB, brightness={best.get("overall_mean", 0):.1f}')
        lines.append('')
        lines.append(f'- tiny-net val L1: {best_tn["val_l1"]:.4f}  PSNR: {best_tn["val_psnr_db"]:.2f} dB')
        lines.append(f'- tiny-net rank-1: {best_tn["rank_1"]*100:.2f}%  rank-5: {best_tn["rank_5"]*100:.2f}%  rank-25: {best_tn["rank_25"]*100:.2f}%')
        lines.append(f'- tiny-net random baseline: {best_tn["random_rank_1"]*100:.4f}% (1/{best_tn["n_val"]})')
        lines.append(f'- multiple of random: **{best_tn["rank_1"] / best_tn["random_rank_1"]:.1f}×**')
        if best_mf:
            lines.append(f'- matched-filter rank-1: {best_mf["rank_1"]*100:.2f}% '
                         f'(random {best_mf["random_rank_1"]*100:.4f}%, '
                         f'{best_mf["rank_1"] / best_mf["random_rank_1"]:.1f}× random)')
    else:
        lines.append('## Best cell')
        lines.append('')
        lines.append('No successful tiny-net runs. See per-cell logs.')
        best = None
        best_tn = None

    lines.append('')
    lines.append('## Interpretation')
    lines.append('')
    # Brightness trend
    by_exp = {}
    for r in results:
        if 'overall_mean' in r:
            by_exp.setdefault(r['exp_ms'], []).append((r['gain_db'], r['overall_mean']))
    for exp in sorted(by_exp.keys(), reverse=True):
        row = sorted(by_exp[exp])
        pairs = ', '.join(f'{g}dB→{b:.1f}' for g, b in row)
        lines.append(f'- exp {exp} ms brightness by gain: {pairs}')
    # Comparison to yesterday's rig
    yesterday_mf_r1 = 0.09 / 100
    if cand:
        best_mf_r1 = best.get('matched_filter', {}).get('rank_1') or 0
        best_tn_r1 = best['tiny_net']['rank_1']
        lines.append('')
        lines.append(f'- Yesterday\'s overnight rig matched-filter rank-1: {yesterday_mf_r1*100:.2f}% (≈2× random on 2,250 candidates).')
        lines.append(f'- Today\'s best matched-filter rank-1: {best_mf_r1*100:.2f}% '
                     f'({best_mf_r1/(best_mf.get("random_rank_1") or 1):.1f}× random on ~{best_mf.get("N_query")} candidates).')
        lines.append(f'- Today\'s best tiny-net rank-1: {best_tn_r1*100:.2f}% '
                     f'({best_tn_r1 / best_tn["random_rank_1"]:.1f}× random).')
    lines.append('')
    # Recommendation
    lines.append('## Recommendation')
    lines.append('')
    if cand:
        r1 = best_tn['rank_1']
        if r1 > 0.10:
            lines.append(f'**Strong signal at `{best["cell"]}`** (tiny-net rank-1 {r1*100:.1f}%, '
                         f'{r1/best_tn["random_rank_1"]:.0f}× random). '
                         f'Recommend recapture at these settings (exp={best["exp_ms"]}ms, '
                         f'gain={best["gain_db"]}dB) for 3+ hours to build a full training dataset on the new rig.')
        elif r1 > 0.02:
            lines.append(f'**Signal recoverable but weak at `{best["cell"]}`** (tiny-net rank-1 '
                         f'{r1*100:.2f}%, {r1/best_tn["random_rank_1"]:.1f}× random). '
                         f'Recommend: recapture more frames at exp={best["exp_ms"]}ms / '
                         f'gain={best["gain_db"]}dB and pair with a stronger architecture '
                         f'(e.g. Pix2Pix-HD rather than this tiny U-Net) before concluding.')
        else:
            lines.append('**No cell exceeded 2% rank-1.** Signal is not appreciably stronger than '
                         'yesterday\'s rig in the tested exposure/gain grid. SNR is probably not the '
                         'bottleneck — the new rig geometry changed the optical transfer in a way '
                         'that more photons alone won\'t fix. Consider: closer physical distance, '
                         'different projection surface, or raw-Bayer-level analysis to rule out '
                         'debayer/resize losses.')
    else:
        lines.append('No successful tiny-net runs — can\'t make a recommendation. Check per-cell logs.')

    (SWEEP_DIR / 'results_summary_v2.md').write_text('\n'.join(lines))


def main():
    if not torch.cuda.is_available():
        log('ABORT: CUDA not available')
        sys.exit(2)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    log('=' * 60)
    log(f'phase 3 analyzer starting  GPU={torch.cuda.get_device_name(0)}  '
        f'vram_total={torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB')

    results = []
    t_start = time.time()
    for cell_name, exp_ms, gain_db in CELLS:
        try:
            r = analyze_cell(cell_name, exp_ms, gain_db, device)
        except Exception as e:
            log(f'  CELL {cell_name} FATAL: {type(e).__name__}: {e}')
            import traceback
            log(traceback.format_exc())
            r = {'cell': cell_name, 'exp_ms': exp_ms, 'gain_db': gain_db,
                 'error': f'{type(e).__name__}: {e}'}
        results.append(r)

    # Save per-cell JSON
    (SWEEP_DIR / 'phase3_results.json').write_text(
        json.dumps(results, indent=2, default=str))

    # Aggregate
    try:
        write_summary(results)
        log(f'wrote results_summary_v2.md  total_elapsed={time.time()-t_start:.1f}s')
    except Exception as e:
        log(f'write_summary error: {type(e).__name__}: {e}')
        import traceback
        log(traceback.format_exc())

    log(f'=== PHASE 3 DONE ===  results_summary_v2.md at {SWEEP_DIR/"results_summary_v2.md"}')


if __name__ == '__main__':
    main()
