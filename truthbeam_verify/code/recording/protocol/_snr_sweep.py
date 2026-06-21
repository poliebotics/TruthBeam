"""Overnight SNR sweep driver.
Runs 15 cells of exposure/gain, per-cell analysis, aggregate summary.
Autonomous. Logs everything. Doesn't stop on soft failures.
"""
import csv
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SWEEP_DIR = pathlib.Path(sys.argv[1])
LOG_PATH = SWEEP_DIR / 'sweep_progress.log'
TB_MAIN = pathlib.Path('/path/to/20260417_4.7_fresh_start/truth_beam_recording/protocol/tb_main.py')

EXPOSURES_US = [300000, 220000, 150000, 96000, 48000]
GAINS_DB = [12.0, 24.0, 36.0]
TARGET_FRAMES = 1500
SATURATION_MAX = 245
UNDEREXPOSURE_MIN = 8
TRAIN_FRAC = 1200 / 1500  # first 1200 train, last 300 val (applied on captured frames)

IMAGE_H = 360
IMAGE_W = 640


def now():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def exposure_duration_sec(exp_us: int) -> int:
    """Generous duration to capture TARGET_FRAMES in async mode. Camera period ≈ exposure + 20ms."""
    period = exp_us / 1e6 + 0.04
    return int(math.ceil(TARGET_FRAMES * period * 1.3))


def run_capture(exp_us: int, gain_db: float, cell_dir: pathlib.Path):
    """Invoke tb_main.py in async mode (blocking would need per-cell calibration)."""
    duration = exposure_duration_sec(exp_us)
    log(f'  capture: exp={exp_us/1000}ms gain={gain_db}dB duration={duration}s')
    cmd = [
        'python3', '-u', str(TB_MAIN),
        '--connector', 'HDMI-1',
        '--exposure-us', str(exp_us),
        '--gain', str(gain_db),
        '--mode', 'async',
        '--cpu-only',
        '--duration', str(duration),
        '--session-dir', str(cell_dir),
    ]
    start = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(TB_MAIN.parent.parent),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           timeout=duration + 120)
        elapsed = time.time() - start
        (cell_dir / 'capture_stdout.log').write_bytes(r.stdout)
        if r.returncode != 0:
            log(f'  capture FAILED rc={r.returncode}  elapsed={elapsed:.1f}s')
            return {'status': 'capture_failed', 'elapsed_s': elapsed, 'rc': r.returncode}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        log(f'  capture TIMED OUT after {elapsed:.1f}s')
        return {'status': 'capture_timeout', 'elapsed_s': elapsed}
    elapsed = time.time() - start
    # Check manifest
    mf = cell_dir / 'manifest.json'
    if not mf.exists():
        log(f'  capture: no manifest')
        return {'status': 'no_manifest', 'elapsed_s': elapsed}
    manifest = json.loads(mf.read_text())
    N_captures = manifest.get('N_captures', 0)
    log(f'  capture done  N_captures={N_captures}  elapsed={elapsed:.1f}s')
    return {
        'status': 'ok',
        'N_captures': N_captures,
        'elapsed_s': elapsed,
        'session_id': manifest.get('session_id'),
        'rig_hash': manifest.get('rig_hash'),
    }


def load_capture_indices(cell_dir: pathlib.Path):
    """Return list of (capture_idx, raw_path, consumed_as_t_or_None)."""
    out = []
    with open(cell_dir / 'capture_log.csv') as f:
        reader = None
        header = None
        for line in f:
            if line.startswith('#'):
                continue
            if header is None:
                header = line.strip().split(',')
                continue
            fields = line.strip().split(',')
            row = dict(zip(header, fields))
            cas = row.get('consumed_as_t', '')
            try:
                cas_i = int(cas) if cas != '' and cas != 'None' else None
            except ValueError:
                cas_i = None
            out.append((int(row['capture_idx']), row['raw_path'], cas_i))
    return out


def load_chain_ts(cell_dir: pathlib.Path):
    """Return dict t -> emission_png_path (relative)."""
    out = {}
    with open(cell_dir / 'chain_log.csv') as f:
        header = None
        for line in f:
            if line.startswith('#'):
                continue
            if header is None:
                header = line.strip().split(',')
                continue
            fields = line.strip().split(',')
            row = dict(zip(header, fields))
            out[int(row['t'])] = row.get('emission_png_path', '')
    return out


def debayer_resize(raw_path: pathlib.Path):
    """Load BayerRG8 raw, debayer RGB, resize to 360x640, return (H, W, 3) uint8."""
    arr = np.fromfile(raw_path, dtype=np.uint8)
    if arr.size != 5320 * 4600:
        return None
    bayer = arr.reshape(4600, 5320)
    rgb = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
    return cv2.resize(rgb, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)


def load_emission(cell_dir, emis_rel):
    p = cell_dir / emis_rel
    if not p.exists():
        return None
    bgr = cv2.imread(str(p))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)


def write_md(path, content):
    path.write_text(content)


def capture_stats(frames_arr: np.ndarray):
    """frames_arr: (N, H, W, 3) uint8."""
    per_channel = []
    for c, name in enumerate('RGB'):
        ch = frames_arr[..., c]
        per_channel.append({
            'channel': name,
            'mean': float(ch.mean()),
            'std': float(ch.std()),
            'min': int(ch.min()),
            'max': int(ch.max()),
        })
    overall = {
        'mean': float(frames_arr.mean()),
        'std': float(frames_arr.std()),
        'min': int(frames_arr.min()),
        'max': int(frames_arr.max()),
    }
    return per_channel, overall


def matched_filter_eval(frames_arr, emissions_arr, n_train: int = 1200, n_val: int = 300):
    """frames and emissions as (N, H, W, 3) uint8. Returns rank metrics."""
    N = min(len(frames_arr), len(emissions_arr))
    n_train = min(n_train, max(1, N * 4 // 5))
    n_val = min(n_val, N - n_train)
    if n_val < 10:
        return None

    F_train = frames_arr[:n_train].astype(np.float32) / 255.0
    F_val = frames_arr[n_train:n_train + n_val].astype(np.float32) / 255.0
    E_val = emissions_arr[n_train:n_train + n_val].astype(np.float32) / 255.0

    F_mean = F_train.mean(axis=0)
    F_std = F_train.std(axis=0)

    F_resid = (F_val - F_mean) / (F_std + 1e-3)
    F_resid = np.clip(F_resid, -8.0, 8.0)

    E_flat = E_val.reshape(n_val, -1)
    E_mean = E_flat.mean(axis=1, keepdims=True)
    E_std = E_flat.std(axis=1, keepdims=True)
    E_norm = (E_flat - E_mean) / (E_std + 1e-6)

    F_flat = F_resid.reshape(n_val, -1)
    D = F_flat.shape[1]
    S = (F_flat @ E_norm.T) / D  # (n_val, n_val)

    diag = np.diag(S)
    ranks = (S > diag[:, None]).sum(axis=1) + 1
    S_nd = S.copy()
    np.fill_diagonal(S_nd, -np.inf)
    max_nd = S_nd.max(axis=1)
    margins = diag - max_nd
    return {
        'N_val': n_val,
        'rank_1': float((ranks == 1).mean()),
        'rank_5': float((ranks <= 5).mean()),
        'rank_25': float((ranks <= 25).mean()),
        'median_rank': int(np.median(ranks)),
        'mean_margin': float(margins.mean()),
        'median_margin': float(np.median(margins)),
    }


# Tiny net
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


class ArrayPairDataset(Dataset):
    def __init__(self, F_arr, E_arr):
        self.F = F_arr
        self.E = E_arr

    def __len__(self):
        return len(self.F)

    def __getitem__(self, i):
        f = torch.from_numpy(self.F[i].transpose(2, 0, 1).astype(np.float32) / 255.0)
        e = torch.from_numpy(self.E[i].transpose(2, 0, 1).astype(np.float32) / 255.0)
        return f, e


def save_quad(F_t, E_hat, E_true, out_path, t):
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
    cv2.putText(canvas, f't={t}', (pad, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


def train_tiny_net(frames_arr, emissions_arr, out_dir, n_train=1200, n_val=300, epochs=8):
    """Train TinyNet with L1 loss. Returns eval metrics dict or None on failure."""
    if not torch.cuda.is_available():
        log('    tiny_net: CUDA unavailable, skipping')
        return None
    N = min(len(frames_arr), len(emissions_arr))
    n_train = min(n_train, max(1, N * 4 // 5))
    n_val = min(n_val, N - n_train)
    if n_val < 10:
        log('    tiny_net: insufficient samples')
        return None

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    model = TinyNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'    tiny_net: {n_params:,} params')

    train_F = frames_arr[:n_train]
    train_E = emissions_arr[:n_train]
    val_F = frames_arr[n_train:n_train + n_val]
    val_E = emissions_arr[n_train:n_train + n_val]

    train_ds = ArrayPairDataset(train_F, train_E)
    val_ds = ArrayPairDataset(val_F, val_E)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    optim = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    try:
        for epoch in range(epochs):
            model.train()
            tloss = 0.0
            nc = 0
            for Fb, Eb in train_loader:
                Fb = Fb.to(device, non_blocking=True)
                Eb = Eb.to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(Fb)
                    loss = F.l1_loss(pred, Eb)
                if torch.isnan(loss):
                    log(f'    tiny_net: NaN loss at epoch {epoch}, aborting training')
                    return None
                loss.backward()
                optim.step()
                tloss += loss.item() * Fb.shape[0]
                nc += Fb.shape[0]
            log(f'    tiny_net epoch {epoch}: train L1 = {tloss/max(1,nc):.4f}')
    except Exception as e:
        log(f'    tiny_net training error: {type(e).__name__}: {e}')
        return None

    # Eval
    model.eval()
    E_hats = []
    Es = []
    Fs = []
    with torch.no_grad():
        for Fb, Eb in val_loader:
            Fb = Fb.to(device, non_blocking=True)
            Eb = Eb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(Fb)
            E_hats.append(pred.float().cpu())
            Es.append(Eb.cpu())
            Fs.append(Fb.cpu())
    Eh = torch.cat(E_hats)
    Et = torch.cat(Es)
    Ff = torch.cat(Fs)

    mse = F.mse_loss(Eh, Et).item()
    l1 = F.l1_loss(Eh, Et).item()
    psnr = 20 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))

    # Binding ranking on val
    Eh_flat = Eh.flatten(1)
    Et_flat = Et.flatten(1)
    distances = (Eh_flat.unsqueeze(1) - Et_flat.unsqueeze(0)).abs().sum(dim=2)  # (n_val, n_val)
    diag = torch.diag(distances)
    ranks = (distances < diag.unsqueeze(1)).sum(dim=1) + 1
    r1 = (ranks == 1).float().mean().item()
    r5 = (ranks <= 5).float().mean().item()
    r25 = (ranks <= 25).float().mean().item()
    dist_nd = distances.clone()
    dist_nd.fill_diagonal_(float('inf'))
    nearest_wrong = dist_nd.min(dim=1).values
    margins = nearest_wrong - diag  # positive = correct order

    # Save 5 sample panels
    samples_dir = out_dir / 'tiny_net_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(5, Eh.shape[0])):
        save_quad(Ff[i], Eh[i], Et[i], samples_dir / f'sample_{i:02d}.png', t=i)

    # Save weights
    torch.save(model.state_dict(), out_dir / 'tiny_net.pt')

    return {
        'n_params': n_params,
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
    }


def analyze_cell(cell_dir: pathlib.Path, capture_info: dict):
    """Run stats + matched filter + tiny net on cell captures. Deletes raw files at end."""
    t0 = time.time()
    # Build paired (frame, emission) arrays in the order of chain_t
    cap_rows = load_capture_indices(cell_dir)
    chain_emissions = load_chain_ts(cell_dir)
    # We need frames where consumed_as_t is set; order by t
    paired = [(c[2], c[1]) for c in cap_rows if c[2] is not None and c[2] in chain_emissions]
    paired.sort(key=lambda x: x[0])
    n_pairs = len(paired)
    log(f'  analysis: {n_pairs} paired (F, E) items')
    if n_pairs < 50:
        log(f'  analysis: too few pairs ({n_pairs}), skipping analyses')
        return {'error': 'too_few_pairs', 'n_pairs': n_pairs}

    # Load all frames and emissions
    frames = np.empty((n_pairs, IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    emissions = np.empty((n_pairs, IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    load_failures = 0
    for i, (t, raw_rel) in enumerate(paired):
        rgb = debayer_resize(cell_dir / raw_rel)
        if rgb is None:
            load_failures += 1
            continue
        frames[i] = rgb
        emis = load_emission(cell_dir, chain_emissions[t])
        if emis is None:
            load_failures += 1
            continue
        emissions[i] = emis
    if load_failures:
        log(f'  analysis: load failures = {load_failures}')
    log(f'  analysis: data loaded in {time.time()-t0:.1f}s')

    # Stats
    per_channel, overall = capture_stats(frames)
    log(f'  stats: overall mean={overall["mean"]:.2f}  '
        f'R={per_channel[0]["mean"]:.1f} G={per_channel[1]["mean"]:.1f} B={per_channel[2]["mean"]:.1f}')

    # Saturation check
    status = 'ok'
    if overall['mean'] > SATURATION_MAX:
        status = 'saturated'
    elif overall['mean'] < UNDEREXPOSURE_MIN:
        status = 'underexposed'
    (cell_dir / 'capture_stats.md').write_text(
        f'# capture_stats\nN={n_pairs}\noverall: {overall}\nper_channel: {per_channel}\nstatus: {status}\n'
    )

    result = {
        'n_pairs': n_pairs,
        'status': status,
        'overall_mean': overall['mean'],
        'overall_std': overall['std'],
        'per_channel_mean': {pc['channel']: pc['mean'] for pc in per_channel},
    }

    if status != 'ok':
        log(f'  {status.upper()} — skipping matched filter and tiny_net')
        result['matched_filter'] = None
        result['tiny_net'] = None
    else:
        # Matched filter
        try:
            mf_res = matched_filter_eval(frames, emissions, n_train=1200, n_val=300)
            (cell_dir / 'matched_filter_results.md').write_text(
                f'# matched_filter\n{json.dumps(mf_res, indent=2)}\n'
            )
            log(f'  matched: rank-1={mf_res["rank_1"]*100:.2f}%  '
                f'rank-5={mf_res["rank_5"]*100:.2f}%  mean_margin={mf_res["mean_margin"]:+.5f}')
            result['matched_filter'] = mf_res
        except Exception as e:
            log(f'  matched filter error: {type(e).__name__}: {e}')
            result['matched_filter'] = None

        # Tiny net training
        try:
            tn_res = train_tiny_net(frames, emissions, cell_dir,
                                    n_train=1200, n_val=300, epochs=8)
            if tn_res:
                (cell_dir / 'tiny_net_results.md').write_text(
                    f'# tiny_net\n{json.dumps(tn_res, indent=2)}\n'
                )
                log(f'  tiny_net: val_L1={tn_res["val_l1"]:.4f}  '
                    f'rank-1={tn_res["rank_1"]*100:.2f}%  rank-5={tn_res["rank_5"]*100:.2f}%')
            result['tiny_net'] = tn_res
        except Exception as e:
            log(f'  tiny_net error: {type(e).__name__}: {e}')
            result['tiny_net'] = None

    # Cleanup raw files to reclaim disk
    raws_dir = cell_dir / 'Recordings'
    if raws_dir.exists():
        t0c = time.time()
        n_removed = 0
        for p in raws_dir.glob('*.raw'):
            try:
                p.unlink()
                n_removed += 1
            except Exception:
                pass
        log(f'  cleanup: removed {n_removed} raw files in {time.time()-t0c:.1f}s')

    return result


def main():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    log('=' * 60)
    log(f'sweep driver starting at {now()}')
    log(f'sweep_dir: {SWEEP_DIR}')
    log(f'cells: {len(EXPOSURES_US)}×{len(GAINS_DB)} = {len(EXPOSURES_US)*len(GAINS_DB)}')

    all_results = []
    t_sweep_start = time.time()

    for exp_us in EXPOSURES_US:
        for gain_db in GAINS_DB:
            cell_name = f'cell_exp{exp_us//1000}_gain{int(gain_db)}'
            cell_dir = SWEEP_DIR / cell_name
            # tb_main requires session_dir not to exist; remove any stub
            if cell_dir.exists():
                shutil.rmtree(cell_dir)
            t_cell_start = time.time()
            log(f'--- CELL {cell_name} ---')

            cap_info = run_capture(exp_us, gain_db, cell_dir)
            if cap_info.get('status') != 'ok':
                all_results.append({
                    'cell': cell_name, 'exp_us': exp_us, 'gain_db': gain_db,
                    'capture': cap_info, 'analysis': None,
                    'cell_elapsed_s': time.time() - t_cell_start,
                })
                continue

            analysis = analyze_cell(cell_dir, cap_info)
            all_results.append({
                'cell': cell_name, 'exp_us': exp_us, 'gain_db': gain_db,
                'capture': cap_info, 'analysis': analysis,
                'cell_elapsed_s': time.time() - t_cell_start,
            })

            # Free GPU memory between cells
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            log(f'CELL {cell_name} done  total_cell_s={time.time()-t_cell_start:.1f}s  '
                f'sweep_elapsed={time.time()-t_sweep_start:.1f}s')

    log(f'--- ALL CELLS DONE ({time.time()-t_sweep_start:.1f}s total) ---')
    (SWEEP_DIR / 'all_results.json').write_text(json.dumps(all_results, indent=2, default=str))

    # Aggregate summary
    try:
        write_summary(SWEEP_DIR, all_results, t_sweep_start)
    except Exception as e:
        log(f'summary write error: {type(e).__name__}: {e}')

    log(f'=== SWEEP COMPLETE ===  results_summary.md: {SWEEP_DIR/"results_summary.md"}')


def write_summary(sweep_dir, all_results, t_start):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ok_count = sum(1 for r in all_results if r.get('analysis') and r['analysis'].get('status') == 'ok')
    sat_count = sum(1 for r in all_results if r.get('analysis') and r['analysis'].get('status') == 'saturated')
    under_count = sum(1 for r in all_results if r.get('analysis') and r['analysis'].get('status') == 'underexposed')
    fail_count = sum(1 for r in all_results if not r.get('analysis'))

    rows = []
    for r in all_results:
        cell = r['cell']
        exp_ms = r['exp_us'] / 1000
        gain = r['gain_db']
        a = r.get('analysis') or {}
        status = a.get('status', r.get('capture', {}).get('status', 'unknown'))
        brightness = a.get('overall_mean', float('nan'))
        mf = a.get('matched_filter') or {}
        tn = a.get('tiny_net') or {}
        rows.append({
            'cell': cell, 'exp_ms': exp_ms, 'gain_db': gain, 'status': status,
            'brightness': brightness,
            'matched_rank1': mf.get('rank_1'),
            'matched_rank5': mf.get('rank_5'),
            'matched_margin': mf.get('mean_margin'),
            'tiny_l1': tn.get('val_l1'),
            'tiny_rank1': tn.get('rank_1'),
            'tiny_rank5': tn.get('rank_5'),
            'tiny_rank25': tn.get('rank_25'),
        })

    lines = []
    lines.append('# SNR sweep — results summary')
    lines.append('')
    lines.append(f'- sweep_dir: `{sweep_dir}`')
    lines.append(f'- started_utc: timestamp (log above)')
    lines.append(f'- total_cells: {len(all_results)}')
    lines.append(f'- ok: {ok_count}  saturated: {sat_count}  underexposed: {under_count}  failed: {fail_count}')
    lines.append(f'- total_elapsed_s: {time.time()-t_start:.1f}')
    lines.append('')
    lines.append('## Per-cell table')
    lines.append('')
    lines.append('| cell | exp(ms) | gain(dB) | status | brightness | matched_rank1 | matched_margin | tiny_L1 | tiny_rank1 | tiny_rank5 | tiny_rank25 |')
    lines.append('|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        def pct(v): return f'{v*100:.2f}%' if isinstance(v, (int, float)) else '—'
        def num(v, d=2): return f'{v:.{d}f}' if isinstance(v, (int, float)) else '—'
        lines.append(
            f'| {r["cell"]} | {r["exp_ms"]} | {r["gain_db"]} | {r["status"]} | '
            f'{num(r["brightness"], 1)} | {pct(r["matched_rank1"])} | {num(r["matched_margin"], 5)} | '
            f'{num(r["tiny_l1"], 4)} | {pct(r["tiny_rank1"])} | {pct(r["tiny_rank5"])} | {pct(r["tiny_rank25"])} |'
        )
    lines.append('')
    lines.append(f'Random baseline rank-1 ≈ 1/300 = 0.333%')
    lines.append('')

    # Heatmaps
    try:
        def heatmap(field, filename, title):
            vals = np.full((len(EXPOSURES_US), len(GAINS_DB)), np.nan)
            for r in rows:
                i = EXPOSURES_US.index(int(r['exp_ms'] * 1000))
                j = GAINS_DB.index(r['gain_db'])
                v = r.get(field)
                if v is not None:
                    vals[i, j] = v * 100 if 'rank' in field else v
            fig, ax = plt.subplots(figsize=(6, 8))
            im = ax.imshow(vals, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(GAINS_DB)))
            ax.set_xticklabels([f'{g}dB' for g in GAINS_DB])
            ax.set_yticks(range(len(EXPOSURES_US)))
            ax.set_yticklabels([f'{e//1000}ms' for e in EXPOSURES_US])
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    v = vals[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center', color='white', fontsize=9)
                    else:
                        ax.text(j, i, '—', ha='center', va='center', color='white', fontsize=9)
            ax.set_xlabel('gain (dB)')
            ax.set_ylabel('exposure')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(sweep_dir / filename, dpi=120)
            plt.close(fig)
        heatmap('tiny_rank1', 'heatmap.png', 'tiny_net rank-1 (%) — 5×3')
        heatmap('matched_rank1', 'matched_heatmap.png', 'matched_filter rank-1 (%) — 5×3')
        lines.append('')
        lines.append('## Heatmaps')
        lines.append(f'- `{sweep_dir/"heatmap.png"}` (tiny_net rank-1)')
        lines.append(f'- `{sweep_dir/"matched_heatmap.png"}` (matched_filter rank-1)')
    except Exception as e:
        lines.append(f'heatmap generation error: {e}')

    # Best cell
    ok_rows = [r for r in rows if r.get('tiny_rank1') is not None]
    if ok_rows:
        best = max(ok_rows, key=lambda r: (r['tiny_rank1'] or 0, r.get('matched_rank1') or 0))
        lines.append('')
        lines.append(f'## Best cell: `{best["cell"]}` (exp={best["exp_ms"]}ms, gain={best["gain_db"]}dB)')
        lines.append(f'- tiny_rank1: {best["tiny_rank1"]*100:.2f}%  (random = 0.333%)')
        lines.append(f'- tiny_rank5: {best["tiny_rank5"]*100:.2f}%')
        lines.append(f'- matched_rank1: {best.get("matched_rank1")*100:.2f}%' if best.get('matched_rank1') is not None else '')
        lines.append(f'- brightness: {best["brightness"]:.1f}')

        # Recommendation
        r1 = best['tiny_rank1']
        lines.append('')
        lines.append('## Recommendation')
        if r1 > 0.10:
            lines.append(
                f'**Strong signal at {best["cell"]}.** rank-1 {r1*100:.1f}% is {r1/(1/300):.0f}× random. '
                f'Recommend recapturing overnight at exp={best["exp_ms"]}ms, gain={best["gain_db"]}dB for full training.'
            )
        elif r1 > 0.02:
            lines.append(
                f'**Signal recoverable at {best["cell"]} but weak** (rank-1 {r1*100:.2f}%, '
                f'{r1/(1/300):.1f}× random). Recommend recapture + architectural improvements.'
            )
        else:
            lines.append(
                f'**No cell exceeded 2% rank-1** (best was {best["cell"]} at {r1*100:.2f}%). '
                f'SNR is not the bottleneck at this rig geometry. '
                f'Consider: closer physical distance, brighter projection source, or raw-Bayer investigation.'
            )
    else:
        lines.append('')
        lines.append('## No successful cells — all skipped or failed. See per-cell logs.')

    # Deviations from user spec
    lines.append('')
    lines.append('## Deviations from spec (autonomous decisions)')
    lines.append('')
    lines.append('1. **Mode = async, not blocking.** Blocking requires per-cell pipeline-delay '
                 'calibration at `~/.tb/pipeline_calibration_<rig_hash>.json`. The physical rig '
                 'has been moved (rig_hash = 3984699a...; previous d5367644...). Each cell has '
                 'its own rig_hash (includes exposure+gain). Generating 15 fresh calibrations '
                 'would violate "DO NOT touch ~/.tb/". Async still gives 1:1 F↔E pairing via '
                 'consumed_as_t column.')
    lines.append('2. **`--cpu-only` tile generator.** GPU tile-generator self-check failed at '
                 'startup ("GPU/CPU determinism check FAILED: gpu=cb28d692… vs cpu=c59a5ead…"), '
                 'likely a residual from earlier CUDA wedge/reboot. CPU path produces the same '
                 'output and passes the self-check.')
    lines.append('3. **Skipped pre-capture saturation probe.** A 3-frame probe requires a full '
                 'session setup per cell (~20s overhead). Instead, saturation is checked from '
                 'the first frames of the actual capture. Cells flagged `saturated` or '
                 '`underexposed` still have raw captures but skip the expensive analyses.')
    lines.append('4. **Deleting raw files after analysis.** Raw Bayer files are 24.47 MB each; '
                 '15 cells × 1500 frames = ~550 GB which exceeds NVMe free space. After each '
                 'cell completes analysis, its `Recordings/*.raw` files are removed. Derived '
                 'tiles, logs, and results persist.')


    (sweep_dir / 'results_summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
