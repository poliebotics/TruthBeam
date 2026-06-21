"""Shared tile-geometry constants for the v8 recording loop and its
GPU/CPU tile generators. Keeping these in one module avoids circular
imports between tb_loop.py and the generator backends.
"""

TILE_W, TILE_H = 1920, 1080
NUM_OCTAVES = 4
GRID_H_TABLE = [17, 34, 68, 135]
GRID_W_TABLE = [30, 60, 120, 240]
TOTAL_XOF_BYTES_PER_CHANNEL = sum(h * w for h, w in zip(GRID_H_TABLE, GRID_W_TABLE))
