# Pipeline-delay calibration — 20260423_225337

- method: `white_black_empirical`
- rig_hash: `3984699a504fb79821e6bcd1d81e8f7e3c64a7a99761680dac9aea104795038b`
- rig_config:
  - exposure_us: 96000
  - gain_db: 24.0
  - projector_connector: HDMI-1
  - hdmi_edid_fingerprint: a0dbcfde1ca1a8ebf617b9fbd9196b9f66da32a2d234e67416981bbdc80d78cc
  - camera_serial: 25420561
  - camera_firmware: IMX540_C/2744/1378 USB3c2rl-IMX/17
- n_trials_per_wait: 50
- wait_ms_tested: [0, 16, 32, 48, 64, 80, 96, 112, 128, 160, 200, 300, 500]

## Stable-state distributions

- stable_white_mean: 67.7283  (std 0.0462, n 20)
- stable_black_mean: 14.8692  (std 0.0086, n 20)
- classification_threshold: 41.2988

## W → B correct fraction by wait

| wait_ms | correct_fraction |
| ---: | ---: |
| 0 | 0.00 |
| 16 | 0.00 |
| 32 | 0.00 |
| 48 | 0.00 |
| 64 | 0.00 |
| 80 | 0.00 |
| 96 | 0.00 |
| 112 | 0.00 |
| 128 | 0.00 |
| 160 | 0.00 |
| 200 | 1.00 |
| 300 | 1.00 |
| 500 | 1.00 |

- p99_wait_ms: 200
- p100_wait_ms: 200

## B → W correct fraction by wait

| wait_ms | correct_fraction |
| ---: | ---: |
| 0 | 0.00 |
| 16 | 0.00 |
| 32 | 0.00 |
| 48 | 0.00 |
| 64 | 0.00 |
| 80 | 0.00 |
| 96 | 1.00 |
| 112 | 1.00 |
| 128 | 1.00 |
| 160 | 1.00 |
| 200 | 1.00 |
| 300 | 1.00 |
| 500 | 1.00 |

- p99_wait_ms: 96
- p100_wait_ms: 96

## Recommendation

- **recommended_wait_ms**: 220
- confidence: 50/50 trials transitioned cleanly at the p100 wait for both directions; recommended = max(W→B.p100=200, B→W.p100=96) + 20ms safety margin = 220 ms

Calibration JSON written to `~/.tb/pipeline_calibration_3984699a504fb798.json`.