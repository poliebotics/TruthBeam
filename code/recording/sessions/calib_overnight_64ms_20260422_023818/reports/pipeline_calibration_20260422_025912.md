# Pipeline-delay calibration — 20260422_025912

- method: `white_black_empirical`
- rig_hash: `d53676449f5e9915a38651b140b6a502ebd288c64eb87911da700a4687c921d0`
- rig_config:
  - exposure_us: 64000
  - gain_db: 24.0
  - projector_connector: HDMI-1
  - hdmi_edid_fingerprint: a0dbcfde1ca1a8ebf617b9fbd9196b9f66da32a2d234e67416981bbdc80d78cc
  - camera_serial: 25420561
  - camera_firmware: IMX540_C/2744/1378 USB3c2rl-IMX/17
- n_trials_per_wait: 50
- wait_ms_tested: [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800]

## Stable-state distributions

- stable_white_mean: 48.4528  (std 0.0236, n 20)
- stable_black_mean: 14.6089  (std 0.013, n 20)
- classification_threshold: 31.5308

## W → B correct fraction by wait

| wait_ms | correct_fraction |
| ---: | ---: |
| 0 | 0.00 |
| 50 | 0.00 |
| 100 | 1.00 |
| 150 | 1.00 |
| 200 | 1.00 |
| 250 | 1.00 |
| 300 | 1.00 |
| 350 | 1.00 |
| 400 | 1.00 |
| 450 | 1.00 |
| 500 | 1.00 |
| 600 | 1.00 |
| 700 | 1.00 |
| 800 | 1.00 |

- p99_wait_ms: 100
- p100_wait_ms: 100

## B → W correct fraction by wait

| wait_ms | correct_fraction |
| ---: | ---: |
| 0 | 0.00 |
| 50 | 0.00 |
| 100 | 1.00 |
| 150 | 1.00 |
| 200 | 1.00 |
| 250 | 1.00 |
| 300 | 1.00 |
| 350 | 1.00 |
| 400 | 1.00 |
| 450 | 1.00 |
| 500 | 1.00 |
| 600 | 1.00 |
| 700 | 1.00 |
| 800 | 1.00 |

- p99_wait_ms: 100
- p100_wait_ms: 100

## Recommendation

- **recommended_wait_ms**: 120
- confidence: 50/50 trials transitioned cleanly at the p100 wait for both directions; recommended = max(W→B.p100=100, B→W.p100=100) + 20ms safety margin = 120 ms

Calibration JSON written to `sessions/calib_overnight_64ms_20260422_023818/calibrations/pipeline_calibration_d53676449f5e9915.json`.