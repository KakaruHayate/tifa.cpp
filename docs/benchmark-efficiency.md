# Compute efficiency: tifa_ggml vs PyTorch

20 clips (123.4 s of audio), warm start, the full pipeline per file (decode + resample + mel + network + Viterbi).

| engine | backend | dtype | ms / file | × realtime | model load |
|---|---|---|---|---|---|
| tifa_ggml | cpu | F16 | 536 | 11.5× | 0.47 s |
| tifa_ggml | cpu | Q4_0 | 629 | 9.8× | 0.29 s |
| tifa_ggml | vulkan | F16 | 148 | 41.6× | 1.01 s |
| tifa_ggml | vulkan | Q4_0 | 153 | 40.3× | 0.77 s |
| PyTorch | cpu | F32 | 373 | 16.5× | 3.07 s |
| PyTorch | cuda | F32 | 227 | 27.2× | 1.87 s |

× realtime = audio seconds per wall-clock second (higher is better).
