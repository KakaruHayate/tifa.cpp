# Compute efficiency: tifa_ggml vs PyTorch

20 clips (123.4 s of audio), warm start, the full pipeline per file (decode + resample + mel + network + Viterbi).

| engine | backend | dtype | ms / file | × realtime | model load |
|---|---|---|---|---|---|
| tifa_ggml | cpu | F16 | 2441 | 2.5× | 0.75 s |
| tifa_ggml | cpu | Q4_0 | 2505 | 2.5× | 0.94 s |
| tifa_ggml | vulkan | F16 | 175 | 35.4× | 1.02 s |
| tifa_ggml | vulkan | Q4_0 | 178 | 34.7× | 0.80 s |
| PyTorch | cpu | F32 | 360 | 17.2× | 2.87 s |
| PyTorch | cuda | F32 | 232 | 26.6× | 1.85 s |

× realtime = audio seconds per wall-clock second (higher is better).
