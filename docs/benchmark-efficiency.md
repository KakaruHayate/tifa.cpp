# Compute efficiency: tifa_ggml vs PyTorch

20 clips (123.4 s of audio), warm start, the full pipeline per file (decode + resample + mel + network + Viterbi).

| engine | backend | dtype | ms / file | × realtime | model load |
|---|---|---|---|---|---|
| tifa_ggml | cpu | F16 | 5456 | 1.1× | 1.93 s |
| tifa_ggml | cpu | Q4_0 | 5619 | 1.1× | 1.63 s |
| tifa_ggml | vulkan | F16 | 3312 | 1.9× | 2.39 s |
| tifa_ggml | vulkan | Q4_0 | 3326 | 1.9× | 1.90 s |
| PyTorch | cpu | F32 | 389 | 15.9× | 1.73 s |
| PyTorch | cuda | F32 | 230 | 26.9× | 2.15 s |

× realtime = audio seconds per wall-clock second (higher is better).
