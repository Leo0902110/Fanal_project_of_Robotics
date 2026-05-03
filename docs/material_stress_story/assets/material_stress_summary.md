# Material-Stress Evaluation Summary

This evaluation combines material-level render profiles with material-matched observation pseudo-blur.
Each profile should use a distinct seed range to avoid identical trajectories across object materials.
When smoke-test and full-run outputs coexist, the summary uses the largest evaluation run per material profile.
Rates are reported with Wilson 95% confidence intervals.

Overall success: 182/200 (91%); overall grasp: 200/200 (100%).

| Object profile | Seed range | Success | Success rate | 95% CI | Grasp | Grasp rate | Mean steps | Mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transparent | 7100-7149 | 43/50 | 86% | 74%-93% | 50/50 | 100% | 23.1 | 6.01 |
| Dark | 8100-8149 | 43/50 | 86% | 74%-93% | 50/50 | 100% | 23.0 | 5.79 |
| Reflective | 9100-9149 | 48/50 | 96% | 87%-99% | 50/50 | 100% | 18.9 | 5.92 |
| Low-texture | 10100-10149 | 48/50 | 96% | 87%-99% | 50/50 | 100% | 19.0 | 6.10 |

## Interpretation Note

These results are stronger than render-only material profiles because the observation stream is also stressed.
They are still simulation stress tests, not real-world transparent/depth-sensor validation.
