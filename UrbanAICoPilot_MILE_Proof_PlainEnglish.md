# Urban AI CoPilot from MILE — Plain-English Proof

This document is the plain-English mathematical proof companion for the `UrbanAICoPilot_MILE.nb` notebook and the `MILEProductModel.wl` package.

It proves the exact claim supported by the demo:

> A MILE-style recurrent latent world model is a mathematically coherent core for a camera-first urban driving product, provided it is wrapped by an explicit safety supervisor and uncertainty-aware deployment logic.

## Scope and Assumptions

### Proposition 1

The demo is a faithful executable reconstruction of the paper's model class, not a claim that the exact trained Wayve network weights have been reproduced.

### Assumptions

1. The latent dynamics use a deterministic recurrent state $h_t$ and a stochastic state $s_t$.
2. The prior and posterior over $s_t$ are diagonal Gaussians with strictly positive standard deviations.
3. The product extension adds an external safety supervisor after the learned trajectory proposal.
4. Runtime claims are taken from the paper-aligned deployment table encoded in the package.

Under these assumptions, the proof below establishes mathematical consistency of the reconstruction and scenario-specific validity of the product extension.

## Generative Model

### Proposition 2

The notebook defines a valid latent state-space model for urban driving.

The reconstructed equations are:

$$
(1)\quad h_1 \sim \delta(0)
$$

$$
(2)\quad s_1 \sim \mathcal N(0, I)
$$

$$
(3)\quad h_{t+1} = f_\theta(h_t, s_t)
$$

$$
(4)\quad s_{t+1} \sim \mathcal N\!\left(\mu_\theta(h_{t+1}, a_t),\, \sigma_\theta(h_{t+1}, a_t) I\right)
$$

$$
(5)\quad o_t \sim \mathcal N\!\left(g_\theta(h_t, s_t), I\right)
$$

$$
(6)\quad y_t \sim \mathrm{Categorical}(l_\theta(h_t, s_t))
$$

$$
(7)\quad a_t \sim \mathrm{Laplace}(\pi_\theta(h_t, s_t), 1)
$$

### Proof

Equation (1) initializes the deterministic memory at a single point, so the recurrent state has a unique start value. Equation (2) initializes the stochastic latent state with a proper Gaussian distribution. Equation (3) is deterministic, so once $(h_t, s_t)$ are known, $h_{t+1}$ is uniquely determined. Equation (4) is a proper probability law because the package constructs each component of $\sigma_\theta$ as a positive base term plus non-negative corrections, so the covariance is diagonal and positive. Equations (5), (6), and (7) are standard likelihood models for continuous observations, discrete BEV labels, and imitation actions. Therefore the pair $(h_t, s_t)$ together with these likelihoods defines a well-posed sequential latent-variable model.

## ELBO and KL Divergence

### Proposition 3

The KL correction used in the demo is finite and non-negative.

The relevant equations are:

$$
(8)\quad q_t = \mathcal N(\mu_q, \operatorname{diag}(\sigma_q^2))
$$

$$
(9)\quad p_t = \mathcal N(\mu_p, \operatorname{diag}(\sigma_p^2))
$$

$$
(10)\quad \mathrm{ELBO}_t = \mathbb E_q\left[\log p(o_t \mid h_t,s_t) + \log p(y_t \mid h_t,s_t) + \log p(a_t \mid h_t,s_t)\right] - \mathrm{KL}(q_t\|p_t)
$$

$$
(11)\quad \mathrm{KL}(q_t\|p_t) = \frac{1}{2} \sum_i \left(\frac{\sigma_{q,i}^2 + (\mu_{q,i}-\mu_{p,i})^2}{\sigma_{p,i}^2} - 1 + \log \frac{\sigma_{p,i}^2}{\sigma_{q,i}^2}\right)
$$

### Proof

Because all components of $\sigma_q$ and $\sigma_p$ are strictly positive, every denominator and logarithm in (11) is well-defined. If $\mu_q = \mu_p$ and $\sigma_q = \sigma_p$, then each summand in (11) is exactly zero, so $\mathrm{KL}(q_t\|p_t)=0$. In the general case, (11) is the standard closed-form KL divergence between diagonal Gaussian distributions, and KL divergence is always non-negative. Therefore the ELBO in (10) subtracts a finite penalty that vanishes only when posterior and prior agree exactly.

### Numerical Verification Table

| Check | Value | Interpretation |
|---|---:|---|
| $\mathrm{KL}(q\|q)$ | 0.000000 | Exact self-match gives zero penalty |
| $\mathrm{KL}(q\|p)$ | 5.950483 | Shifted posterior/prior pair gives positive penalty |

## Observation Dropout and Imagination

### Proposition 4

If each frame is dropped independently with probability $p$, then the expected number of consecutive prior-only imagination steps is $p/(1-p)$.

The relevant equations are:

$$
(12)\quad \Pr[K=k] = (1-p) p^k, \qquad k=0,1,2,\dots
$$

$$
(13)\quad \mathbb E[K] = \sum_{k\ge 0} k(1-p)p^k = \frac{p}{1-p}
$$

### Proof

Equation (12) is the geometric distribution for the number of failed observation opportunities before the next observed frame arrives. Summing the geometric series gives (13). With $p=0.25$, the expectation is

$$
\mathbb E[K] = \frac{0.25}{0.75} = \frac{1}{3}.
$$

This matches the package output exactly.

The product interpretation is also clear from the code path. In an observe step, the model uses posterior moments. In an imagine step, it rolls forward with the prior and adds a fixed uncertainty increment. Therefore the mean uncertainty should increase as the imagination fraction increases.

### Numerical Verification Table

| Regime | Observed Fraction | Mean Uncertainty |
|---|---:|---:|
| Observe-heavy | 0.9375 | 0.222929 |
| Balanced | 0.8125 | 0.231301 |
| Imagination-heavy | 0.5000 | 0.251713 |

This verifies the monotone trend predicted by the design.

## Safety Supervisor

### Proposition 5

In the audited `PedestrianConflict` rollout, the external safety supervisor reduces measured trajectory risk.

The relevant equations are:

$$
(14)\quad \mathrm{Risk}(\text{path}) = \operatorname{Clamp}(0.55\,\mathrm{ObstacleRisk} + 0.45\,\mathrm{RoadRisk}, 0, 1.5)
$$

$$
(15)\quad \mathrm{ObstacleRisk} = \sum_j \mathrm{severity}_j \exp\!\left(-\frac{d_j}{\mathrm{radius}_j + 0.5}\right)
$$

$$
(16)\quad \text{if signal is red and the path crosses the stop line, clip } y \text{ to } \min(y, L_{\mathrm{stop}}-0.4)
$$

$$
(17)\quad \text{if } d_{\min} < d_{\mathrm{safe}}, \text{ shift the path laterally and scale speed by } 0.55
$$

### Proof

Equation (14) defines the scalar risk used by the simulator. Equation (16) removes red-light stop-line violations by construction, because the clipped path can no longer pass beyond $L_{\mathrm{stop}}-0.4$. Equation (17) increases clearance in the audited pedestrian scenario by moving the path away from the obstacle field and reducing speed. The package does not claim a universal theorem that every imaginable scenario must improve under every rule intervention, but it does verify the benchmark claim numerically for the seeded scenario used by the audit.

### Numerical Verification Table

| Metric | Without Safety | With Safety |
|---|---:|---:|
| Mean risk | 0.438953 | 0.334470 |
| Normalized reward | 1.910236 | 0.610554 |
| Intervention rate | 0.0000 | 0.9375 |

So the supervisor improves risk in the audited scenario, while lowering reward because it acts conservatively almost every step.

## Runtime and Product Conclusion

### Proposition 6

Fully recurrent deployment is faster than reset-state deployment in the default demo parameters.

The relevant equations are:

$$
(18)\quad m_{\mathrm{reset}} = \frac{1000}{6.2}\ \text{milliseconds}
$$

$$
(19)\quad m_{\mathrm{recurrent}} = \frac{1000}{43.0}\ \text{milliseconds}
$$

$$
(20)\quad f = \frac{1000}{m}
$$

### Proof

Equations (18) and (19) are the deployment latencies encoded in the paper-aligned runtime table. Since $43.0 > 6.2$, equation (20) immediately implies $m_{\mathrm{recurrent}} < m_{\mathrm{reset}}$, so recurrent deployment is faster. The same deployment table keeps normalized reward at $0.67$ for both reset-state and fully recurrent modes, so the product story is not "trade all quality for speed"; it is "preserve the main reward metric while greatly increasing cycle rate."

### Numerical Verification Table

| Mode | Frequency (Hz) |
|---|---:|
| Reset-state | 6.2 |
| Fully recurrent | 43.0 |

## Final Conclusion

The proof establishes the exact claim supported by the demo:

1. The latent recurrent world model is mathematically well-defined.
2. The ELBO correction is finite and non-negative.
3. Observation dropout has the expected geometric interpretation and increases uncertainty under imagination-heavy rollout.
4. The safety supervisor improves risk in the benchmark pedestrian scenario used by the audit.
5. Fully recurrent deployment preserves the main reward metric while greatly improving cycle rate.

That is enough to justify the demo's product framing: MILE is a plausible performance layer for an urban L2+/L3 stack, but it must be wrapped by an explicit safety and deployment layer to become product-ready.

## Executable Verification

```wolfram
SeedRandom[42];
Get[FileNameJoin[{NotebookDirectory[], "MILEProductModel.wl"}]];

params = DefaultDemoParameters[];
posterior = PosteriorMoments[
  ConstantArray[0., params["Model"]["HistoryDimension"]],
  {0.7, 0.2, 0.0, 0.8, 0.6, 0.3, 0.2, 0.1},
  {0., 0.},
  params
];
prior = PriorMoments[
  ConstantArray[0., params["Model"]["HistoryDimension"]],
  ConstantArray[0., params["Model"]["LatentDimension"]],
  {0., 0.},
  params
];

unsafeRun = SimulateEpisode[
  "PedestrianConflict", params,
  "Steps" -> 16,
  "ImaginationFraction" -> 0.25,
  "EnableSafetySupervisor" -> False,
  "Seed" -> 42
];
safeRun = SimulateEpisode[
  "PedestrianConflict", params,
  "Steps" -> 16,
  "ImaginationFraction" -> 0.25,
  "EnableSafetySupervisor" -> True,
  "Seed" -> 42
];

<|
  "KLSelf" -> GaussianKLDivergence[
    posterior["Mean"], posterior["Sigma"],
    posterior["Mean"], posterior["Sigma"]
  ],
  "KLShifted" -> GaussianKLDivergence[
    posterior["Mean"], posterior["Sigma"],
    prior["Mean"], prior["Sigma"]
  ],
  "DropoutMean" -> ObservationDropoutMean[params["Paper"]["ObservationDropout"]],
  "UnsafeMeanRisk" -> unsafeRun["Metrics"]["MeanRisk"],
  "SafeMeanRisk" -> safeRun["Metrics"]["MeanRisk"],
  "ResetHz" -> RuntimeModel[params, "ResetState"]["FrequencyHz"],
  "RecurrentHz" -> RuntimeModel[params, "FullyRecurrent"]["FrequencyHz"]
|>
```
