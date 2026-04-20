# Urban AI CoPilot from MILE

This standalone demo project turns Wayve's `Model-Based Imitation Learning for Urban Driving` into an executable Wolfram Language product study with four goals:

1. reconstruct the paper's latent probabilistic structure,
2. make the world-model mechanics executable in Wolfram Language,
3. connect the research model to a real-car product architecture,
4. validate the reconstruction with repeatable audit checks.

## Files

- `MILEProductModel.wl` — reusable Wolfram package for the latent world model, short-horizon trajectory generator, safety supervisor, and scenario study.
- `BuildUrbanAICoPilotNotebook.wls` — notebook builder that exports `UrbanAICoPilot_MILE.nb`.
- `UrbanAICoPilot_MILE.nb` - generated Mathematica notebook with the main demo sections, a plain-English proof appendix, ablations, BEV visualizations, scenario studies, and an interactive dashboard.
- `UrbanAICoPilot_MILE_Proof_PlainEnglish.md` - standalone plain-English proof companion with numbered equations, verification tables, and Wolfram verification code.
- `proof_audit.wls` - automated proof and consistency audit for the reconstruction.
- `exports/` - output location for any figures or tabular exports produced during exploration.

## Source Paper

- Anthony Hu, Gianluca Corrado, Nicolas Griffiths, Zak Murez, Corina Gurau, Hudson Yeo, Alex Kendall, Roberto Cipolla, Jamie Shotton
- `Model-Based Imitation Learning for Urban Driving`
- `NeurIPS 2022`
- Wayve / University of Cambridge

## What Is Reconstructed vs Extended

### Paper Reconstruction

- latent deterministic history state `h_t`
- stochastic latent state `s_t`
- action-conditioned prior
- posterior inference moments
- closed-form diagonal Gaussian KL
- recurrent vs reset-state deployment comparison
- image-resolution, deployment, and latent-dimension ablation tables

### Product Extension

- trajectory-output control interface
- independent safety supervisor
- uncertainty-aware imagination-mode product framing
- urban scenario library for OEM-facing L2+/L3 productization

## How To Run

### 1. Build the notebook

```powershell
wolframscript -file .\BuildUrbanAICoPilotNotebook.wls
```

The builder now performs a post-export syntax validation pass on `UrbanAICoPilot_MILE.nb` and will exit non-zero if the notebook fails round-trip parsing.
The generated notebook also includes a proof appendix, and the same proof is available as `UrbanAICoPilot_MILE_Proof_PlainEnglish.md`.

### 2. Run the audit

```powershell
wolframscript -file .\proof_audit.wls
```

### 3. Open the notebook

Open `UrbanAICoPilot_MILE.nb` in Mathematica and evaluate the notebook from top to bottom.

## Validation Notes

- The demo is intentionally honest: it reconstructs the `mathematical core` of MILE rather than pretending to retrain Wayve's exact neural network.
- The package uses `Associations` throughout, explicit `Module` scoping, and `SeedRandom[42]` in notebook entry points for reproducibility.
- The audit script checks KL behavior, observation-dropout logic, safety impact, imagination degradation, and consistency of paper ablation trends.
- Notebook validation should be run through the file-based builder or a `.wls` script. Inline `wolframscript -code` checks on Windows can fail because of shell/path quoting even when the `.nb` file itself is valid.

## Recommended Narrative

The central claim supported by the demo is:

> A MILE-style recurrent latent world model is a technically plausible core for a real urban L2+/L3 driving product, provided it is wrapped with a trajectory interface, uncertainty-aware gating, and an independent safety supervisor.
