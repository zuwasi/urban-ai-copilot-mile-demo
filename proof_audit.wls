#!/usr/bin/env wolframscript

Get[FileNameJoin[{DirectoryName[$InputFileName], "MILEProductModel.wl"}]];

Print["╔══════════════════════════════════════════╗"];
Print["║   PROOF AUDIT — Zero-Day Verification    ║"];
Print["╚══════════════════════════════════════════╝\n"];

passed = 0;
failed = 0;
warnings = 0;

auditResult[name_, test_, detail_] := If[
  TrueQ[test],
  (Print["  ✓ PASS: ", name]; passed++),
  (Print["  ✗ FAIL: ", name, " — ", detail]; failed++)
];

auditWarn[name_, test_, detail_] := If[
  !TrueQ[test],
  (Print["  ⚠ WARN: ", name, " — ", detail]; warnings++),
  Null
];

params = DefaultDemoParameters[];
scenario = BuildScenario["PedestrianConflict", params];

Print["\n═══ AUDIT 1: Equation Spot-Check ═══"];
posterior = PosteriorMoments[ConstantArray[0., params["Model"]["HistoryDimension"]], {0.7, 0.2, 0.0, 0.8, 0.6, 0.3, 0.2, 0.1}, {0., 0.}, params];
prior = PriorMoments[ConstantArray[0., params["Model"]["HistoryDimension"]], ConstantArray[0., params["Model"]["LatentDimension"]], {0., 0.}, params];
klSame = GaussianKLDivergence[posterior["Mean"], posterior["Sigma"], posterior["Mean"], posterior["Sigma"]];
klShifted = GaussianKLDivergence[posterior["Mean"], posterior["Sigma"], prior["Mean"], prior["Sigma"]];
auditResult["KL(self || self) ≈ 0", Abs[klSame] < 10^-8, StringForm["KL = `1`", klSame]];
auditResult["KL(posterior || prior) is non-negative", klShifted >= 0., StringForm["KL = `1`", klShifted]];

Print["\n═══ AUDIT 2: Observation Dropout Logic ═══"];
dropoutMean = ObservationDropoutMean[params["Paper"]["ObservationDropout"]];
auditResult["Observation dropout mean is finite", NumericQ[dropoutMean] && dropoutMean > 0., StringForm["mean = `1`", dropoutMean]];
auditResult["Observation dropout mean matches p/(1-p)", Abs[dropoutMean - 1./3.] < 10^-8, StringForm["mean = `1`", dropoutMean]];

Print["\n═══ AUDIT 3: Deployment Runtime Trend ═══"];
resetRuntime = RuntimeModel[params, "ResetState"];
recurrentRuntime = RuntimeModel[params, "FullyRecurrent"];
auditResult["Recurrent mode is faster than reset mode", recurrentRuntime["FrequencyHz"] > resetRuntime["FrequencyHz"], StringForm["reset = `1` Hz, recurrent = `2` Hz", resetRuntime["FrequencyHz"], recurrentRuntime["FrequencyHz"]]];

Print["\n═══ AUDIT 4: Safety Supervisor Effect ═══"];
unsafeRun = SimulateEpisode[scenario, params, "Steps" -> 16, "ImaginationFraction" -> 0.25, "EnableSafetySupervisor" -> False, "Seed" -> 42];
safeRun = SimulateEpisode[scenario, params, "Steps" -> 16, "ImaginationFraction" -> 0.25, "EnableSafetySupervisor" -> True, "Seed" -> 42];
auditResult[
  "Safety reduces mean risk in pedestrian conflict",
  safeRun["Metrics"]["MeanRisk"] <= unsafeRun["Metrics"]["MeanRisk"],
  StringForm["unsafe = `1`, safe = `2`", unsafeRun["Metrics"]["MeanRisk"], safeRun["Metrics"]["MeanRisk"]]
];
auditWarn[
  "Safety may reduce normalized reward if it becomes highly conservative",
  safeRun["Metrics"]["NormalizedReward"] >= unsafeRun["Metrics"]["NormalizedReward"],
  StringForm["unsafe = `1`, safe = `2`", unsafeRun["Metrics"]["NormalizedReward"], safeRun["Metrics"]["NormalizedReward"]]
];

Print["\n═══ AUDIT 5: Imagination Degradation ═══"];
obsHeavy = SimulateEpisode["MergeIntoTraffic", params, "Steps" -> 16, "ImaginationFraction" -> 0.05, "EnableSafetySupervisor" -> True, "Seed" -> 42];
balanced = SimulateEpisode["MergeIntoTraffic", params, "Steps" -> 16, "ImaginationFraction" -> 0.25, "EnableSafetySupervisor" -> True, "Seed" -> 42];
imagHeavy = SimulateEpisode["MergeIntoTraffic", params, "Steps" -> 16, "ImaginationFraction" -> 0.50, "EnableSafetySupervisor" -> True, "Seed" -> 42];
auditResult[
  "Mean uncertainty rises as imagination fraction rises",
  obsHeavy["Metrics"]["MeanUncertainty"] <= balanced["Metrics"]["MeanUncertainty"] <= imagHeavy["Metrics"]["MeanUncertainty"],
  StringForm["obs-heavy = `1`, balanced = `2`, imag-heavy = `3`", obsHeavy["Metrics"]["MeanUncertainty"], balanced["Metrics"]["MeanUncertainty"], imagHeavy["Metrics"]["MeanUncertainty"]]
];

Print["\n═══ AUDIT 6: Paper Ablation Consistency ═══"];
ablations = BuildAblationDataset[params];
imageScores = Lookup[ablations["ImageResolution"], "DrivingScore"];
deploymentFrequencies = Lookup[ablations["Deployment"], "FrequencyHz"];
auditResult["Image-resolution driving scores are monotone increasing", And @@ Thread[Rest[imageScores] >= Most[imageScores]], StringForm["scores = `1`", imageScores]];
auditResult["Reported deployment frequencies peak in fully recurrent mode", Max[deploymentFrequencies] == deploymentFrequencies[[2]], StringForm["frequencies = `1`", deploymentFrequencies]];

Print["\n═══ AUDIT 7: BEV Decode Shape ═══"];
Module[{bevState, bevGrid},
  bevState = DecodeBevState[
    InitializeLatentState[params],
    MILEProductModel`Private`ScenarioAtTime[BuildScenario["PedestrianConflict", params], 1.2, params],
    params
  ];
  bevGrid = bevState["CombinedGrid"];
  auditResult[
    "BEV decode returns a 2D matrix for ArrayPlot",
    MatrixQ[bevGrid] && Dimensions[bevGrid] == {params["Model"]["BevGridSize"], params["Model"]["BevGridSize"]},
    StringForm["Dimensions = `1`", Dimensions[bevGrid]]
  ];
];

Print["\n╔══════════════════════════════════════════╗"];
Print["║          AUDIT REPORT SUMMARY            ║"];
Print["╠══════════════════════════════════════════╣"];
Print["║  Passed:   ", passed, "                              ║"];
Print["║  Failed:   ", failed, "                              ║"];
Print["║  Warnings: ", warnings, "                              ║"];
Print["╠══════════════════════════════════════════╣"];
If[failed == 0,
  Print["║  OVERALL:  ✓ ALL AUDITS PASSED           ║"],
  Print["║  OVERALL:  ✗ AUDIT FAILURES DETECTED     ║"]
];
Print["╚══════════════════════════════════════════╝"];

Exit[If[failed > 0, 1, 0]]
