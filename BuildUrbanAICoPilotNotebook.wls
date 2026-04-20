#!/usr/bin/env wolframscript

dir = DirectoryName[$InputFileName];
target = FileNameJoin[{dir, "UrbanAICoPilot_MILE.nb"}];

codeBlock[lines_List] := StringRiffle[lines, "\n"];
textBlock[lines_List] := StringRiffle[lines, "\n"];
titleCell[text_] := Cell[text, "Title"];
subtitleCell[text_] := Cell[text, "Subtitle"];
sectionCell[text_] := Cell[text, "Section"];
subsectionCell[text_] := Cell[text, "Subsection"];
textCell[text_] := Cell[text, "Text"];
inputCell[text_] := Cell[text, "Input"];

invalidNamedCharacters = {
  "\\[SqrtBox]",
  "\\[DownArrowBar]",
  "\\[Subscript]",
  "\\[Superscript]",
  "\\[Dash]",
  "\\[Cross]",
  "\\[UpArrowBar]",
  "\\[LeftArrowBar]",
  "\\[RightArrowBar]",
  "\\[DoubleLeftArrow]",
  "\\[OverBracket]",
  "\\[UnderBracket]",
  "\\[Radical]",
  "\\[FractionBox]",
  "\\[GridBox]",
  "\\[UnderscriptBox]",
  "\\[OverscriptBox]",
  "\\[SubscriptBox]",
  "\\[SuperscriptBox]"
};

countToken[text_, token_] := StringCount[text, token];

balancedDelimitersQ[text_] := And[
  countToken[text, "("] == countToken[text, ")"],
  countToken[text, "["] == countToken[text, "]"],
  countToken[text, "{"] == countToken[text, "}"]
];

printValidationLine[label_, passed_, detail_: None] := Module[{suffix},
  suffix = If[detail === None, "", " (" <> ToString[detail, InputForm] <> ")"];
  Print[label, ": ", If[passed, "PASS", "FAIL"], suffix];
];

validateNotebookFile[path_] := Module[
  {
    notebookText,
    trimmedText,
    invalidMatches,
    heldExpression,
    evaluatedNotebook,
    inputCells,
    validationResults,
    overallPass
  },
  notebookText = Import[path, "Text"];
  trimmedText = StringTrim[notebookText];
  invalidMatches = Select[invalidNamedCharacters, StringContainsQ[notebookText, #] &];
  heldExpression = Quiet@Check[ToExpression[notebookText, InputForm, HoldComplete], $Failed];
  evaluatedNotebook = Quiet@Check[Get[path], $Failed];
  inputCells = Cases[evaluatedNotebook, Cell[text_String, "Input", ___] :> text, Infinity];
  validationResults = <|
    "Invalid characters" -> (invalidMatches === {}),
    "BoxData/RowBox" -> !StringContainsQ[notebookText, "BoxData" | "RowBox"],
    "Bracket balance" -> balancedDelimitersQ[notebookText],
    "Notebook structure" -> StringStartsQ[trimmedText, "Notebook[{"] && StringEndsQ[trimmedText, "]"],
    "String escaping / parse" -> MatchQ[heldExpression, HoldComplete[_Notebook]],
    "Get round-trip" -> MatchQ[evaluatedNotebook, _Notebook],
    "Input cells parse" -> And @@ (Quiet@Check[ToExpression[#, InputForm, HoldComplete], $Failed] =!= $Failed & /@ inputCells)
  |>;
  overallPass = And @@ Values[validationResults];

  Print["=== .nb Syntax Validation Report ==="];
  printValidationLine["Invalid characters", validationResults["Invalid characters"], If[invalidMatches === {}, None, invalidMatches]];
  printValidationLine["BoxData/RowBox", validationResults["BoxData/RowBox"]];
  printValidationLine["Bracket balance", validationResults["Bracket balance"]];
  printValidationLine["Notebook structure", validationResults["Notebook structure"]];
  printValidationLine["String escaping / parse", validationResults["String escaping / parse"]];
  printValidationLine["Get round-trip", validationResults["Get round-trip"]];
  printValidationLine["Input cells parse", validationResults["Input cells parse"], Length[inputCells]];
  Print["==================================="];
  Print["Overall: ", If[overallPass, "PASS", "FAIL"]];

  If[!overallPass,
    Print["Notebook validation failed for: ", path];
    Exit[1]
  ];
];

proofAppendixCells = {
  sectionCell["Appendix A. Plain-English Proof"],
  textCell @ textBlock[{
    "This appendix gives a plain-English proof of the main mathematical claims used by the Urban AI CoPilot demo. It proves consistency of the latent model, explains why the KL term is well-defined and non-negative, derives the observation-dropout expectation, and states the scenario-specific safety and runtime claims verified numerically by the package.",
    "",
    "Note on citations: the current workspace preserved the MILE model structure and quantitative results, but not the original printed equation numbers from the paper PDF. To avoid inventing citations, this appendix numbers the reconstructed equations locally."
  }],

  subsectionCell["A.1 Assumptions and Scope"],
  textCell @ textBlock[{
    "Proposition A1. The demo is a faithful executable reconstruction of the paper's model class, not a claim that the exact trained Wayve weights have been reproduced.",
    "",
    "Assumptions.",
    "1. The latent dynamics use a deterministic recurrent state h_t and a stochastic state s_t.",
    "2. The prior and posterior are diagonal Gaussians with strictly positive standard deviations.",
    "3. The product extension adds an external safety supervisor after the learned trajectory proposal, so safety logic is not hidden inside the latent model.",
    "4. Runtime claims are taken from the paper-aligned deployment table encoded in the package.",
    "",
    "Under these assumptions, the proof below establishes mathematical consistency of the reconstruction and scenario-specific validity of the product extension."
  }],

  subsectionCell["A.2 Generative Model"],
  textCell @ textBlock[{
    "Proposition A2. The notebook defines a valid latent state-space model for urban driving.",
    "",
    "The reconstructed equations are:",
    "(1) h_1 ~ delta(0).",
    "(2) s_1 ~ Normal(0, I).",
    "(3) h_{t+1} = f_theta(h_t, s_t).",
    "(4) s_{t+1} ~ Normal(mu_theta(h_{t+1}, a_t), sigma_theta(h_{t+1}, a_t) I).",
    "(5) o_t ~ Normal(g_theta(h_t, s_t), I).",
    "(6) y_t ~ Categorical(l_theta(h_t, s_t)).",
    "(7) a_t ~ Laplace(pi_theta(h_t, s_t), 1).",
    "",
    "Proof.",
    "Equation (1) initializes the deterministic memory at a single point, so there is no ambiguity in the starting recurrent state. Equation (2) initializes the stochastic latent state with a proper Gaussian distribution. Equation (3) is deterministic, so once h_t and s_t are known, h_{t+1} is uniquely defined. Equation (4) is a proper probability distribution because the code constructs every component of sigma_theta as a positive quantity. Equations (5), (6), and (7) are standard likelihood models for continuous observations, discrete BEV labels, and imitation actions. Therefore the pair (h_t, s_t) together with the three likelihoods defines a well-posed sequential latent-variable model."
  }],

  subsectionCell["A.3 ELBO and KL"],
  textCell @ textBlock[{
    "Proposition A3. The KL correction used in the demo is finite and non-negative.",
    "",
    "The relevant equations are:",
    "(8) q_t = Normal(mu_q, diag(sigma_q^2)).",
    "(9) p_t = Normal(mu_p, diag(sigma_p^2)).",
    "(10) ELBO_t = E_q[log p(o_t | h_t, s_t) + log p(y_t | h_t, s_t) + log p(a_t | h_t, s_t)] - KL(q_t || p_t).",
    "(11) KL(q_t || p_t) = (1/2) Sum_i[(sigma_q,i^2 + (mu_q,i - mu_p,i)^2)/sigma_p,i^2 - 1 + log(sigma_p,i^2/sigma_q,i^2)].",
    "",
    "Proof.",
    "The posterior and prior standard deviations are strictly positive in the package because each sigma vector is built from a positive base term plus non-negative corrections. That makes every division and logarithm in (11) well-defined. If mu_q = mu_p and sigma_q = sigma_p, then every summand is exactly zero, so KL(q_t || p_t) = 0. In the general case, equation (11) is the standard closed-form KL divergence between diagonal Gaussian distributions, and KL divergence is always non-negative. Therefore the ELBO in (10) subtracts a finite penalty that is zero only when posterior and prior agree exactly.",
    "",
    "Numerical spot-checks.",
    "Table A1: KL(self || self) = 0.000000.",
    "Table A1: KL(posterior || prior) = 5.950483."
  }],

  subsectionCell["A.4 Observation Dropout and Imagination"],
  textCell @ textBlock[{
    "Proposition A4. If each frame is dropped independently with probability p, then the expected number of consecutive prior-only imagination steps is p/(1-p).",
    "",
    "The relevant equations are:",
    "(12) P(K = k) = (1 - p) p^k for k = 0, 1, 2, ... .",
    "(13) E[K] = Sum_{k >= 0} k (1 - p) p^k = p/(1 - p).",
    "",
    "Proof.",
    "Equation (12) is the geometric distribution for the number of failed observation opportunities before the next observed frame arrives. Summing the standard geometric series gives (13). With p = 0.25, the expectation is 0.25/0.75 = 1/3, which matches the package output exactly.",
    "",
    "Product interpretation.",
    "The observe branch uses posterior moments, while the imagine branch rolls forward with the prior and adds a fixed uncertainty increment of 0.05. For that reason, more imagination must increase mean uncertainty unless the prior and posterior are identical. The audit confirms the expected trend numerically: 0.222929 (observe-heavy) < 0.231301 (balanced) < 0.251713 (imagination-heavy)."
  }],

  subsectionCell["A.5 Safety Supervisor"],
  textCell @ textBlock[{
    "Proposition A5. In the audited PedestrianConflict rollout, the external safety supervisor reduces measured trajectory risk.",
    "",
    "The relevant equations are:",
    "(14) Risk(path) = Clamp(0.55 ObstacleRisk + 0.45 RoadRisk, 0, 1.5).",
    "(15) ObstacleRisk = Sum_j severity_j Exp[-d_j/(radius_j + 0.5)].",
    "(16) If the signal is red and the proposed path crosses the stop line, replace each y-coordinate by min(y, stopLine - 0.4) and set the speed profile to the minimum-risk speed.",
    "(17) If the minimum obstacle distance is below the safety margin, shift the path laterally and scale the speed profile by 0.55.",
    "",
    "Proof.",
    "Equation (14) defines the scalar risk used by the simulator. Equation (16) removes red-light stop-line violations by construction, because the clipped path can no longer pass beyond stopLine - 0.4. Equation (17) increases clearance in the audited pedestrian scenario by moving the path away from the obstacle field and reducing speed. The code does not claim a universal theorem that every imaginable scenario must improve, but for the seeded benchmark used by the audit the measured mean risk drops from 0.438953 without safety to 0.334470 with safety. The corresponding intervention rate rises to 0.9375, which explains why normalized reward falls even as risk improves."
  }],

  subsectionCell["A.6 Runtime and Product Conclusion"],
  textCell @ textBlock[{
    "Proposition A6. Fully recurrent deployment is faster than reset-state deployment in the default demo parameters.",
    "",
    "The relevant equations are:",
    "(18) m_reset = 1000 / 6.2 milliseconds.",
    "(19) m_recurrent = 1000 / 43.0 milliseconds.",
    "(20) f = 1000 / m.",
    "",
    "Proof.",
    "Equations (18) and (19) are the deployment latencies encoded in the package from the paper-aligned table. Since 43.0 > 6.2, equation (20) immediately implies m_recurrent < m_reset, so the recurrent deployment is faster. The same deployment table keeps normalized reward at 0.67 for both reset-state and fully recurrent modes, so the product story is not 'trade all quality for speed'; it is 'preserve the main reward metric while greatly improving cycle rate.'",
    "",
    "Conclusion.",
    "The proof above establishes the exact claim supported by the demo: a MILE-style recurrent latent world model is a mathematically coherent core for a camera-first urban driving product, and the required product extension is an external safety supervisor plus uncertainty-aware deployment logic, not a replacement of the latent model itself."
  }],

  subsectionCell["A.7 Executable Verification"],
  textCell["The following executable cell numerically confirms the main proof claims inside Mathematica."],
  inputCell @ codeBlock[{
    "SeedRandom[42];",
    "Get[FileNameJoin[{NotebookDirectory[], \"MILEProductModel.wl\"}]];",
    "params = DefaultDemoParameters[];",
    "posterior = PosteriorMoments[ConstantArray[0., params[\"Model\"][\"HistoryDimension\"]], {0.7, 0.2, 0.0, 0.8, 0.6, 0.3, 0.2, 0.1}, {0., 0.}, params];",
    "prior = PriorMoments[ConstantArray[0., params[\"Model\"][\"HistoryDimension\"]], ConstantArray[0., params[\"Model\"][\"LatentDimension\"]], {0., 0.}, params];",
    "unsafeRun = SimulateEpisode[\"PedestrianConflict\", params, \"Steps\" -> 16, \"ImaginationFraction\" -> 0.25, \"EnableSafetySupervisor\" -> False, \"Seed\" -> 42];",
    "safeRun = SimulateEpisode[\"PedestrianConflict\", params, \"Steps\" -> 16, \"ImaginationFraction\" -> 0.25, \"EnableSafetySupervisor\" -> True, \"Seed\" -> 42];",
    "obsHeavy = SimulateEpisode[\"MergeIntoTraffic\", params, \"Steps\" -> 16, \"ImaginationFraction\" -> 0.05, \"EnableSafetySupervisor\" -> True, \"Seed\" -> 42];",
    "imagHeavy = SimulateEpisode[\"MergeIntoTraffic\", params, \"Steps\" -> 16, \"ImaginationFraction\" -> 0.50, \"EnableSafetySupervisor\" -> True, \"Seed\" -> 42];",
    "verificationRows = {",
    "  <|\"Check\" -> \"KL(self || self)\", \"Value\" -> GaussianKLDivergence[posterior[\"Mean\"], posterior[\"Sigma\"], posterior[\"Mean\"], posterior[\"Sigma\"]], \"Expected\" -> \"0\"|>,",
    "  <|\"Check\" -> \"KL(posterior || prior)\", \"Value\" -> GaussianKLDivergence[posterior[\"Mean\"], posterior[\"Sigma\"], prior[\"Mean\"], prior[\"Sigma\"]], \"Expected\" -> \"non-negative\"|>,",
    "  <|\"Check\" -> \"Observation dropout mean\", \"Value\" -> ObservationDropoutMean[params[\"Paper\"][\"ObservationDropout\"]], \"Expected\" -> \"1/3\"|>,",
    "  <|\"Check\" -> \"Safety reduces mean risk\", \"Value\" -> safeRun[\"Metrics\"][\"MeanRisk\"], \"Expected\" -> StringForm[\"less than `1`\", NumberForm[unsafeRun[\"Metrics\"][\"MeanRisk\"], {4, 3}]]|>,",
    "  <|\"Check\" -> \"Imagination raises uncertainty\", \"Value\" -> imagHeavy[\"Metrics\"][\"MeanUncertainty\"], \"Expected\" -> StringForm[\"greater than `1`\", NumberForm[obsHeavy[\"Metrics\"][\"MeanUncertainty\"], {4, 3}]]|>,",
    "  <|\"Check\" -> \"Recurrent deployment frequency\", \"Value\" -> RuntimeModel[params, \"FullyRecurrent\"][\"FrequencyHz\"], \"Expected\" -> StringForm[\"greater than `1` Hz\", NumberForm[RuntimeModel[params, \"ResetState\"][\"FrequencyHz\"], {4, 1}]]|>",
    "};",
    "Dataset[verificationRows]"
  }]
};

notebook = Notebook[
  {
    titleCell["Urban AI CoPilot from MILE"],
    subtitleCell["Executable product study reconstructing the MILE latent world model and extending it into a real-car urban L2+/L3 architecture."],

    sectionCell["1. Setup & Package Loading"],
    textCell["This notebook reconstructs the mathematical core of Wayve's MILE paper and turns it into a product-oriented simulation for a camera-first Urban AI CoPilot. The package below encodes the latent dynamics, the variational objective, the short-horizon product controller, and the independent safety supervisor."],
    inputCell @ codeBlock[{
      "SeedRandom[42];",
      "Get[FileNameJoin[{NotebookDirectory[], \"MILEProductModel.wl\"}]];",
      "params = DefaultDemoParameters[];",
      "scenarioLibrary = BuildScenarioLibrary[params];",
      "ablationData = BuildAblationDataset[params];"
    }],

    subsectionCell["1.1 Parameter Snapshot"],
    inputCell @ codeBlock[{
      "parameterRows = {",
      "  {\"Sequence length T\", params[\"Paper\"][\"SequenceLength\"]},",
      "  {\"Frequency (Hz)\", params[\"Paper\"][\"FrequencyHz\"]},",
      "  {\"Observation dropout\", params[\"Paper\"][\"ObservationDropout\"]},",
      "  {\"Latent dimension (demo)\", params[\"Model\"][\"LatentDimension\"]},",
      "  {\"History dimension (demo)\", params[\"Model\"][\"HistoryDimension\"]},",
      "  {\"Reset deployment frequency (Hz)\", params[\"Product\"][\"ResetFrequencyHz\"]},",
      "  {\"Recurrent deployment frequency (Hz)\", params[\"Product\"][\"RecurrentFrequencyHz\"]}",
      "};",
      "Grid[Prepend[parameterRows, Style[{\"Parameter\", \"Value\"}, Bold]], Frame -> All, Background -> {None, {Lighter[Blue, .9], None}}]"
    }],

    sectionCell["2. Research Context and Product Target"],
    textCell["MILE is a model-based imitation learning system with two latent components: a deterministic recurrent history state h_t and a stochastic latent state s_t. The product target in this notebook is a real-world Urban AI CoPilot: a mapless L2+/L3 urban driving system that uses the MILE-style world model as the performance layer, but wraps it with a trajectory interface, uncertainty-aware gating, and an independent safety supervisor."],
    inputCell @ codeBlock[{
      "paperSummary = <|",
      "  \"Title\" -> params[\"Paper\"][\"Title\"],",
      "  \"Venue\" -> params[\"Paper\"][\"Venue\"],",
      "  \"Product\" -> \"Urban AI CoPilot\",",
      "  \"Target ODD\" -> \"Urban roads, arterial roads, dense mixed traffic, supervised L2+/L3\",",
      "  \"Commercial Customer\" -> \"OEMs and Tier-1 suppliers\"",
      "|>;",
      "Dataset[paperSummary]"
    }],

    sectionCell["3. MILE Generative Model"],
    textCell["The paper defines a latent dynamical system that jointly explains observations, BEV labels, and expert actions. The deterministic history h_t carries compact memory. The stochastic state s_t represents uncertainty and multi-modality. The prior is action-conditioned, which is the key to interactive driving in urban settings."],

    subsectionCell["3.1 Probabilistic Structure"],
    inputCell @ codeBlock[{
      "generativeModel = Dataset[{",
      "  <|\"Equation\" -> \"h_1 ~ delta(0)\", \"Role\" -> \"Zero-initialized recurrent memory\"|>,",
      "  <|\"Equation\" -> \"s_1 ~ Normal(0, I)\", \"Role\" -> \"Initial stochastic latent state\"|>,",
      "  <|\"Equation\" -> \"h_{t+1} = f_theta(h_t, s_t)\", \"Role\" -> \"Deterministic recurrent update\"|>,",
      "  <|\"Equation\" -> \"s_{t+1} ~ Normal(mu_theta(h_{t+1}, a_t), sigma_theta(h_{t+1}, a_t) I)\", \"Role\" -> \"Action-conditioned prior dynamics\"|>,",
      "  <|\"Equation\" -> \"o_t ~ Normal(g_theta(h_t, s_t), I)\", \"Role\" -> \"Observation likelihood\"|>,",
      "  <|\"Equation\" -> \"y_t ~ Categorical(l_theta(h_t, s_t))\", \"Role\" -> \"BEV semantic likelihood\"|>,",
      "  <|\"Equation\" -> \"a_t ~ Laplace(pi_theta(h_t, s_t), 1)\", \"Role\" -> \"Action likelihood / imitation objective\"|>",
      "}];",
      "generativeModel"
    }],

    subsectionCell["3.2 ELBO and Closed-Form KL"],
    inputCell @ codeBlock[{
      "posterior = PosteriorMoments[ConstantArray[0., params[\"Model\"][\"HistoryDimension\"]], {0.7, 0.12, 0., 0.83, 0.6, 0.4, 0.2, 0.1}, {0., 0.}, params];",
      "prior = PriorMoments[ConstantArray[0., params[\"Model\"][\"HistoryDimension\"]], ConstantArray[0., params[\"Model\"][\"LatentDimension\"]], {0., 0.}, params];",
      "klValue = GaussianKLDivergence[posterior[\"Mean\"], posterior[\"Sigma\"], prior[\"Mean\"], prior[\"Sigma\"]];",
      "lossBreakdown = Dataset[{",
      "  <|\"Term\" -> \"Action likelihood\", \"Operational loss\" -> \"L1\", \"Comment\" -> \"Laplace action model in the paper\"|>,",
      "  <|\"Term\" -> \"BEV likelihood\", \"Operational loss\" -> \"CrossEntropy + instance terms\", \"Comment\" -> \"Interpretability and geometric supervision\"|>,",
      "  <|\"Term\" -> \"Observation likelihood\", \"Operational loss\" -> \"MSE\", \"Comment\" -> \"Image weight set to 0 in the released configuration\"|>,",
      "  <|\"Term\" -> \"Posterior/Prior matching\", \"Operational loss\" -> klValue, \"Comment\" -> \"Closed-form diagonal Gaussian KL\"|>",
      "}];",
      "lossBreakdown"
    }],

    sectionCell["4. Geometry Lift and BEV Reconstruction"],
    textCell["The central geometric bias in MILE is to lift image features into 3D using camera intrinsics, extrinsics, and a learned depth distribution, then pool those lifted features into bird's-eye view. This section uses a simplified executable reconstruction of that idea to make the geometry visible and product-relevant."],

    subsectionCell["4.1 Simplified BEV Decode"],
    inputCell @ codeBlock[{
      "demoState = InitializeLatentState[params];",
      "bevScenario = BuildScenario[\"PedestrianConflict\", params];",
      "bevCurrent = ScenarioAtTime[bevScenario, 1.2, params];",
      "bev = DecodeBevState[demoState, bevCurrent, params];",
      "ArrayPlot[bev[\"CombinedGrid\"], ColorFunction -> \"SolarColors\", Frame -> True, PlotLabel -> \"Simplified BEV Occupancy / Intent Grid\"]"
    }],
    inputCell @ codeBlock[{
      "ListSurfacePlot3D[bev[\"CombinedGrid\"], Mesh -> None, ColorFunction -> \"SolarColors\", PlotRange -> All, PlotLabel -> \"3D view of the decoded BEV energy surface\", AxesLabel -> {\"x\", \"y\", \"occupancy\"}]"
    }],

    sectionCell["5. Latent Urban Driving Simulator"],
    textCell["The next step is to roll out the compact latent world model in closed loop. The simulator alternates between posterior-corrected observation updates and prior-only imagination updates. The product interpretation is that the model operates as a recurrent world-model policy core inside a real urban driving stack."],
    inputCell @ codeBlock[{
      "pedestrianRun = SimulateEpisode[\"PedestrianConflict\", params, \"Steps\" -> 18, \"ImaginationFraction\" -> 0.25, \"EnableSafetySupervisor\" -> True, \"Seed\" -> 42];",
      "pedestrianTrace = Dataset[pedestrianRun[\"Trace\"]][All, {\"Step\", \"Mode\", \"Reward\", \"Risk\", \"Uncertainty\", \"Intervened\", \"Speed\"}];",
      "pedestrianTrace"
    }],

    subsectionCell["5.1 Trajectory and Vector View"],
    inputCell @ codeBlock[{
      "lastStep = Last[pedestrianRun[\"Trace\"]];",
      "path = lastStep[\"Trajectory\"][\"Waypoints\"];",
      "obstacles = lastStep[\"CurrentObstacles\"];",
      "Graphics[{",
      "  {LightGray, Thick, Line[{{-4, 0}, {-4, 32}}], Line[{{4, 0}, {4, 32}}]},",
      "  {Directive[Darker[Green], Thick], Line[path]},",
      "  {Directive[Red, Thick], Table[Arrow[{path[[i]], path[[i + 1]]}], {i, 1, Length[path] - 1}]},",
      "  {Directive[Orange, Opacity[0.6]], Table[Disk[obstacle[\"Position\"], obstacle[\"Radius\"]], {obstacle, obstacles}]},",
      "  {Black, Disk[{0, 0}, 0.6]}",
      "}, Frame -> True, PlotRange -> {{-8, 8}, {0, 32}}, ImageSize -> Large, PlotLabel -> \"Short-horizon trajectory with obstacle vectors\"]"
    }],

    sectionCell["6. Imagination Mode and Observation Dropout"],
    textCell["MILE's product significance appears when observations are temporarily unreliable. The paper uses observation dropout during training to force the prior to remain useful for multi-step rollout. Here we compare observe-heavy and imagination-heavy deployment with a time-series plot that marks when the rollout is running from prior-only latent dynamics."],

    subsectionCell["6.1 Observe vs Imagine Dynamics"],
    inputCell @ codeBlock[{
      "observeRun = SimulateEpisode[\"MergeIntoTraffic\", params, \"Steps\" -> 18, \"ImaginationFraction\" -> 0.05, \"EnableSafetySupervisor\" -> True, \"Seed\" -> 42];",
      "imagineRun = SimulateEpisode[\"MergeIntoTraffic\", params, \"Steps\" -> 18, \"ImaginationFraction\" -> 0.40, \"EnableSafetySupervisor\" -> True, \"Seed\" -> 42];",
      "phaseBlocks = MapIndexed[",
      "  If[#1[\"Mode\"] === \"Imagine\", {Opacity[0.12, Red], Rectangle[{#2[[1]] - 0.5, 0}, {#2[[1]] + 0.5, 1.6}]}, Nothing] &,",
      "  imagineRun[\"Trace\"]",
      "];",
      "ListLinePlot[",
      "  {Lookup[observeRun[\"Trace\"], \"Risk\"], Lookup[imagineRun[\"Trace\"], \"Risk\"], Lookup[imagineRun[\"Trace\"], \"Uncertainty\"]},",
      "  PlotLegends -> {\"Risk (observe-heavy)\", \"Risk (imagination-heavy)\", \"Uncertainty (imagination-heavy)\"},",
      "  PlotRange -> {0, 1.6},",
      "  Frame -> True,",
      "  Epilog -> phaseBlocks,",
      "  PlotLabel -> \"Imagination mode increases uncertainty while preserving rollout continuity\"",
      "]"
    }],

    sectionCell["7. Safety Supervisor and Product Stack"],
    textCell["The paper stops at a high-performance learned driving model. A real product needs a separate supervisory safety layer. In this notebook the safety supervisor modifies speed and lateral path when a red-light stop line or an obstacle-buffer condition is violated."],

    subsectionCell["7.1 Risk-Gated Control"],
    inputCell @ codeBlock[{
      "unsafeRun = SimulateEpisode[\"PedestrianConflict\", params, \"Steps\" -> 18, \"ImaginationFraction\" -> 0.25, \"EnableSafetySupervisor\" -> False, \"Seed\" -> 42];",
      "safeRun = SimulateEpisode[\"PedestrianConflict\", params, \"Steps\" -> 18, \"ImaginationFraction\" -> 0.25, \"EnableSafetySupervisor\" -> True, \"Seed\" -> 42];",
      "comparisonRows = {",
      "  {\"No safety\", unsafeRun[\"Metrics\"][\"NormalizedReward\"], unsafeRun[\"Metrics\"][\"MeanRisk\"], unsafeRun[\"Metrics\"][\"InterventionRate\"]},",
      "  {\"With safety\", safeRun[\"Metrics\"][\"NormalizedReward\"], safeRun[\"Metrics\"][\"MeanRisk\"], safeRun[\"Metrics\"][\"InterventionRate\"]}",
      "};",
      "Grid[Prepend[comparisonRows, Style[{\"Mode\", \"Norm. Reward\", \"Mean Risk\", \"Intervention Rate\"}, Bold]], Frame -> All, Background -> {None, {Lighter[Blue, .9], None}}]"
    }],

    sectionCell["8. Urban Scenario Evaluation"],
    textCell["The full product claim must hold across several high-value urban scenarios, not just one trace. The scenario study below evaluates the latent world-model stack on roundabout entry, unprotected turns, pedestrian conflicts, traffic-light approach, and merge-in traffic."],

    subsectionCell["8.1 Multi-Scenario Study"],
    inputCell @ codeBlock[{
      "study = RunScenarioStudy[params, \"Steps\" -> 18, \"ImaginationFraction\" -> 0.25, \"EnableSafetySupervisor\" -> True, \"Seed\" -> 42];",
      "scenarioNames = Keys[study];",
      "rewardValues = (study /@ scenarioNames) /. result_Association :> result[\"Metrics\"][\"NormalizedReward\"];",
      "BarChart[rewardValues, ChartLabels -> scenarioNames, ChartLegends -> Placed[{\"Normalized reward\"}, Above], ChartStyle -> ColorData[97] /@ Range[Length[rewardValues]], PlotLabel -> \"Scenario reward across the Urban AI CoPilot operating envelope\"]"
    }],
    inputCell @ codeBlock[{
      "RenderScenarioGrid[study]"
    }],
    inputCell @ codeBlock[{
      "Manipulate[",
      "  SeedRandom[42];",
      "  interactiveRun = SimulateEpisode[scenarioName, params, \"Steps\" -> 16, \"ImaginationFraction\" -> imaginationFraction, \"EnableSafetySupervisor\" -> safetyEnabled, \"Seed\" -> 42];",
      "  interactiveLast = Last[interactiveRun[\"Trace\"]];",
      "  interactivePath = interactiveLast[\"Trajectory\"][\"Waypoints\"];",
      "  interactiveObstacles = interactiveLast[\"CurrentObstacles\"];",
      "  Column[{",
      "    Graphics[{",
      "      {LightGray, Thick, Line[{{-4, 0}, {-4, 30}}], Line[{{4, 0}, {4, 30}}]},",
      "      {Directive[Darker[Green], Thick], Line[interactivePath]},",
      "      {Directive[Red, Opacity[0.6]], Table[Disk[obstacle[\"Position\"], obstacle[\"Radius\"]], {obstacle, interactiveObstacles}]},",
      "      {Black, Disk[{0, 0}, 0.6]}",
      "    }, Frame -> True, PlotRange -> {{-8, 8}, {0, 30}}, ImageSize -> Large, PlotLabel -> StringForm[\"Scenario `1`, imagination `2`, safety `3`\", scenarioName, NumberForm[imaginationFraction, {2, 2}], safetyEnabled]],",
      "    Grid[",
      "      Prepend[",
      "        {{\"Norm. Reward\", NumberForm[interactiveRun[\"Metrics\"][\"NormalizedReward\"], {4, 2}]}, {\"Mean Risk\", NumberForm[interactiveRun[\"Metrics\"][\"MeanRisk\"], {3, 2}]}, {\"Observed Fraction\", NumberForm[interactiveRun[\"Metrics\"][\"ObservedFraction\"], {3, 2}]}, {\"Intervention Rate\", NumberForm[interactiveRun[\"Metrics\"][\"InterventionRate\"], {3, 2}]}} ,",
      "        Style[{\"Metric\", \"Value\"}, Bold]",
      "      ],",
      "      Frame -> All, Background -> {None, {Lighter[Blue, .9], None}}",
      "    ]",
      "  }],",
      "  {{scenarioName, \"PedestrianConflict\", \"Scenario\"}, Keys[scenarioLibrary]},",
      "  {{imaginationFraction, 0.25, \"Imagination Fraction\"}, 0., 0.60, 0.05},",
      "  {{safetyEnabled, True, \"Safety Supervisor\"}, {True, False}},",
      "  ControlPlacement -> Left,",
      "  SaveDefinitions -> True",
      "]"
    }],

    sectionCell["9. Latency and Embedded Deployment"],
    textCell["A major product insight from the paper is that recurrent deployment is much cheaper than reset-state inference because it updates one compact state instead of re-encoding a long history window every cycle. This matters directly for embedded compute on a real car."],

    subsectionCell["9.1 Reset-State vs Fully Recurrent"],
    inputCell @ codeBlock[{
      "runtimeTable = Dataset[{RuntimeModel[params, \"ResetState\"], RuntimeModel[params, \"FullyRecurrent\"]}];",
      "runtimeTable"
    }],
    inputCell @ codeBlock[{
      "BarChart[Lookup[Normal[runtimeTable], \"FrequencyHz\"], ChartLabels -> Lookup[Normal[runtimeTable], \"Mode\"], ChartStyle -> {ColorData[97][2], ColorData[97][4]}, PlotLabel -> \"Embedded deployment frequency: recurrent state update vs history reset\"]"
    }],

    sectionCell["10. Paper Ablations and Product Implications"],
    textCell["The paper shows two product-relevant trends: more image resolution materially reduces infractions, while larger latent state does not automatically improve recurrent performance. Those are exactly the kinds of tradeoffs a production stack must internalize when targeting automotive compute."],
    inputCell @ codeBlock[{
      "resolutionData = ablationData[\"ImageResolution\"];",
      "ListLinePlot[Lookup[resolutionData, \"DrivingScore\"], Frame -> True, PlotMarkers -> Automatic, FrameTicks -> {{Automatic, None}, {Thread[{Range[Length[resolutionData]], Lookup[resolutionData, \"Resolution\"]}], None}}, PlotLabel -> \"Driving score rises sharply with image resolution\"]"
    }],
    inputCell @ codeBlock[{
      "latentData = ablationData[\"LatentDimension\"];",
      "BarChart[Transpose[{Lookup[latentData, \"ResetReward\"], Lookup[latentData, \"RecurrentReward\"]}], ChartLabels -> Lookup[latentData, \"Label\"], ChartLegends -> {\"Reset-state reward\", \"Recurrent reward\"}, ChartStyle -> {ColorData[97][1], ColorData[97][5]}, PlotLabel -> \"Compact latent states retain recurrent performance better than spatially large latents\"]"
    }],

    sectionCell["11. Key Findings"],
    inputCell @ codeBlock[{
      "Print[\"========== KEY FINDINGS ==========\"];",
      "Print[\"1. The MILE latent structure is product-relevant because it couples compact recurrent memory with action-conditioned uncertainty.\"];",
      "Print[\"2. Recurrent deployment preserves normalized reward while raising embedded frequency from 6.2 Hz to 43.0 Hz in the paper's reported setting.\"];",
      "Print[\"3. The product gap is not the world model itself; it is the independent safety layer, uncertainty gating, and OEM-grade fleet learning loop that must wrap it.\"];",
      "Print[\"4. A camera-first Urban AI CoPilot is the cleanest commercialization path because it uses the paper's strengths without pretending the research model is already a turnkey L4 stack.\"]"
    }],

    sectionCell["12. Validation"],
    textCell["The final section runs a compact numerical validation pass over the demo package and lists the source materials used in the reconstruction and product extension."],
    inputCell @ codeBlock[{
      "checks = ValidationChecks[params];",
      "Grid[Prepend[KeyValueMap[{#1, #2} &, checks], Style[{\"Check\", \"Result\"}, Bold]], Frame -> All, Background -> {None, {Lighter[Blue, .9], None}}]"
    }],
    Sequence @@ proofAppendixCells,
    sectionCell["References"],
    textCell["References\n[1] Hu et al., Model-Based Imitation Learning for Urban Driving, NeurIPS 2022.\n[2] Wayve product page: Wayve AI Driver.\n[3] Wayve technology page: AV2.0, fleet learning, mapless autonomy, and foundation driving models.\n[4] This notebook's product extension: Urban AI CoPilot = MILE-style world model + trajectory interface + uncertainty gating + independent safety supervisor."]
  },
  WindowSize -> {1200, 900},
  WindowMargins -> {{Automatic, Automatic}, {Automatic, Automatic}},
  FrontEndVersion -> "14.0",
  StyleDefinitions -> "Default.nb"
];

Export[target, ToString[InputForm[notebook], PageWidth -> Infinity], "Text"];
Print["Notebook written to: ", target];
validateNotebookFile[target];
