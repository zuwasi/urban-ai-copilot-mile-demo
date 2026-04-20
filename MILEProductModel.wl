BeginPackage["MILEProductModel`"]

DefaultDemoParameters::usage = "DefaultDemoParameters[] returns the nested Association of paper, model, and product parameters for the Urban AI CoPilot demo.";
BuildScenarioLibrary::usage = "BuildScenarioLibrary[params] returns the Association of urban driving scenarios used in the demo.";
BuildScenario::usage = "BuildScenario[name, params] returns one scenario Association by name.";
InitializeLatentState::usage = "InitializeLatentState[params] creates the deterministic history, stochastic latent state, and vehicle state used at rollout start.";
ObservationDropoutMean::usage = "ObservationDropoutMean[p] returns the expected number of consecutive prior-only unrolls under a geometric observation-dropout process.";
GaussianKLDivergence::usage = "GaussianKLDivergence[muQ, sigmaQ, muP, sigmaP] computes the closed-form KL divergence for diagonal Gaussian posteriors and priors.";
ScenarioObservationVector::usage = "ScenarioObservationVector[scenario, state, step, params] returns an Association containing a compact observation vector and the current obstacle set.";
PosteriorMoments::usage = "PosteriorMoments[history, observation, previousAction, params] returns the posterior Gaussian parameters for the stochastic latent state.";
PriorMoments::usage = "PriorMoments[history, latent, previousAction, params] returns the action-conditioned prior Gaussian parameters for the stochastic latent state.";
SampleLatentState::usage = "SampleLatentState[mean, sigma, seed] draws one latent-state sample using the reparameterization-style Normal sample.";
UpdateDeterministicState::usage = "UpdateDeterministicState[history, latent, params] updates the deterministic recurrent history state.";
PredictAction::usage = "PredictAction[state, scenario, params] predicts acceleration and steering commands for the current scenario state.";
PredictTrajectory::usage = "PredictTrajectory[state, action, scenario, params] produces a waypoint path and speed profile for the short-horizon product controller.";
MeasureTrajectoryRisk::usage = "MeasureTrajectoryRisk[trajectory, obstacles, params] evaluates obstacle proximity and lane-boundary risk for a candidate trajectory.";
SafetySupervisor::usage = "SafetySupervisor[trajectory, scenario, params] applies an independent supervisory safety layer to a candidate trajectory.";
DecodeBevState::usage = "DecodeBevState[state, scenario, params] returns a simplified BEV occupancy representation that can be visualized in the notebook.";
RuntimeModel::usage = "RuntimeModel[params, mode] estimates the embedded runtime of reset-state and recurrent deployment modes.";
BuildAblationDataset::usage = "BuildAblationDataset[params] returns paper-aligned ablation tables for image resolution, latent state dimension, and deployment mode.";
SimulateEpisode::usage = "SimulateEpisode[scenario, params, opts] rolls out the MILE-style latent product model for one scenario and returns traces and summary metrics.";
RunScenarioStudy::usage = "RunScenarioStudy[params, opts] evaluates all scenarios in the library and returns an Association of simulation outputs.";
RenderScenarioGrid::usage = "RenderScenarioGrid[study] creates a styled Grid summarizing scenario metrics.";
ValidationChecks::usage = "ValidationChecks[params] runs a compact suite of numerical sanity checks over the demo model.";

Begin["`Private`"]

ClearAll[
  ClampValue,
  VectorPad,
  CurrentSignalState,
  ScenarioAtTime,
  ActionVectorPad,
  NumericMean,
  MakeRows,
  MonotoneIncreasingQ
]

ClampValue[x_, {xmin_, xmax_}] := N @ Clip[x, {xmin, xmax}]

VectorPad[list_List, n_Integer] := PadRight[N[list], n, 0.]

ActionVectorPad[action_List, n_Integer] := VectorPad[action, n]

NumericMean[list_List] := If[list === {}, 0., N @ Mean[list]]

MonotoneIncreasingQ[list_List] := And @@ Thread[Rest[list] >= Most[list]]

DefaultDemoParameters[] := <|
  "Paper" -> <|
    "Title" -> "Model-Based Imitation Learning for Urban Driving",
    "Venue" -> "NeurIPS 2022",
    "SequenceLength" -> 12,
    "FrequencyHz" -> 5.,
    "ObservationDropout" -> 0.25,
    "CameraResolution" -> {600, 960}
  |>,
  "Model" -> <|
    "LatentDimension" -> 8,
    "HistoryDimension" -> 16,
    "ActionDimension" -> 2,
    "BevGridSize" -> 48,
    "LaneWidthMeters" -> 3.6,
    "ForwardRangeMeters" -> 60.,
    "LateralRangeMeters" -> 12.,
    "PlanningHorizonSeconds" -> 3.0,
    "PlanningStepSeconds" -> 0.3,
    "ImageWeight" -> 0.,
    "SegmentationWeight" -> 0.1,
    "KLWeight" -> 0.001
  |>,
  "Product" -> <|
    "ResetFrequencyHz" -> 6.2,
    "RecurrentFrequencyHz" -> 43.0,
    "SafetyMarginMeters" -> 4.5,
    "RiskThreshold" -> 0.38,
    "MaxSpeedMS" -> 14.0,
    "MinimumRiskSpeedMS" -> 0.5,
    "StopLineMeters" -> 18.0,
    "CameraCount" -> 6,
    "RadarEnabled" -> True
  |>
|>

BuildScenarioLibrary[params_: DefaultDemoParameters[]] := <|
  "TrafficLightStop" -> <|
    "Name" -> "TrafficLightStop",
    "Maneuver" -> "Urban stop-line approach with cross traffic.",
    "RouteCurvature" -> 0.02,
    "TargetSpeed" -> 9.0,
    "BaseSignalState" -> "Red",
    "LaneOffset" -> 0.,
    "Visibility" -> 0.95,
    "Complexity" -> 0.35,
    "Obstacles" -> {
      <|"Label" -> "CrossTraffic", "Position" -> {0.5, 24.0}, "Velocity" -> {0.3, 0.0}, "Radius" -> 2.0, "Severity" -> 0.70|>
    }
  |>,
  "UnprotectedTurn" -> <|
    "Name" -> "UnprotectedTurn",
    "Maneuver" -> "Right turn across an active mixed-agent junction.",
    "RouteCurvature" -> 0.22,
    "TargetSpeed" -> 7.5,
    "BaseSignalState" -> "None",
    "LaneOffset" -> -0.2,
    "Visibility" -> 0.88,
    "Complexity" -> 0.62,
    "Obstacles" -> {
      <|"Label" -> "OncomingVehicle", "Position" -> {1.5, 19.0}, "Velocity" -> {-0.2, -0.4}, "Radius" -> 2.1, "Severity" -> 0.86|>,
      <|"Label" -> "Cyclist", "Position" -> {-2.0, 13.0}, "Velocity" -> {0.1, 0.4}, "Radius" -> 1.2, "Severity" -> 0.58|>
    }
  |>,
  "RoundaboutEntry" -> <|
    "Name" -> "RoundaboutEntry",
    "Maneuver" -> "Gap acceptance at a compact urban roundabout.",
    "RouteCurvature" -> 0.30,
    "TargetSpeed" -> 8.5,
    "BaseSignalState" -> "Yield",
    "LaneOffset" -> 0.1,
    "Visibility" -> 0.82,
    "Complexity" -> 0.73,
    "Obstacles" -> {
      <|"Label" -> "RoundaboutFlowA", "Position" -> {2.0, 17.0}, "Velocity" -> {-0.4, 0.0}, "Radius" -> 2.0, "Severity" -> 0.74|>,
      <|"Label" -> "RoundaboutFlowB", "Position" -> {-2.5, 21.0}, "Velocity" -> {0.3, 0.0}, "Radius" -> 2.0, "Severity" -> 0.71|>
    }
  |>,
  "PedestrianConflict" -> <|
    "Name" -> "PedestrianConflict",
    "Maneuver" -> "Urban approach with a pedestrian stepping into the lane.",
    "RouteCurvature" -> 0.00,
    "TargetSpeed" -> 10.5,
    "BaseSignalState" -> "None",
    "LaneOffset" -> 0.,
    "Visibility" -> 0.78,
    "Complexity" -> 0.81,
    "Obstacles" -> {
      <|"Label" -> "Pedestrian", "Position" -> {0.4, 12.0}, "Velocity" -> {-0.05, 0.15}, "Radius" -> 0.9, "Severity" -> 1.00|>,
      <|"Label" -> "ParkedVan", "Position" -> {-2.3, 15.0}, "Velocity" -> {0.0, 0.0}, "Radius" -> 2.2, "Severity" -> 0.35|>
    }
  |>,
  "MergeIntoTraffic" -> <|
    "Name" -> "MergeIntoTraffic",
    "Maneuver" -> "High-density arterial merge with one opening gap.",
    "RouteCurvature" -> 0.10,
    "TargetSpeed" -> 11.5,
    "BaseSignalState" -> "None",
    "LaneOffset" -> -0.4,
    "Visibility" -> 0.86,
    "Complexity" -> 0.69,
    "Obstacles" -> {
      <|"Label" -> "LeadVehicle", "Position" -> {0.8, 18.0}, "Velocity" -> {0.0, 0.2}, "Radius" -> 2.1, "Severity" -> 0.64|>,
      <|"Label" -> "AdjacentFlow", "Position" -> {2.6, 16.5}, "Velocity" -> {0.0, 0.45}, "Radius" -> 2.0, "Severity" -> 0.76|>
    }
  |>
|>

BuildScenario[name_String, params_: DefaultDemoParameters[]] := BuildScenarioLibrary[params][name]
BuildScenario[scenario_Association, _ : DefaultDemoParameters[]] := scenario

InitializeLatentState[params_: DefaultDemoParameters[]] := Module[
  {
    historyDimension = params["Model"]["HistoryDimension"],
    latentDimension = params["Model"]["LatentDimension"]
  },
  <|
    "History" -> ConstantArray[0., historyDimension],
    "Latent" -> ConstantArray[0., latentDimension],
    "Speed" -> 0.,
    "PreviousAction" -> {0., 0.},
    "TimeSeconds" -> 0.
  |>
]

ObservationDropoutMean[p_?NumericQ] := If[p >= 1., Infinity, N[p/(1. - p)]]

CurrentSignalState[name_String, time_?NumericQ] := Switch[
  name,
  "TrafficLightStop", If[time < 2.1, "Red", "Green"],
  "RoundaboutEntry", If[time < 1.5, "Yield", "Clear"],
  _, "None"
]

ScenarioAtTime[scenario_Association, time_?NumericQ, params_: DefaultDemoParameters[]] := Module[
  {currentObstacles},
  currentObstacles = Map[
    Association[
      #,
      "Position" -> N[# ["Position"] + time # ["Velocity"]],
      "Distance" -> N @ Norm[# ["Position"] + time # ["Velocity"]]
    ] &,
    Lookup[scenario, "Obstacles", {}]
  ];
  Association[
    scenario,
    "CurrentTime" -> N[time],
    "SignalState" -> CurrentSignalState[scenario["Name"], time],
    "CurrentObstacles" -> currentObstacles
  ]
]

ScenarioObservationVector[scenario_Association, state_Association, step_Integer, params_: DefaultDemoParameters[]] := Module[
  {
    obstacles = Lookup[scenario, "CurrentObstacles", {}],
    maxSpeed = params["Product"]["MaxSpeedMS"],
    forwardRange = params["Model"]["ForwardRangeMeters"],
    laneWidth = params["Model"]["LaneWidthMeters"],
    nearestDistance,
    meanSeverity,
    meanLongitudinalVelocity,
    signalIndicator,
    vector
  },
  nearestDistance = If[obstacles === {}, forwardRange, Min[Lookup[obstacles, "Distance", forwardRange]]];
  meanSeverity = NumericMean @ Lookup[obstacles, "Severity", {}];
  meanLongitudinalVelocity = NumericMean @ (Lookup[obstacles, "Velocity", {}][[All, 2]] /. {} -> {0.});
  signalIndicator = Switch[scenario["SignalState"], "Red", 1., "Yield", 0.5, _, 0.];
  vector = N @ {
    scenario["TargetSpeed"]/maxSpeed,
    scenario["RouteCurvature"],
    signalIndicator,
    scenario["Visibility"],
    scenario["Complexity"],
    nearestDistance/forwardRange,
    meanSeverity,
    meanLongitudinalVelocity/maxSpeed,
    state["Speed"]/maxSpeed,
    scenario["LaneOffset"]/laneWidth,
    Sin[0.1 step],
    Cos[0.1 step]
  };
  <|"Vector" -> vector, "CurrentObstacles" -> obstacles|>
]

PosteriorMoments[history_List, observation_List, previousAction_List, params_: DefaultDemoParameters[]] := Module[
  {
    latentDimension = params["Model"]["LatentDimension"],
    historySlice,
    actionSlice,
    basis,
    mean,
    sigma
  },
  historySlice = VectorPad[Take[history, UpTo[latentDimension]], latentDimension];
  actionSlice = ActionVectorPad[previousAction, latentDimension];
  basis = VectorPad[observation, latentDimension];
  mean = N @ Tanh[0.55 basis + 0.25 historySlice + 0.15 actionSlice + 0.05 Range[latentDimension]/latentDimension];
  sigma = N @ (0.18 + 0.08 Abs[basis - 0.5 historySlice] + 0.02 Range[latentDimension]/latentDimension);
  <|"Mean" -> mean, "Sigma" -> sigma|>
]

PriorMoments[history_List, latent_List, previousAction_List, params_: DefaultDemoParameters[]] := Module[
  {
    latentDimension = params["Model"]["LatentDimension"],
    historySlice,
    actionSlice,
    latentSlice,
    mean,
    sigma
  },
  historySlice = VectorPad[Take[history, UpTo[latentDimension]], latentDimension];
  latentSlice = VectorPad[latent, latentDimension];
  actionSlice = ActionVectorPad[previousAction, latentDimension];
  mean = N @ Tanh[0.60 historySlice + 0.35 latentSlice + 0.18 actionSlice - 0.03 Range[latentDimension]/latentDimension];
  sigma = N @ (0.22 + 0.05 Abs[latentSlice - historySlice] + 0.01 Range[latentDimension]/latentDimension);
  <|"Mean" -> mean, "Sigma" -> sigma|>
]

GaussianKLDivergence[muQ_List, sigmaQ_List, muP_List, sigmaP_List] := Module[
  {varQ, varP},
  varQ = N[sigmaQ^2];
  varP = N[sigmaP^2];
  N @ (0.5 Total[(varQ + (muQ - muP)^2)/varP - 1. + Log[varP/varQ]])
]

SampleLatentState[mean_List, sigma_List, seed_: Automatic] := Module[{noise},
  If[seed =!= Automatic, SeedRandom[seed]];
  noise = RandomVariate[NormalDistribution[0., 1.], Length[mean]];
  N[mean + sigma noise]
]

UpdateDeterministicState[history_List, latent_List, params_: DefaultDemoParameters[]] := Module[
  {
    historyDimension = params["Model"]["HistoryDimension"],
    latentDimension = params["Model"]["LatentDimension"],
    historyVector,
    latentVector
  },
  historyVector = VectorPad[history, historyDimension];
  latentVector = VectorPad[latent, latentDimension];
  N @ Table[
    Tanh[
      0.68 historyVector[[i]]
      + 0.26 latentVector[[1 + Mod[i - 1, latentDimension]]]
      + 0.12 historyVector[[Max[i - 1, 1]]]
      + 0.04 Mean[latentVector]
    ],
    {i, historyDimension}
  ]
]

PredictAction[state_Association, scenario_Association, params_: DefaultDemoParameters[]] := Module[
  {
    obstacles = Lookup[scenario, "CurrentObstacles", {}],
    maxSpeed = params["Product"]["MaxSpeedMS"],
    laneWidth = params["Model"]["LaneWidthMeters"],
    latentInfluence,
    interactionPenalty,
    signalPenalty,
    nearestLateralBias,
    acceleration,
    steering
  },
  latentInfluence = NumericMean @ Take[state["Latent"], UpTo[3]];
  interactionPenalty = Total[Lookup[obstacles, "Severity", {}] Exp[-Lookup[obstacles, "Distance", {}]/12.]];
  signalPenalty = Switch[scenario["SignalState"], "Red", 0.82, "Yield", 0.35, _, 0.];
  nearestLateralBias = If[
    obstacles === {},
    0.,
    NumericMean[(Lookup[obstacles, "Position"][[All, 1]])/(1. + Lookup[obstacles, "Distance"])]
  ];
  steering = ClampValue[
    0.90 scenario["RouteCurvature"]
    + 0.16 latentInfluence
    - 0.14 scenario["LaneOffset"]/laneWidth
    - 0.18 nearestLateralBias,
    {-1., 1.}
  ];
  acceleration = ClampValue[
    scenario["TargetSpeed"]/maxSpeed
    - 0.40 interactionPenalty
    - signalPenalty
    - 0.10 state["Speed"]/maxSpeed,
    {-1., 1.}
  ];
  <|
    "Acceleration" -> acceleration,
    "Steering" -> steering,
    "RawAction" -> {acceleration, steering},
    "InteractionPenalty" -> N[interactionPenalty],
    "SignalPenalty" -> N[signalPenalty]
  |>
]

PredictTrajectory[state_Association, action_Association, scenario_Association, params_: DefaultDemoParameters[]] := Module[
  {
    dt = params["Model"]["PlanningStepSeconds"],
    horizon = params["Model"]["PlanningHorizonSeconds"],
    steps,
    maxSpeed = params["Product"]["MaxSpeedMS"],
    speedProfile,
    curvature,
    positions,
    path
  },
  steps = Round[horizon/dt];
  speedProfile = N @ Table[
    ClampValue[state["Speed"] + i dt 2.1 action["Acceleration"], {0., maxSpeed}],
    {i, 0, steps}
  ];
  curvature = N[0.08 action["Steering"] + scenario["RouteCurvature"]];
  positions = Rest @ FoldList[
    Function[{acc, speed},
      Module[{heading},
        heading = acc[[3]] + curvature dt;
        {
          acc[[1]] + speed dt Sin[heading],
          acc[[2]] + speed dt Cos[heading],
          heading
        }
      ]
    ],
    {scenario["LaneOffset"], 0., 0.},
    Rest[speedProfile]
  ];
  path = Join[{{scenario["LaneOffset"], 0.}}, positions[[All, {1, 2}]]];
  <|
    "Waypoints" -> N[path],
    "SpeedProfile" -> N[speedProfile],
    "Curvature" -> curvature,
    "Command" -> action["RawAction"]
  |>
]

MeasureTrajectoryRisk[trajectory_Association, obstacles_List, params_: DefaultDemoParameters[]] := Module[
  {
    path = trajectory["Waypoints"],
    laneWidth = params["Model"]["LaneWidthMeters"],
    minDistances,
    obstacleRisk,
    roadRisk,
    minimumDistance
  },
  If[
    obstacles === {},
    minDistances = {params["Model"]["ForwardRangeMeters"]};
    obstacleRisk = 0.,
    minDistances = Table[
      Min[EuclideanDistance[#, obstacle["Position"]] & /@ path],
      {obstacle, obstacles}
    ];
    obstacleRisk = Total @ MapThread[
      #2["Severity"] Exp[-#1/(#2["Radius"] + 0.5)] &,
      {minDistances, obstacles}
    ];
  ];
  minimumDistance = Min[minDistances];
  roadRisk = N @ Mean[Boole /@ Thread[Abs[path[[All, 1]]] > 0.70 laneWidth]];
  <|
    "MinimumDistance" -> N[minimumDistance],
    "ObstacleRisk" -> N[obstacleRisk],
    "RoadRisk" -> roadRisk,
    "Risk" -> ClampValue[0.55 obstacleRisk + 0.45 roadRisk, {0., 1.5}]
  |>
]

SafetySupervisor[trajectory_Association, scenario_Association, params_: DefaultDemoParameters[]] := Module[
  {
    safeTrajectory = trajectory,
    obstacles = Lookup[scenario, "CurrentObstacles", {}],
    originalRisk,
    safetyMargin = params["Product"]["SafetyMarginMeters"],
    minimumRiskSpeed = params["Product"]["MinimumRiskSpeedMS"],
    stopLine = params["Product"]["StopLineMeters"],
    reasons = {},
    intervene = False,
    shift
  },
  originalRisk = MeasureTrajectoryRisk[trajectory, obstacles, params];
  If[
    scenario["SignalState"] === "Red" && Max[safeTrajectory["Waypoints"][[All, 2]]] > stopLine,
    safeTrajectory = Association[
      safeTrajectory,
      "Waypoints" -> ({#[[1]], Min[#[[2]], stopLine - 0.4]} &) /@ safeTrajectory["Waypoints"],
      "SpeedProfile" -> ConstantArray[minimumRiskSpeed, Length[safeTrajectory["SpeedProfile"]]],
      "Command" -> {-0.9, Last[safeTrajectory["Command"]]}
    ];
    reasons = Append[reasons, "red-light-stop"];
    intervene = True;
  ];
  If[
    originalRisk["MinimumDistance"] < safetyMargin,
    shift = If[obstacles === {}, -0.8, Sign[Total[Lookup[obstacles, "Position"][[All, 1]]]]];
    If[shift == 0., shift = -0.8];
    safeTrajectory = Association[
      safeTrajectory,
      "Waypoints" -> ({ClampValue[#[[1]] - 0.7 shift, {-1.2 params["Model"]["LaneWidthMeters"], 1.2 params["Model"]["LaneWidthMeters"]}], #[[2]]} &) /@ safeTrajectory["Waypoints"],
      "SpeedProfile" -> N[0.55 safeTrajectory["SpeedProfile"]],
      "Command" -> {-0.6, 0.6 Last[safeTrajectory["Command"]]}
    ];
    reasons = Append[reasons, "obstacle-buffer"];
    intervene = True;
  ];
  <|
    "Trajectory" -> safeTrajectory,
    "Intervened" -> intervene,
    "Reasons" -> reasons,
    "OriginalRisk" -> originalRisk,
    "SafeRisk" -> MeasureTrajectoryRisk[safeTrajectory, obstacles, params]
  |>
]

DecodeBevState[state_Association, scenario_Association, params_: DefaultDemoParameters[]] := Module[
  {
    gridSize = params["Model"]["BevGridSize"],
    forwardRange = params["Model"]["ForwardRangeMeters"],
    lateralRange = params["Model"]["LateralRangeMeters"],
    laneWidth = params["Model"]["LaneWidthMeters"],
    xs,
    ys,
    obstacles = Lookup[scenario, "CurrentObstacles", {}],
    nominalAction,
    nominalTrajectory,
    roadGrid,
    obstacleGrid,
    trajectoryGrid,
    combinedGrid
  },
  nominalAction = PredictAction[state, scenario, params];
  nominalTrajectory = PredictTrajectory[state, nominalAction, scenario, params];
  xs = N @ Subdivide[-lateralRange, lateralRange, gridSize - 1];
  ys = N @ Subdivide[0., forwardRange, gridSize - 1];
  roadGrid = Table[
    Exp[-(x - scenario["LaneOffset"])^2/(2. (laneWidth/2.)^2)],
    {y, ys}, {x, xs}
  ];
  obstacleGrid = Table[
    Total[
      # ["Severity"] Exp[-Norm[{x, y} - # ["Position"]]^2/(2.4^2)] & /@ obstacles
    ],
    {y, ys}, {x, xs}
  ];
  trajectoryGrid = Table[
    Total[Exp[-Norm[{x, y} - #]^2/(3.2^2)] & /@ nominalTrajectory["Waypoints"]]/Max[1, Length[nominalTrajectory["Waypoints"]]],
    {y, ys}, {x, xs}
  ];
  combinedGrid = N[0.55 roadGrid + 0.80 trajectoryGrid + 1.10 obstacleGrid];
  <|
    "XCoordinates" -> xs,
    "YCoordinates" -> ys,
    "RoadGrid" -> N[roadGrid],
    "ObstacleGrid" -> N[obstacleGrid],
    "TrajectoryGrid" -> N[trajectoryGrid],
    "CombinedGrid" -> combinedGrid,
    "Trajectory" -> nominalTrajectory
  |>
]

RuntimeModel[params_: DefaultDemoParameters[], mode_String : "ResetState"] := Module[
  {
    sequenceLength = params["Paper"]["SequenceLength"],
    baseSequence = params["Paper"]["SequenceLength"],
    cameraCount = params["Product"]["CameraCount"],
    cameraScale,
    baseMilliseconds,
    milliseconds
  },
  cameraScale = cameraCount/6.;
  baseMilliseconds = Switch[
    mode,
    "ResetState", 1000./params["Product"]["ResetFrequencyHz"],
    "FullyRecurrent", 1000./params["Product"]["RecurrentFrequencyHz"],
    _, 1000./params["Product"]["RecurrentFrequencyHz"]
  ];
  milliseconds = N @ Which[
    mode === "ResetState", baseMilliseconds cameraScale (sequenceLength/baseSequence),
    True, baseMilliseconds cameraScale
  ];
  <|
    "Mode" -> mode,
    "Milliseconds" -> milliseconds,
    "FrequencyHz" -> N[1000./milliseconds],
    "CameraCount" -> cameraCount,
    "SequenceLength" -> sequenceLength
  |>
]

BuildAblationDataset[params_: DefaultDemoParameters[]] := <|
  "ImageResolution" -> {
    <|"Resolution" -> "75x120", "DrivingScore" -> 20.9, "NormalizedReward" -> 0.65|>,
    <|"Resolution" -> "150x240", "DrivingScore" -> 27.9, "NormalizedReward" -> 0.65|>,
    <|"Resolution" -> "300x480", "DrivingScore" -> 43.3, "NormalizedReward" -> 0.55|>,
    <|"Resolution" -> "600x960", "DrivingScore" -> 61.1, "NormalizedReward" -> 0.67|>
  },
  "Deployment" -> {
    <|"Mode" -> "ResetState", "DrivingScore" -> 61.1, "RouteCompletion" -> 97.4, "NormalizedReward" -> 0.67, "FrequencyHz" -> 6.2|>,
    <|"Mode" -> "FullyRecurrent", "DrivingScore" -> 62.1, "RouteCompletion" -> 93.5, "NormalizedReward" -> 0.67, "FrequencyHz" -> 43.0|>,
    <|"Mode" -> "RecurrentWithNoise", "DrivingScore" -> 48.8, "RouteCompletion" -> 81.1, "NormalizedReward" -> 0.35, "FrequencyHz" -> 43.0|>
  },
  "LatentDimension" -> {
    <|"Label" -> "512x1x1", "ResetReward" -> 7621., "RecurrentReward" -> 7532.|>,
    <|"Label" -> "256x12x12", "ResetReward" -> 7465., "RecurrentReward" -> 6998.|>,
    <|"Label" -> "128x24x24", "ResetReward" -> 6407., "RecurrentReward" -> 4596.|>,
    <|"Label" -> "64x48x48", "ResetReward" -> 5637., "RecurrentReward" -> 3794.|>
  }
|>

Options[SimulateEpisode] = {
  "Steps" -> 18,
  "ImaginationFraction" -> 0.25,
  "Seed" -> 42,
  "EnableSafetySupervisor" -> True
}

SimulateEpisode[scenarioInput_, params_: DefaultDemoParameters[], OptionsPattern[]] := Module[
  {
    scenario = BuildScenario[scenarioInput, params],
    steps = OptionValue["Steps"],
    imaginationFraction = N @ OptionValue["ImaginationFraction"],
    seed = OptionValue["Seed"],
    safetyEnabled = TrueQ[OptionValue["EnableSafetySupervisor"]],
    state,
    trace = {},
    observedCount = 0,
    t,
    time,
    currentScenario,
    observation,
    posterior,
    prior,
    latent,
    history,
    action,
    trajectory,
    safetyResult,
    risk,
    uncertainty,
    reward,
    stepProgress,
    routeProgress = 0.,
    kl,
    mode,
    metrics,
    currentBev,
    meanKLObserved
  },
  SeedRandom[seed];
  state = InitializeLatentState[params];
  Do[
    time = N[(t - 1)/params["Paper"]["FrequencyHz"]];
    currentScenario = ScenarioAtTime[scenario, time, params];
    observation = ScenarioObservationVector[currentScenario, state, t, params];
    prior = PriorMoments[state["History"], state["Latent"], state["PreviousAction"], params];
    mode = If[RandomReal[] < imaginationFraction, "Imagine", "Observe"];
    If[
      mode === "Observe",
      observedCount++;
      posterior = PosteriorMoments[state["History"], observation["Vector"], state["PreviousAction"], params];
      latent = SampleLatentState[posterior["Mean"], posterior["Sigma"]];
      kl = GaussianKLDivergence[posterior["Mean"], posterior["Sigma"], prior["Mean"], prior["Sigma"]];
      uncertainty = NumericMean[posterior["Sigma"]],
      posterior = <|"Mean" -> prior["Mean"], "Sigma" -> prior["Sigma"]|>;
      latent = SampleLatentState[prior["Mean"], prior["Sigma"]];
      kl = 0.;
      uncertainty = NumericMean[prior["Sigma"]] + 0.05
    ];
    history = UpdateDeterministicState[state["History"], latent, params];
    state = Association[state, "History" -> history, "Latent" -> latent, "TimeSeconds" -> time];
    action = PredictAction[state, currentScenario, params];
    trajectory = PredictTrajectory[state, action, currentScenario, params];
    safetyResult = If[
      safetyEnabled,
      SafetySupervisor[trajectory, currentScenario, params],
      <|
        "Trajectory" -> trajectory,
        "Intervened" -> False,
        "Reasons" -> {},
        "OriginalRisk" -> MeasureTrajectoryRisk[trajectory, currentScenario["CurrentObstacles"], params],
        "SafeRisk" -> MeasureTrajectoryRisk[trajectory, currentScenario["CurrentObstacles"], params]
      |>
    ];
    risk = safetyResult["SafeRisk"]["Risk"];
    stepProgress = Total[Most[safetyResult["Trajectory"]["SpeedProfile"]]] params["Model"]["PlanningStepSeconds"];
    routeProgress = N[routeProgress + stepProgress];
    reward = N[
      1.0
      + 0.04 stepProgress
      - 1.25 risk
      - 0.35 Boole[safetyResult["Intervened"]]
      - 0.30 Boole[currentScenario["SignalState"] === "Red" && Max[safetyResult["Trajectory"]["Waypoints"][[All, 2]]] > params["Product"]["StopLineMeters"]]
    ];
    state = Association[
      state,
      "Speed" -> Last[safetyResult["Trajectory"]["SpeedProfile"]],
      "PreviousAction" -> safetyResult["Trajectory"]["Command"]
    ];
    currentBev = DecodeBevState[state, currentScenario, params];
    AppendTo[
      trace,
      <|
        "Step" -> t,
        "TimeSeconds" -> time,
        "Mode" -> mode,
        "Risk" -> N[risk],
        "OriginalRisk" -> N[safetyResult["OriginalRisk"]["Risk"]],
        "Reward" -> reward,
        "Uncertainty" -> N[uncertainty],
        "KL" -> N[kl],
        "Intervened" -> safetyResult["Intervened"],
        "Reasons" -> StringRiffle[safetyResult["Reasons"], ","],
        "Speed" -> N[state["Speed"]],
        "Acceleration" -> action["Acceleration"],
        "Steering" -> action["Steering"],
        "Trajectory" -> safetyResult["Trajectory"],
        "CurrentObstacles" -> currentScenario["CurrentObstacles"],
        "Bev" -> currentBev,
        "HistoryNorm" -> N @ Norm[history],
        "LatentNorm" -> N @ Norm[latent]
      |>
    ],
    {t, steps}
  ];
  meanKLObserved = DeleteCases[Lookup[trace, "KL", 0.], 0.];
  metrics = <|
    "Scenario" -> scenario["Name"],
    "NormalizedReward" -> N @ Mean[Lookup[trace, "Reward"]],
    "MeanRisk" -> N @ Mean[Lookup[trace, "Risk"]],
    "MeanUncertainty" -> N @ Mean[Lookup[trace, "Uncertainty"]],
    "InterventionRate" -> N @ Mean[Boole /@ Lookup[trace, "Intervened"]],
    "ObservedFraction" -> N[observedCount/steps],
    "ImaginedFraction" -> N[1. - observedCount/steps],
    "RouteProgressMeters" -> routeProgress,
    "MeanKLObserved" -> If[meanKLObserved === {}, 0., N @ Mean[meanKLObserved]]
  |>;
  <|
    "Scenario" -> scenario,
    "Trace" -> trace,
    "Metrics" -> metrics
  |>
]

RunScenarioStudy[params_: DefaultDemoParameters[], opts : OptionsPattern[SimulateEpisode]] := AssociationMap[
  SimulateEpisode[#, params, opts] &,
  Keys @ BuildScenarioLibrary[params]
]

MakeRows[study_Association] := KeyValueMap[
  {
    #1,
    NumberForm[#2["Metrics"]["NormalizedReward"], {4, 2}],
    NumberForm[#2["Metrics"]["MeanRisk"], {3, 2}],
    NumberForm[#2["Metrics"]["InterventionRate"], {3, 2}],
    NumberForm[#2["Metrics"]["ObservedFraction"], {3, 2}],
    NumberForm[#2["Metrics"]["RouteProgressMeters"], {5, 1}]
  } &,
  study
]

RenderScenarioGrid[study_Association] := Module[
  {
    header = Style[#, Bold, Darker[Blue]] & /@ {"Scenario", "Norm. Reward", "Mean Risk", "Intervention Rate", "Observed Fraction", "Route Progress (m)"},
    rows = MakeRows[study]
  },
  Grid[
    Prepend[rows, header],
    Frame -> All,
    Background -> {None, {Lighter[Blue, 0.9], None}},
    Alignment -> Left,
    ItemSize -> All
  ]
]

ValidationChecks[params_: DefaultDemoParameters[]] := Module[
  {
    pedestrianNoSafety,
    pedestrianSafety,
    runtimeReset,
    runtimeRecurrent,
    ablations,
    bevState,
    imageScores,
    deploymentScores,
    checks
  },
  pedestrianNoSafety = SimulateEpisode["PedestrianConflict", params, "EnableSafetySupervisor" -> False, "Seed" -> 42, "Steps" -> 16];
  pedestrianSafety = SimulateEpisode["PedestrianConflict", params, "EnableSafetySupervisor" -> True, "Seed" -> 42, "Steps" -> 16];
  runtimeReset = RuntimeModel[params, "ResetState"];
  runtimeRecurrent = RuntimeModel[params, "FullyRecurrent"];
  ablations = BuildAblationDataset[params];
  bevState = DecodeBevState[
    InitializeLatentState[params],
    ScenarioAtTime[BuildScenario["PedestrianConflict", params], 1.2, params],
    params
  ];
  imageScores = Lookup[ablations["ImageResolution"], "DrivingScore"];
  deploymentScores = Lookup[ablations["Deployment"], "FrequencyHz"];
  checks = <|
    "GaussianKLNonNegative" -> (GaussianKLDivergence[{0.2, -0.1}, {0.4, 0.3}, {0., 0.}, {0.6, 0.5}] >= 0.),
    "ObservationDropoutMeanFinite" -> NumericQ[ObservationDropoutMean[params["Paper"]["ObservationDropout"]]],
    "SafetyReducesPedestrianRisk" -> (pedestrianSafety["Metrics"]["MeanRisk"] <= pedestrianNoSafety["Metrics"]["MeanRisk"]),
    "RecurrentFasterThanReset" -> (runtimeRecurrent["FrequencyHz"] > runtimeReset["FrequencyHz"]),
    "BevGridIsMatrix" -> (Dimensions[bevState["CombinedGrid"]] == {params["Model"]["BevGridSize"], params["Model"]["BevGridSize"]}),
    "ImageResolutionDrivingScoreMonotone" -> MonotoneIncreasingQ[imageScores],
    "DeploymentTableMatchesPaperTrend" -> (Max[deploymentScores] == deploymentScores[[2]])
  |>;
  checks
]

End[]
EndPackage[]
