"""
Centralized configuration for the CARLA speed planning comparison project.
All adjustable parameters are defined here for easy modification.
"""


# ==========================================================================
#  PIPELINE CONTROL — enable/disable independent stages
# ==========================================================================

# Which planners to include (any subset of ['rule', 'ann', 'snn'])
ACTIVE_PLANNERS = ['rule', 'ann', 'snn']

# Stage toggles — each stage runs independently
STAGE_TRAIN = True               # run imitation learning training
STAGE_EVALUATE = True            # run CARLA evaluation episodes
STAGE_PLOT = True                # generate matplotlib plots

# Training data source
USE_SYNTHETIC_DATA = True        # True = synthetic, False = collect from CARLA

# Which plots to generate (toggle individually)
PLOT_SPEED_VS_TIME = True
PLOT_ENERGY_VS_TIME = True
PLOT_CONTROL_SIGNALS = True
PLOT_JERK_VS_TIME = True
PLOT_ENERGY_BAR = True
PLOT_METRICS_COMPARISON = True
PLOT_SPIKE_RASTER = True         # only if SNN is active

# Vehicle blueprint filter
VEHICLE_BLUEPRINT = 'vehicle.tesla.model3'


# ==========================================================================
#  CARLA CONNECTION
# ==========================================================================
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
CARLA_TIMEOUT = 30.0            # seconds


# ==========================================================================
#  SIMULATION
# ==========================================================================
DT = 0.05                       # simulation timestep (seconds), 20 Hz
MAX_STEPS = 2000                 # max steps per episode


# ==========================================================================
#  ROUTE
# ==========================================================================
ROUTE_TOTAL_DISTANCE = 500.0     # meters
ROUTE_SPACING = 2.0              # waypoint spacing (meters)
WAYPOINT_ADVANCE_THRESHOLD = 3.0 # distance to advance waypoint (meters)
SPAWN_POINT_INDEX = 0            # default spawn point (used by training data collection)
SPAWN_POINT_INDICES = [0, 1, 2, 3]  # spawn points for evaluation diversity


# ==========================================================================
#  LATERAL CONTROL
# ==========================================================================
STEERING_GAIN = 2.0              # steering sensitivity multiplier


# ==========================================================================
#  SPECTATOR CAMERA
# ==========================================================================
SPECTATOR_HEIGHT = 5             # camera Z offset (meters)
SPECTATOR_PITCH = -15            # camera pitch (degrees, slight look-down)
SPECTATOR_DISTANCE = 8           # camera distance behind vehicle (meters)


# ==========================================================================
#  ENERGY MODEL
# ==========================================================================
# energy = throttle^2 + ENERGY_BRAKE_COEFF * brake
ENERGY_BRAKE_COEFF = 0.1


# ==========================================================================
#  PID CONTROLLER
# ==========================================================================
PID_KP = 1.0                    # proportional gain
PID_KI = 0.1                    # integral gain
PID_KD = 0.05                   # derivative gain
PID_WINDUP_LIMIT = 10.0         # integral anti-windup clamp [-limit, +limit]


# ==========================================================================
#  RULE-BASED PLANNER
# ==========================================================================
# target_speed = RULE_K / (1 + RULE_ALPHA * |curvature|)
RULE_MAX_SPEED = 12.0            # max target speed (m/s)
RULE_MIN_SPEED = 3.0             # min target speed (m/s)
RULE_K = 12.0                   # base speed constant
RULE_ALPHA = 50.0               # curvature sensitivity


# ==========================================================================
#  ANN PLANNER
# ==========================================================================
ANN_INPUT_DIM = 3
ANN_HIDDEN_DIM = 64
ANN_OUTPUT_CLIP_MIN = 0.0       # target speed clamp (m/s)
ANN_OUTPUT_CLIP_MAX = 15.0


# ==========================================================================
#  SNN PLANNER
# ==========================================================================
SNN_INPUT_DIM = 3
SNN_HIDDEN_DIM = 64
SNN_NUM_STEPS = 25               # SNN simulation timesteps per inference
SNN_BETA = 0.85                  # LIF membrane decay rate
SNN_SURROGATE_SLOPE = 25         # surrogate gradient slope
SNN_OUTPUT_CLIP_MIN = 0.0
SNN_OUTPUT_CLIP_MAX = 15.0


# ==========================================================================
#  TRAINING — IMITATION LEARNING
# ==========================================================================
TRAIN_EPOCHS = 2000              # training epochs for both ANN and SNN
TRAIN_BATCH_SIZE = 64
TRAIN_LOG_INTERVAL = 20          # print loss every N epochs

# ANN training
ANN_LR = 1e-3                   # ANN learning rate
ANN_GRAD_CLIP_NORM = 1.0        # gradient clipping max norm (matches SNN)

# SNN training
SNN_LR = 5e-4                   # SNN learning rate (lower for stability)
SNN_GRAD_CLIP_NORM = 1.0        # gradient clipping max norm


# ==========================================================================
#  TRAINING — RL (OPTIONAL)
# ==========================================================================
RL_LR = 1e-4
RL_GAMMA = 0.99                  # discount factor
RL_LAMBDA_JERK = 0.1             # jerk penalty weight
RL_LAMBDA_COLLISION = 10.0       # collision penalty weight
RL_LAMBDA_PROGRESS = 1.0         # progress reward weight
RL_EXPLORATION_NOISE = 0.5       # Gaussian noise std


# ==========================================================================
#  SYNTHETIC DATASET
# ==========================================================================
SYNTHETIC_NUM_SAMPLES = 20000
SYNTHETIC_SPEED_RANGE = (0, 15)          # uniform (m/s)
SYNTHETIC_CURVATURE_SCALE = 0.02         # exponential distribution scale
SYNTHETIC_DISTANCE_RANGE = (1, 20)       # uniform (meters)


# ==========================================================================
#  DATA COLLECTION (from CARLA)
# ==========================================================================
COLLECT_EPISODES = 5             # episodes to collect for training data


# ==========================================================================
#  EVALUATION
# ==========================================================================
EVAL_EPISODES = 1                # episodes per planner per spawn point


# ==========================================================================
#  VISUALIZATION
# ==========================================================================
PLOT_DPI = 150
PLOT_FIGSIZE_WIDE = (12, 5)
PLOT_FIGSIZE_BAR = (8, 5)
PLOT_FIGSIZE_METRICS = (26, 5)
PLOT_FIGSIZE_SPIKE = (12, 6)
PLOT_COLORS = ['#2196F3', '#FF9800', '#4CAF50']  # blue, orange, green
SPIKE_RASTER_NEURONS = 20       # neurons to show in spike raster


# ==========================================================================
#  DEBUG DRAW (CARLA overlay)
# ==========================================================================
DEBUG_TEXT_Z_OFFSET = 3.0        # text height above vehicle (meters)
DEBUG_LIFE_TIME = 0.1            # draw refresh interval (seconds)
DEBUG_LOOK_AHEAD = 30            # waypoints to draw ahead
DEBUG_LOOK_BEHIND = 5            # waypoints to draw behind
DEBUG_WP_Z_OFFSET = 0.5         # waypoint height above road
DEBUG_TRAJECTORY_Z_OFFSET = 0.3
DEBUG_TRAJECTORY_THICKNESS = 0.05
