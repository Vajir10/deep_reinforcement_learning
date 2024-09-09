class FrozenLakeConfig:
    # Training parameters
    LEARNING_RATE = 0.7
    GAMMA = 0.9
    N_TRAINING_EPISODES = 1000
    # EVALUDATION PARAMETERS
    N_EVAL_EPISODES = 100

    # ENV PARAMETERS
    ENV_ID = "FrozenLake-v1"
    MAX_STEPS = 99
    EVAL_SEED = []

    # EXPLORATION [PARAMETERS
    MAX_EPSILON = 1.0
    MIN_EPSILON = 0.05
    DECAY_RATE = 0.0005

    VIDEO_FILENAME = 'frozen_lake.mp4'
