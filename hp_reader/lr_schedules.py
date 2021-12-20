from tensorflow.keras import optimizers

exp_3_5 = optimizers.Adam(
    learning_rate=optimizers.schedules.ExponentialDecay(
        initial_learning_rate=10e-3,
        decay_steps=10e4,
        decay_rate=0.99,
    )
)