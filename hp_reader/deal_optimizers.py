from tensorflow.keras import optimizers

# Wrong rate
exp_3_5_wrong = optimizers.Adam(
    learning_rate=optimizers.schedules.ExponentialDecay(
        initial_learning_rate=10e-3,
        decay_steps=10e4,
        decay_rate=0.99,
    )
)

exp_4_6 = optimizers.Adam(
    learning_rate=optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=100,
        decay_rate=0.99,
    )
)

pol_4_5 = optimizers.Adam(
    learning_rate=optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1e-4,
        decay_steps=1e5,
        end_learning_rate=1e-5,
        power=0.5
    )
)