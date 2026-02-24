n_samples, n_window, n_features = (None, 30, 9)

# Encoder architecture parameter
kernel_size = 5
latent_dim = 15
n_filters = 20
kernel_size = 5

epochs = 100
batch = 32

# 2. Define the encoder architecture

encoder = tf.keras.Sequential()
input_shape = (n_window, n_features)

# 2. Define the encoder architecture
encoder.add(tf.keras.layers.Input(shape=input_shape))
encoder.add(tf.keras.layers.Conv1D(filters=n_filters*n_features, kernel_size=kernel_size, activation='relu', padding='same'))
encoder.add(tf.keras.layers.Conv1D(filters=n_filters*n_features, kernel_size=kernel_size, activation='relu', padding='same'))
encoder.add(tf.keras.layers.GlobalMaxPooling1D())
encoder.add(tf.keras.layers.Dense(latent_dim, activation='relu'))



_,n_window_after_pool2,n_filters_after_conv2 = (None, 30, 180)

units_before_reshape = 180


decoder = tf.keras.Sequential()
decoder.add(tf.keras.layers.Input(shape=(latent_dim,)))
decoder.add(tf.keras.layers.Dense(units=n_window_after_pool2*n_filters_after_conv2, activation='relu'))
decoder.add(tf.keras.layers.Reshape((n_window_after_pool2, n_filters_after_conv2)))
decoder.add(tf.keras.layers.Conv1DTranspose(filters=n_filters*n_features, kernel_size=kernel_size, activation='relu'))
decoder.add(tf.keras.layers.Conv1D(filters=n_filters*n_features, kernel_size=kernel_size, activation='relu'))

decoder.add(tf.keras.layers.Dense(9,activation='relu'))

class FixedLearningRateAdam(tf.keras.optimizers.Adam):
    def __init__(self, name='FixedAdam', **kwargs):
        # Define the fixed learning rate here
        fixed_lr = 0.0001  # You can change this value if needed
        super().__init__(learning_rate=fixed_lr, name=name, **kwargs)

encoder_decoder = tf.keras.Sequential()
encoder_decoder.add(encoder)
encoder_decoder.add(decoder)

encoder_decoder.compile(optimizer=FixedLearningRateAdam(), loss='mean_squared_error')