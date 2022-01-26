import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import RepeatVector

class Sampling(keras.layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return (tf.exp(z_log_var * 0.5) * epsilon) + z_mean 
        
def vae_encoder(latent_dim=5000, cols = 214):

    encoder_inputs = keras.Input(shape=(75,cols))
    x = Dense(cols, activation="relu")(encoder_inputs)
    x = LSTM(cols, activation='relu', return_sequences=True, name='lstm1')(x)
    x = LSTM(int(cols//1.3), activation="relu", return_sequences=True, name='lstm2')(x)
    x = LSTM(int(cols//2), activation="relu", return_sequences=True, name='lstm3')(x)

    x = Dense(100, activation="relu", name='Dense1')(x)
    x = tf.keras.layers.Flatten(name='Netflow_burst')(x)
    x = tf.keras.layers.LeakyReLU()(x)

    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    VAE_encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return VAE_encoder

def vae_decoder(latent_dim=5000, cols = 214):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = Dense(latent_dim, activation="relu", name='Dense1')(latent_inputs)
    x = Dense(7500, activation="relu", name='Dense2')(x)

    x = tf.keras.layers.Reshape((75,100))(x)

    x = LSTM(int(cols//2), activation="relu", return_sequences=True, name='lstm1')(x)
    x = LSTM(int(cols//1.3), activation="relu", return_sequences=True, name='lstm2')(x)
    x = LSTM(cols, activation='relu', return_sequences=True, name='lstm3')(x)

    decoder_outputs = Dense(cols, activation="sigmoid")(x)

    VAE_decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    return VAE_decoder

def VAE_loss(data, reconstruction, z_mean, z_log_var, z):
    
    mae = tf.keras.losses.MeanSquaredError()
    reconstruction_loss = tf.math.reduce_sum(mae(data,reconstruction))
    kl_loss = 0.5*tf.math.reduce_sum(tf.math.exp(z_log_var)+tf.math.square(z_mean)-1.-z_log_var)
    
    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    return reconstruction_loss, kl_loss, total_loss

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name="accuracy")

    def pred(data):
        z_mean, z_log_var, z = self.encoder(data)
        return self.decoder(z)
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.accuracy_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:  
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss, kl_loss, total_loss = VAE_loss(data, reconstruction, z_mean, z_log_var, z)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))           

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.accuracy_tracker.update_state(data,reconstruction)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }
    
    def test_step(self,data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss, kl_loss, total_loss = VAE_loss(data, reconstruction, z_mean, z_log_var, z)
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.accuracy_tracker.update_state(data,reconstruction)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "accuracy": self.accuracy_tracker.result(),
        }

def compile_vae(lr=1e-5):
    vae = VAE(encoder=vae_encoder(), decoder=vae_decoder())
    vae.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr)
    )
    return vae