import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LeakyReLU, BatchNormalization
from keras.optimizers import Adam

# Função para criar o gerador
def build_generator(latent_dim, n_features):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(n_features, activation='tanh'))
    return model

# Função para criar o discriminador
def build_discriminator(n_features):
    model = Sequential()
    model.add(Dense(512, input_dim=n_features))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Função para treinar o GAN
def train_gan(generator, discriminator, combined, data, latent_dim, epochs=10000, batch_size=64, interval=1000):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Treinar o discriminador
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, valid)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Treinar o gerador
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

        # Exibir progresso
        if epoch % interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100 * d_loss[1]:.2f}] [G loss: {g_loss:.4f}]")

# Carregar os dados originais (substitua pelo caminho do seu arquivo)
file_path = 'datasets/data_training.csv'
data = pd.read_csv(file_path).values

# Normalizar os dados para o intervalo [-1, 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)

# Dimensão do espaço latente
latent_dim = 100
n_features = data.shape[1]

# Construir o gerador e o discriminador
generator = build_generator(latent_dim, n_features)
discriminator = build_discriminator(n_features)

# Compilar o discriminador
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Construir o modelo combinado (Gerador + Discriminador)
discriminator.trainable = False
z = Input(shape=(latent_dim,))
generated_data = generator(z)
validity = discriminator(generated_data)
combined = Model(z, validity)
combined.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Treinar o GAN
train_gan(generator, discriminator, combined, data, latent_dim, epochs=10000, batch_size=64, interval=1000)

# Gerar novos dados sintéticos
n_linhas_sinteticas = 1000
noise = np.random.normal(0, 1, (n_linhas_sinteticas, latent_dim))
synthetic_data = generator.predict(noise)

# Desnormalizar os dados sintéticos
data_sintetica = scaler.inverse_transform(synthetic_data)

# Salvar os dados sintéticos em um arquivo CSV
synthetic_data_df = pd.DataFrame(data_sintetica, columns=[f"feature_{i}" for i in range(n_features)])
output_path = 'dados_sinteticos_gan.csv'
synthetic_data_df.to_csv(output_path, index=False)
print(f"Dados sintéticos salvos em '{output_path}'")
