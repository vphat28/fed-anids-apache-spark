from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))

# retrieve the last two layers of the autoencoder model
decoder_layer1 = autoencoder.layers[-2]
decoder_layer2 = autoencoder.layers[-1]

# create the decoder model
decoder = Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))

autoencoder.compile(optimizer='adam', loss='mse')

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Train the autoencoder for 5 epochs
autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=1028,
                shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode some digits
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Select a random sample to display
sample_index = np.random.randint(x_test.shape[0])
original_image = x_test[sample_index]
encoded_repr = encoded_imgs[sample_index]
decoded_image = decoded_imgs[sample_index]
print(encoded_repr)

# Reshape the original and decoded images into a 28x28 matrix
original_image = original_image.reshape(28, 28)
decoded_image = decoded_image.reshape(28, 28)

# Display the original and decoded images
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Hình gốc')
plt.axis('off')

plt.subplot(1, 3, 2)
encoded_vector = encoded_repr.flatten()  # Reshape the encoded image into a vector
plt.plot(encoded_vector)  # Plot the vector
plt.title('Dạng mã hóa')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(decoded_image, cmap='gray')
plt.title('Hình phục dựng')
plt.axis('off')

plt.tight_layout(pad=5.0)  # Add padding between subplots
plt.show()
