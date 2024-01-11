from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, Dense, Flatten, Activation

def create_model():
    input_layer = Input(shape=(64,1))
    conv1 = Conv1D(64, 1, activation='relu')(input_layer)
    conv2 = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    conv3 = Conv1D(64, 5, activation='relu', padding='same')(input_layer)
    concat1 = concatenate([conv1, conv2, conv3])
    conv4 = Conv1D(64, 7, activation='relu', padding='same')(concat1)
    flatten = Flatten()(conv4)
    dense1 = Dense(256, activation='relu')(flatten)
    dense2 = Dense(128, activation='relu')(dense1)
    output_layer = Dense(12, activation='softmax')(dense2)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

model = create_model()
