from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Model
import tensorflow as tf


def SelfAttention(conv):
    output_layer_SA_dot = layers.Attention()([conv, conv])

    output_layer_softmax = layers.Softmax()(output_layer_SA_dot)

    output_layer_SA_concat = layers.Attention(score_mode='concat')([output_layer_softmax, conv])
    
    return output_layer_SA_concat

def createModel(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, IMAGE_CHANNEL):
    """
    createModel() function creates the model architecture for the 3D CNN model.
    :return: model 
    
    The model architecture is as follows:
    1. Input layer
    2. 3D Convolution layer with 8 filters, kernel size (3,3,7), activation function 'relu' and padding 'same'
    3. Spatial Dropout layer with dropout rate 0.3
    4. 3D Convolution layer with 16 filters, kernel size (3,3,5), activation function 'relu' and padding 'same'
    5. Spatial Dropout layer with dropout rate 0.3
    6. 3D Convolution layer with 32 filters, kernel size (3,3,3), activation function 'relu'
    7. Reshape layer to reshape the output of 3D Convolution layer to 2D
    8. 2D Convolution layer with 64 filters, kernel size (3,3), activation function 'relu'
    9. Flatten layer to flatten the output of 2D Convolution layer
    10. Dropout layer with dropout rate 0.4
    11. Dense layer with 256 neurons and activation function 'relu'
    12. Dropout layer with dropout rate 0.4
    13. Dense layer with 128 neurons and activation function 'relu'
    14. Output layer with 128 neurons and activation function 'relu'
    
    """
    
    input_layer = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IMAGE_CHANNEL))

    output_layer_1_conv = layers.Conv3D(filters=8, kernel_size=(3,3,7), activation='relu',input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IMAGE_CHANNEL),padding='same')(input_layer)

    output_layer_1_drop3d = layers.SpatialDropout3D(rate=0.3, data_format='channels_last')(output_layer_1_conv,training=True)

    output_layer_2_conv = layers.Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu', padding='same')(output_layer_1_drop3d)

    output_layer_2_drop3d = layers.SpatialDropout3D(rate=0.3, data_format='channels_last')(output_layer_2_conv,training=True)
    
    output_layer_3_conv = layers.Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(output_layer_2_drop3d)

    output_layer_3_SA = SelfAttention(output_layer_3_conv)
    
    output_layer_3_reshaped = layers.Reshape((output_layer_3_SA.shape[1], output_layer_3_SA.shape[2], output_layer_3_SA.shape[3]*output_layer_3_SA.shape[4]))(output_layer_3_SA)

    output_layer_4_conv = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(output_layer_3_reshaped)
    
    output_layer_5_SA = SelfAttention(output_layer_4_conv)
    
    output_layer_5_normalization = layers.LayerNormalization()(output_layer_5_SA)

    output_layer_6_flatten = layers.Flatten()(output_layer_5_normalization)

    output_layer_6_drop = layers.Dropout(rate=0.4)(output_layer_6_flatten,training=True)

    output_layer_6_dense = layers.Dense(256, activation='relu')(output_layer_6_drop)

    output_layer_7_conv = layers.Dropout(0.4)(output_layer_6_dense,training=True)

    output_layer_7_dense = layers.Dense(128, activation='relu')(output_layer_7_conv)

    model = Model(inputs=input_layer, outputs=output_layer_7_dense)
    
    print(model.summary())
    return model

if __name__ == '__main__':
    model = createModel(11, 11, 30, 1)
    print(model.summary())
    
    