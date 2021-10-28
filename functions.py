import matplotlib.pyplot as plt

def plot_history(history, plot_name):
    # Visualize history
    # Plot history: Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('%s (Loss)' % plot_name)
    plt.ylabel('Loss value')
    plt.xlabel('No. epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Plot history: Accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('%s (Accuracy)' % plot_name)
    plt.ylabel('Accuracy value (%)')
    plt.xlabel('No. epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt

images_per_row = 16

def get_layer_outputs_and_names(model, num_selected_layers):
    selected_layers = model.layers[:num_selected_layers]
    layer_outputs = [layer.output for layer in selected_layers] 
    layer_names = [layer.name for layer in selected_layers]
    return layer_outputs, layer_names

def display_image_activations(layer_outputs, layer_names, activations, selected_image):
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[selected_image,
                                                :, :,
                                                col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                            row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


# TODO: refactor this to avoid repeated computation when checking multiple images
def visualise_layers(model, num_selected_layers, test_x, selected_image):
    layer_outputs, layer_names = get_layer_outputs_and_names(model, num_selected_layers)

    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(test_x) 

    display_image_activations(layer_outputs, layer_names, activations, selected_image)

