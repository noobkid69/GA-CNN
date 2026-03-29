from loss import binary_cross_entropy, binary_cross_entropy_prime, categorical_cross_entropy, categorical_cross_entropy_prime
from dense import Dense
from conv import Convolutional
from activation import Activation
from reshape import Reshape 
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import numpy as np
import numpy as np

def one_hot(y, num_classes=10):
    vec = np.zeros((num_classes, 1))
    vec[int(y.item())] = 1
    return vec

def train_model(samples, learning_rate, epochs, use_l2, activation_type):
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    x_train = x_train_full / 255.0
    x_test = x_test_full / 255.0
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
    y_test = y_test_full.reshape(-1, 1, 1)

    x_train = x_train[:samples]
    y_train = y_train_full[:samples]
    y_train = np.array([one_hot(y) for y in y_train])
    y_test = np.array([one_hot(y) for y in y_test_full[:samples]])

    l2_lambda = 0.01 if use_l2 else 0.0
    network = [
        Convolutional((1, 28, 28), 3, 5),
        Activation(activation_type),
        Reshape((5, 26, 26), (5*26*26, 1)),
        Dense(5*26*26, 10, l2_lambda=l2_lambda)]
    
    losses = []
    
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = x
            # forward pass
            for layer in network:
                output = layer.forward(output)
            error += categorical_cross_entropy(y, output)
            # backward pass
            grad = categorical_cross_entropy_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        error /= len(x_train)
        losses.append(error)
        print(f"{e} {error}")
    
    correct = 0
    wrong = []
    for x, y, idx in zip(x_test[:samples], y_test[:samples], range(samples)):
        output = x
        for layer in network:
            output = layer.forward(output)
        prediction = np.argmax(output)
        label = np.argmax(y)
        if prediction == label:
            correct += 1
        else:
            wrong.append((x.squeeze(), label, prediction))
    
    accuracy = correct / samples
    return losses, accuracy, wrong

# GUI setup
fig, ax = plt.subplots(figsize=(10, 8))
ax_loss = fig.add_subplot(2, 1, 1)
ax_loss.set_title('Training Loss')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
line, = ax_loss.plot([], [], 'b-')

# Initial values
samples_init = 10000
learning_rate_init = 0.01
epochs_init = 100
use_l2_init = True
activation_type_init = 'relu'

wrong_data = None
current_wrong_idx = 0

# Widgets
# Sliders
ax_slider_samples = fig.add_axes([0.1, 0.05, 0.65, 0.03])
slider_samples = Slider(ax_slider_samples, 'Samples', 1000, 60000, valinit=samples_init, valstep=1000)

ax_slider_lr = fig.add_axes([0.1, 0.1, 0.65, 0.03])
slider_lr = Slider(ax_slider_lr, 'Learning Rate', 0.001, 0.1, valinit=learning_rate_init)

ax_slider_epochs = fig.add_axes([0.1, 0.15, 0.65, 0.03])
slider_epochs = Slider(ax_slider_epochs, 'Epochs', 10, 200, valinit=epochs_init, valstep=10)

# Check for L2
ax_check = fig.add_axes([0.1, 0.2, 0.15, 0.1])
check_l2 = CheckButtons(ax_check, ['L2 Regularization'], [use_l2_init])

# Radio for activation
ax_radio = fig.add_axes([0.3, 0.2, 0.15, 0.1])
radio_act = RadioButtons(ax_radio, ['sigmoid', 'relu'], active=1)  # relu active

# Buttons
ax_button_run = fig.add_axes([0.6, 0.2, 0.15, 0.05])
button_run = Button(ax_button_run, 'Run Model')

ax_button_wrong = fig.add_axes([0.8, 0.2, 0.15, 0.05])
button_wrong = Button(ax_button_wrong, 'Show Wrong Predictions')

def run_training(event):
    global wrong_data
    samp = int(slider_samples.val)
    lr = slider_lr.val
    ep = int(slider_epochs.val)
    l2 = check_l2.get_status()[0]
    act = radio_act.value_selected
    print(f"Running with samples={samp}, lr={lr}, epochs={ep}, l2={l2}, activation={act}")
    losses, acc, wrong = train_model(samp, lr, ep, l2, act)
    line.set_data(range(len(losses)), losses)
    ax_loss.relim()
    ax_loss.autoscale_view()
    ax_loss.set_title(f'Training Loss (Accuracy: {acc:.2%})')
    fig.canvas.draw_idle()
    wrong_data = wrong
    print(f"Training complete. Accuracy: {acc:.2%}")

def show_wrong(event):
    if wrong_data is None:
        print("Please run the model first to generate predictions.")
        return
    num_wrong = len(wrong_data)
    if num_wrong == 0:
        print("No wrong predictions!")
        return
    
    current_idx = [0]  # Use list to allow modification in nested functions
    
    fig_wrong = plt.figure(figsize=(8, 6))
    ax_img = fig_wrong.add_subplot(111)
    ax_img.axis('off')
    
    # Buttons
    ax_prev = fig_wrong.add_axes([0.2, 0.05, 0.1, 0.075])
    button_prev = Button(ax_prev, 'Previous', color='lightblue')
    
    ax_next = fig_wrong.add_axes([0.7, 0.05, 0.1, 0.075])
    button_next = Button(ax_next, 'Next', color='lightblue')
    
    def update_image():
        ax_img.clear()
        ax_img.axis('off')
        img, true, pred = wrong_data[current_idx[0]]
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f'Wrong Prediction {current_idx[0]+1}/{num_wrong}\nTrue Label: {true}, Predicted: {pred}', fontsize=12)
        fig_wrong.canvas.draw_idle()
    
    def on_prev(event):
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_image()
    
    def on_next(event):
        if current_idx[0] < num_wrong - 1:
            current_idx[0] += 1
            update_image()
    
    def on_key(event):
        if event.key == 'left' and current_idx[0] > 0:
            current_idx[0] -= 1
            update_image()
        elif event.key == 'right' and current_idx[0] < num_wrong - 1:
            current_idx[0] += 1
            update_image()
    
    button_prev.on_clicked(on_prev)
    button_next.on_clicked(on_next)
    fig_wrong.canvas.mpl_connect('key_press_event', on_key)
    
    update_image()  # Show first image
    plt.show()

button_run.on_clicked(run_training)
button_wrong.on_clicked(show_wrong)

plt.show()