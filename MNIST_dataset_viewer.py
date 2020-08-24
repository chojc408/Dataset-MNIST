import numpy as np
import matplotlib.pyplot as plt

def get_mnist_dataset(file_name="mnist.npz"):
    mnist = np.load(file_name, allow_pickle=True)
    # print(mnist.files)
    X_train = mnist['x_train']
    X_test  = mnist['x_test']
    y_train = mnist['y_train']
    y_test  = mnist['y_test']
    # image reshape to (28, 28, 1)
    image_size = X_train.shape[1]
    print("image_size:", image_size)
    X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
    X_test  = np.reshape(X_train, [-1, image_size, image_size, 1])
    # pixel rescaling
    X_train = X_train.astype('float')/255
    X_test  = X_test
    return X_train, X_test, y_train, y_test

def view_first_mnist_images_of_digits(X, y):
    # view first images of digits
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y==i][0].reshape(28,28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def view_first_10_mnist_image_of_digit(X, y, digit):
    # view first images for digits
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y==digit][i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

X_train, X_test, y_train, y_test = get_mnist_dataset(file_name="mnist.npz")
print("X_train.shape =", X_train.shape)
print("y_train.shape =", y_train.shape)
print("X_test.shape =", X_test.shape)
print("y_test.shape =", y_test.shape)
print("View first images of digits")
view_first_mnist_images_of_digits(X_train, y_train)
print("View first 10 images images of digit == 3")
view_first_10_mnist_image_of_digit(X_train, y_train, digit=3)
