import numpy as np
import numpy_dl as nn


def compute_nb_errors(model,
                      criterion,
                      train_input,
                      train_target,
                      test_input,
                      test_target,
                      mini_batch_size,
                      eta,
                      epochs=25,
                      printV=True):

    model = nn.train_model(model, criterion, train_input, train_target, mini_batch_size, eta, epochs, printV=printV)

    # Train accuracy # TODO: process by batch
    output = np.zeros((len(train_input), 1))
    for i in range(train_input.shape[0]):
        tmp = model(np.expand_dims(train_input[i], 1)).reshape(1, -1)
        output[i, :] = tmp

    print(f'Train Accuracy: {nn.evaluate_accuracy(train_target, output)}\n')

    # Test accuracy
    output = np.zeros((len(test_input), 1))
    for i in range(test_input.shape[0]):
        tmp = model(np.expand_dims(test_input[i], 1)).reshape(1, -1)
        output[i, :] = tmp

    accuracy = nn.evaluate_accuracy(test_target, output)
    print(f'Test Accuracy: {accuracy}\n')
        
    return accuracy, output


# Generate test set
def generate_disc_set(nb):

    inputs = np.zeros((nb, 2))
    targets = np.ones((nb, 1))

    while abs(targets.mean()-0.5) > 0.01:
        inputs = np.random.uniform(-1, 1, (nb, 2))
        targets = np.expand_dims((np.sqrt(inputs[:, 0] ** 2 + inputs[:, 1] ** 2) < 0.79788456), 1)
        print(targets.mean())

    mu, std = inputs.mean(0), inputs.std(0)
    inputs = (inputs - mu) / std

    return inputs, (targets*2)-1


def main():
    inputs, targets = generate_disc_set(1000)
    split = 0.8

    accuracy, output = compute_nb_errors(model=nn.SimpleNet(),
                                         criterion=nn.BCEwithSoftmaxLoss(),
                                         train_input=inputs[:int(split * len(inputs)), :],
                                         train_target=targets[:int(split * len(inputs))],
                                         test_input=inputs[int(split * len(inputs)):, :],
                                         test_target=targets[int(split * len(inputs)):],
                                         mini_batch_size=100,
                                         eta=1e-4,
                                         epochs=50)
    print(f'Test Accuracy: {accuracy}')


if __name__ == "__main__":
    main()

