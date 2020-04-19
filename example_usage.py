import numpy as np
import numpy_dl as nn

# pylint: disable=too-many-arguments
def compute_nb_errors(model,
                      criterion,
                      train_input,
                      train_target,
                      test_input,
                      test_target,
                      mini_batch_size,
                      eta,
                      epochs=25,
                      print_=True):

    model = nn.train_model(model, criterion, train_input, train_target, mini_batch_size, eta, epochs, print_=print_)

    # Train accuracy
    output = model(train_input)
    print(f'input {train_input.shape}')
    print(f'output {output.shape}')

    model.eval()
    output = model(train_input)
    print(f'Train Accuracy: {nn.evaluate_accuracy(train_target, output)}\n')

    # Test accuracy
    output = model(test_input)
    accuracy = nn.evaluate_accuracy(test_target, output)
    print(f'Test Accuracy: {accuracy}\n')
    return accuracy, output


# Generate test set
def generate_disc_set(number_samples):

    inputs = np.zeros((number_samples, 2))
    targets = np.ones((number_samples, 1))

    while abs(targets.mean()-0.5) > 0.01:
        inputs = np.random.uniform(-1, 1, (number_samples, 2))
        targets = np.expand_dims((np.sqrt(inputs[:, 0] ** 2 + inputs[:, 1] ** 2) < 0.79788456), 1)
        print(targets.mean())

    mean, std = inputs.mean(0), inputs.std(0)
    inputs = (inputs - mean) / std

    return inputs, (targets*2)-1


def main():
    inputs, targets = generate_disc_set(10000)
    split = 0.8

    accuracy, _ = compute_nb_errors(model=nn.SimpleNet(),
                                    criterion=nn.BCEwithSoftmaxLoss(),
                                    train_input=inputs[:int(split * len(inputs)), :],
                                    train_target=targets[:int(split * len(inputs))],
                                    test_input=inputs[int(split * len(inputs)):, :],
                                    test_target=targets[int(split * len(inputs)):],
                                    mini_batch_size=10,
                                    eta=1e-4,
                                    epochs=50)
    print(f'Test Accuracy: {accuracy}')


if __name__ == "__main__":
    main()
