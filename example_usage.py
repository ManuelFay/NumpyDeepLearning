import numpy as np
import numpy_dl as nn


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
    inputs, targets = generate_disc_set(100000)
    split = 0.8

    train_input = inputs[:int(split * len(inputs)), :]
    train_target = targets[:int(split * len(inputs))]
    test_input = inputs[int(split * len(inputs)):, :]
    test_target = targets[int(split * len(inputs)):]

    model = nn.train_model(model=nn.SimpleNet(),
                           criterion=nn.BCEwithSoftmaxLoss(),
                           train_input=train_input,
                           train_target=train_target,
                           mini_batch_size=10,
                           eta=1e-4,
                           epochs=50,
                           print_=True)

    # Train accuracy
    model.eval()
    output = model(train_input)
    print(f'Train Accuracy: {nn.evaluate_accuracy(train_target, output)}\n')

    # Test accuracy
    output = model(test_input)
    accuracy = nn.evaluate_accuracy(test_target, output)
    print(f'Test Accuracy: {accuracy}\n')


if __name__ == "__main__":
    main()
