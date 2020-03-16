import numpy as np


# TODO: make CLI
def train_model(model, criterion, train_input, train_target, mini_batch_size, eta, epochs=25, print_=False):
    for e in range(0, epochs):
        sum_loss = 0
        # We do this with mini-batches
        for b in range(0, train_input.shape[0], mini_batch_size):
            model.zero_grad()
            for i in range(mini_batch_size):
                output = model(np.expand_dims(train_input[i + b], 1))
                loss = criterion(output.squeeze(1), np.expand_dims(train_target[i + b], 1))
                sum_loss = sum_loss + loss.item()
                model.backward(criterion.backward())

            for p in model.param_func():
                p.sub_grad(eta)

        if print_:
            print("Epoch: {} \t -> Loss: {} ".format(e, sum_loss))

    return model
