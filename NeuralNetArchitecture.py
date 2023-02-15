import numpy as np
import time

class activations(object):
    @staticmethod
    def sigmoid(z, derivative = False):
        if not derivative:
            return 1 / (1+np.exp(-z))
        else:
            return np.exp(-z) / (1 + np.exp(-z))**2
    @staticmethod
    def softmax(z, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(z - z.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

class costs(object):
    @staticmethod
    def MSE(a,label, derivative = True):
        if derivative:
            return 2*(a - label)
        else:
            return (a - label)**2
    @staticmethod
    def CrossEntropy(a,label, derivative = True):
        if derivative:
            pass
        else:
            return (-(label*np.log(a) - (1-label)*np.log(1-a)))

class helper(object):
    @staticmethod
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

class layer(object):
    layer_ID = 1
    def __init__(self, n_in, n_out, activation = activations.sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation
        self.ID = layer.layer_ID
        layer.layer_ID += 1

    def print_layer():
        return f'Layer {self.ID}: n_in = {self.n_in} / n_out = {self.n_out} / activation = {self.activation}'

class dense(layer):
    def __init__(self, n_in, n_out, activation = activations.sigmoid):
        super().__init__(n_in, n_out, activation)
        self.weights = np.random.randn(n_out, n_in) * np.sqrt(1 / n_out)
        self.biases = np.zeros((n_out, 1))

    def forwards_pass(self, input):
        ''' Takes in an input of size n_in and calculates the outpur vector given the weights and biases '''
        assert(len(input) == self.n_in), 'Input dimmension != n_in'
        z = np.matmul(self.weights, input) + self.biases
        a = self.activation(z)
        return z, a

    def backwards_pass(self, error, z, prev_z):
        ''' Takes in an error vector of size n_out and calculates the error of the previous layer, size n_in. Note
        that to do so we need the weighted sums (z) of the current layer.

         NOTE: The return is NOT the error of the previous layer. The layer error is this error AFTER we multuply by
         self.activation(z, derivative=True)'''
        assert(len(error) == self.n_out), 'Error dimmension != n_out'
        assert(len(prev_z) == self.n_in), 'Z values do not conform'
        in_error = np.dot(self.weights.T, error) * self.activation(prev_z, derivative=True)
        return in_error

class dense_output(dense):
    def __init__(self, n_in, n_out, activation = activations.softmax, cost = costs.MSE):
        super().__init__(n_in, n_out, activation)
        self.cost = cost

    def final_error(self, z, a, label):
        ''' Calculates the error of the final layer to propogate backwards given the label and z/a values of the output
        layer '''
        assert len(label) == self.n_out, 'Label does not conform to output size'
        grad_C = self.cost(a=a, label=label, derivative=True)
        error = (grad_C / self.n_out) * self.activation(z, derivative=True)
        return error

class Network(object):
    def __init__(self, layers, learning_rate = 0.01, batch_size = 50, epochs = 10, lmbda = 1):
        print('Compiling Network...')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lmbda = lmbda
        self.layers = layers
        self.num_layers = len(self.layers)

    def network_info(self):
        print(f'{type(self).__name__} Network with {self.num_layers} Layers:')
        for layer in self.layers:
            if 'output' not in type(layer).__name__:
                print(f'\tLayer {layer.ID}: Type: {type(layer).__name__} / Activation: {layer.activation} / '
                      f'Params: {layer.weights.size + layer.biases.size}')
            else:
                print(f'\tLayer {layer.ID}: Type: {type(layer).__name__} / Activation: {layer.activation} / '
                      f'Cost: {layer.cost} / Params: {layer.weights.size + layer.biases.size}')

    def train(self, x_train, y_train, x_val=None, y_val=None, show=True, shuffle = True):
        ''' Trains the network over x_train (features) and y_train (labels)
         Optional inputs are x_val and y_val, for validation sets and show. If show is true then the network will check
         it's accuracy after each epoch on the validation set or the training set, if no validation set is inputted. '''
        # For each epoch
        assert len(x_train) == len(y_train), 'Inputs and Labels are not the same size'

        if show:
            print(f'Beginning Training!')
            print(f'\tStarting Accuracy: {self.test_accuracy(x_train, y_train)}\n')
        for epoch in range(self.epochs):
            start = time.perf_counter()
            if show:
                print(f'Starting Epoch {epoch + 1}')
            data = [(x,y) for x,y in zip(x_train, y_train)]
            if shuffle:
                np.random.shuffle(data)
            # Batch the data
            batches = helper.batch(data, n=self.batch_size)
            for batch in batches:
                # Add up all the gradients of all the points in the batch one by one
                change_params = None
                for data_point in batch:
                    update_dict = self.backpropogate(input=data_point[0], label=data_point[1])
                    if change_params == None:
                        change_params = update_dict
                    else:
                        for key in change_params.keys():
                            change_params[key] += update_dict[key]

                self.update_network(change_params, len(x_train))

            end = time.perf_counter()
            if show:
                accuracy = self.test_accuracy(x_val, y_val)
                print(f'Epoch {epoch + 1} Complete')
                print(f'\tCurrent Accuracy: {accuracy}')
                print(f'\tCurrent Train Accuracy: {self.test_accuracy(x_train, y_train)}')
                print(f'\tTime Elapsed: {round(end-start, 2)} seconds\n')
        print('Done Training!')

    def update_network(self, change_params, set_size):
        for key, value in change_params.items():
            # Regularize The Weights
            if key[0] == 'W':
                scaling_factor = 1 - ((self.learning_rate * self.lmbda) / set_size)
                self.params[key] *= scaling_factor
            self.params[key] -= (self.learning_rate * value)


    def test_accuracy(self, x_data, y_data):
        ''' Returns the accuracy of the network over the given dataset as a decimal. Not necessarily right for
         quantitative data. This is more categorically focused '''
        data = [(x,y) for x,y in zip(x_data, y_data)]
        np.random.shuffle(data)
        predictions = []
        for point in data[0:1000]:
            if self.layers[-1].n_out != 1:
                output = self.network_values(point[0])[f'a{self.num_layers}']
                pred = np.argmax(output)
                predictions.append(pred == np.argmax(point[1]))
            else:
                output = self.network_values(point[0])[f'a{self.num_layers}']
                difference = point[1] - output
                predictions.append(difference/len(x_data))
        return np.mean(predictions)


class Sequential(Network):
    def __init__(self, layers, learning_rate = 0.01, batch_size = 50, epochs = 10, lmbda = 1):
        ''' layers is a list of layer objects, lmbda is the regularization constant. Other inputs are
         self explanatory. Params is a list of tuples containing weights and biases for each layer.
         Weights and Biases are tensors created from the params.'''
        super().__init__(layers, learning_rate, batch_size, epochs, lmbda)
        self.params = {}
        self.size = [self.layers[0].n_in]
        last_out = None
        for i in range(len(self.layers)):
            # Make sure sizes are proper
            assert last_out == None or last_out == self.layers[i].n_in, 'Layer size discrepency'
            last_out = self.layers[i].n_out
            # Get sizes
            self.size.append(self.layers[i].n_out)
            # Append parameters to self.params
            layer_weight = self.layers[i].weights
            layer_bias = self.layers[i].biases
            self.params[f'W{i+1}'] = layer_weight
            self.params[f'B{i+1}'] = layer_bias
        assert 'output' in type(self.layers[-1]).__name__ , 'Final layer is not an output layer'
        print('Done Compiling!\n')

    def backpropogate(self, input, label, queue = None):
        ''' Returns the partial derivatives of each weight and bias through backpropogation. '''
        errors = {}
        # Forwards Pass
        vals = self.network_values(input)
        # Backwards Pass
        layer_count = self.num_layers
        for layer in self.layers[::-1]:
            # Get a and z values for current layer and last layer. If no values in previous layer, break.
            layer_a = vals[f'a{layer_count}']
            layer_z = vals[f'z{layer_count}']
            try:
                prev_z = vals[f'z{layer_count-1}']
            except:
                break
            # Final Layer
            if layer_count == self.num_layers:
                # Initial Error
                error = layer.final_error(layer_z, layer_a, label)
                errors[f'e{layer_count}'] = error
                # First sendback
                error = layer.backwards_pass(error, layer_z, prev_z)
                errors[f'e{layer_count-1}'] = error
            # Other Layers
            else:
                error = layer.backwards_pass(error, layer_z, prev_z)
                errors[f'e{layer_count - 1}'] = error
            layer_count -= 1
        # Get update values from the error
        updates = self.error_to_update(errors, vals)
        # For multiprocessing
        if queue != None:
            queue.put(updates)
        return updates

    def network_values(self, input):
        ''' Stores the z and a values for each layer given an input. '''
        assert (len(input) == self.size[0]), 'Input not of right size'
        val_dict = {}
        # Input activations
        val_dict['a0'] = input
        # Other acivations
        count = 1
        for layer in self.layers:
            z, a = layer.forwards_pass(input)
            val_dict[f'z{count}'] = z
            val_dict[f'a{count}'] = a
            input = a
            count += 1
        return val_dict

    def error_to_update(self, errors, vals):
        ''' Given a dictionary of errors for each layer, return another dictionary of how to update each weight and
         bias '''
        to_update = {}
        for key in errors.keys():
            # Get the index of the error we are on so we know which weight/bias to update (precaution)
            index_num = int(key[1::])
            error = errors[key]
            # Find weights update
            prev_a = vals[f'a{index_num - 1}']
            weight_change = np.outer(error, prev_a)
            to_update[f'W{index_num}'] = weight_change
            # Find bias update
            bias_change = error
            to_update[f'B{index_num}'] = bias_change
        return to_update

if __name__ == '__main__':
    pass