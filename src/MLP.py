import numpy as np

class Layer:
    def __init__(self, n_input, n_output, activation_func="relu", random_seed=1):
        self.n_input = n_input
        self.n_output = n_output
        self.activation_func = activation_func
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.weights = self.init()
        self.bias = np.full((n_output, 1), 0.01)

    def He_init(self):
        # Uso la inicializacion He porque es la mejor si las funciones de activacion son de la familia de las ReLU
        sigma_cuad = 2/self.n_input
        sigma = np.sqrt(sigma_cuad)
        return np.random.normal(0, sigma, size=(self.n_output, self.n_input))

    def Glorot_init(self):
        # Uso la inicializacion Glorot si la funcion de activacion es softmax
        sigma_cuad = 2/(self.n_input + self.n_output)
        sigma = np.sqrt(sigma_cuad)
        return np.random.normal(0, sigma, size=(self.n_output, self.n_input))

    def init(self):
        if self.activation_func == "relu":
            return self.He_init()
        if self.activation_func == "softmax":
            return self.Glorot_init()

    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        # Shift values by max to prevent numerical overflow in exp
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        # Use axis=0 to sum along columns (classes), keeping batches separated
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def activation(self, Z):
        if self.activation_func == "relu":
            return self.relu(Z)
        if self.activation_func == "softmax":
            return self.softmax(Z)

    def output(self, input):
        return self.activation(self.weights @ input + self.bias)

class MLP:
    def __init__(self, n_layers, nodes_per_layer, random_seed=1):
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer
        self.random_seed = random_seed
        self.layer_list = []
        np.random.seed(random_seed)
        
        for i in range(n_layers-1):
            self.layer_list.append(Layer(nodes_per_layer[i], nodes_per_layer[i+1], random_seed=random_seed))
        
        self.layer_list.append(Layer(nodes_per_layer[n_layers-1], nodes_per_layer[n_layers], "softmax", random_seed=random_seed))
    
    def print_info(self):
        print("Numero de capas: ", self.n_layers)
        print("Numero de nodos por capa: ", self.nodes_per_layer)
        for i, layer in enumerate(self.layer_list):
            print("Capa ", i+1, ": ", layer.n_input, "->", layer.n_output, " con funcion de activacion ", layer.activation_func)
        print("Pesos de la capa 1: ", self.layer_list[0].weights)
        print("Bias de la capa 1: ", self.layer_list[0].bias)

    def cross_entropy(self, y_true, y_pred):
        epsilon = 1e-12
        m = y_true.shape[1]
        loss = - (1/m) * np.sum(y_true * np.log(y_pred + epsilon))
        return loss
    
    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def forward_pass(self, X):
        Z = [None] * (self.n_layers + 1)
        A = [None] * self.n_layers
        
        Z[0] = X
        for l in range(self.n_layers): 
            A[l] = self.layer_list[l].weights @ Z[l] + self.layer_list[l].bias
            Z[l+1] = self.layer_list[l].activation(A[l])
        
        return Z, A

    def back_propagation(self, X, y): 
        delta = [None] * self.n_layers
        weights_grad = [None] * self.n_layers
        bias_grad = [None] * self.n_layers

        # Forward pass
        Z, A = self.forward_pass(X)
        y_pred = Z[-1]
        loss = self.cross_entropy(y, y_pred)
        
        L = self.n_layers - 1 
        
        # Backward pass
        delta[L] = y_pred - y
        m = X.shape[1]
        weights_grad[L] = delta[L] @ Z[L].T
        bias_grad[L] = np.sum(delta[L], axis=1, keepdims=True)

        for l in range(L-1, -1, -1):
            delta[l] = (self.layer_list[l+1].weights.T @ delta[l+1]) * self.relu_derivative(A[l])
            weights_grad[l] = delta[l] @ Z[l].T
            bias_grad[l] = np.sum(delta[l], axis=1, keepdims=True)
        
        return loss, weights_grad, bias_grad

    def fit(self, X, y, learning_rate, epochs, X_val=None, y_val=None):
        loss_history = []
        val_loss_history = []
        m = X.shape[1]
        for epoch in range(epochs):
            loss, weights_grad, bias_grad = self.back_propagation(X, y)
            for l in range(self.n_layers):
                self.layer_list[l].weights -= (learning_rate / m) * weights_grad[l]
                self.layer_list[l].bias -= (learning_rate / m) * bias_grad[l]
            loss_history.append(loss)
            
            if X_val is not None and y_val is not None:
                Z_val, _ = self.forward_pass(X_val)
                val_loss = self.cross_entropy(y_val, Z_val[-1])
                val_loss_history.append(val_loss)
        
        if X_val is not None and y_val is not None:
            return loss_history, val_loss_history
        return loss_history

    def predict(self, X):
        Z, A = self.forward_pass(X)
        return Z[-1]
    
