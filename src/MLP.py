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
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
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

    def cross_entropy(self, y_true, y_pred, lambda_l2=0.0):
        epsilon = 1e-12
        m = y_true.shape[1]
        loss = - (1/m) * np.sum(y_true * np.log(y_pred + epsilon))

        # Agrego el término de regularización L2
        if lambda_l2 > 0.0:
            l2_reg = sum(np.sum(np.square(layer.weights)) for layer in self.layer_list)
            loss += (lambda_l2 / (2 * m)) * l2_reg
        
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

    def back_propagation(self, X, y, lambda_l2=0.0): 
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
        weights_grad[L] = delta[L] @ Z[L].T + lambda_l2 * self.layer_list[L].weights
        bias_grad[L] = np.sum(delta[L], axis=1, keepdims=True)

        for l in range(L-1, -1, -1):
            delta[l] = (self.layer_list[l+1].weights.T @ delta[l+1]) * self.relu_derivative(A[l])
            weights_grad[l] = delta[l] @ Z[l].T + lambda_l2 * self.layer_list[l].weights
            bias_grad[l] = np.sum(delta[l], axis=1, keepdims=True)
        
        return loss, weights_grad, bias_grad

    def fit(self, X, y, eta_0, epochs, X_val=None, y_val=None,
            lr_schedule=None, K=100, eta_K=0.001, s=10.0, c=0.95, batch_size=None,
            optimizer="gd", beta1=0.9, beta2=0.999, epsilon=1e-8,
            lambda_l2=0.0, early_stopping=False, patience=10, min_delta=1e-4):

        loss_history = []
        val_loss_history = []
        m = X.shape[1]

        # Si batch_size es None, se usa todo el dataset (Batch GD)
        if batch_size is None:
            batch_size = m

        if optimizer == "adam":
            m_W = [np.zeros_like(layer.weights) for layer in self.layer_list]
            v_W = [np.zeros_like(layer.weights) for layer in self.layer_list]
            m_b = [np.zeros_like(layer.bias) for layer in self.layer_list]
            v_b = [np.zeros_like(layer.bias) for layer in self.layer_list]
            t = 0  # contador para el bias correction
        
        # Variables para early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None

        for k in range(epochs):
            if lr_schedule == "lineal":
                if k < K:
                    eta_k = (1 - k/K) * eta_0 + (k/K) * eta_K
                else:
                    eta_k = eta_K  # Evitar saturacion del rate
                
            elif lr_schedule == "exponencial":
                eta_k = eta_0 * (c ** (k/s))

            else:
                # Constante (GD Standard)
                eta_k = eta_0
            
            # Mezclo los datos al inicio de cada epoch para que los mini-batches sean lo mas representativos posible
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]

            for i in range(0, m, batch_size):
                # Mini batch actual
                X_batch = X_shuffled[:, i:i+batch_size]
                y_batch = y_shuffled[:, i:i+batch_size]
                
                # cantidad real de muestras en este batch, ya que el ultimo puede ser mas chico
                batch_m = X_batch.shape[1]

                _, weights_grad, bias_grad = self.back_propagation(X_batch, y_batch, lambda_l2)
                
                # actualizo pesos dividiendo por el tamaño del batch actual
                if optimizer == "adam":
                    t += 1 
                    
                    for l in range(self.n_layers):
                        # gradiente promedio del batch
                        g_W = weights_grad[l] / batch_m
                        g_b = bias_grad[l] / batch_m
                        
                        # actualizo estimaciones del 1er momento (momentum)
                        m_W[l] = beta1 * m_W[l] + (1 - beta1) * g_W
                        m_b[l] = beta1 * m_b[l] + (1 - beta1) * g_b
                        
                        # actualizo estimaciones del 2do momento (rmsprop)
                        v_W[l] = beta2 * v_W[l] + (1 - beta2) * (g_W ** 2)
                        v_b[l] = beta2 * v_b[l] + (1 - beta2) * (g_b ** 2)
                        
                        # bias correction
                        m_hat_W = m_W[l] / (1 - beta1 ** t)
                        m_hat_b = m_b[l] / (1 - beta1 ** t)
                        v_hat_W = v_W[l] / (1 - beta2 ** t)
                        v_hat_b = v_b[l] / (1 - beta2 ** t)
                        
                        # actualizo pesos y sesgos con la fórmula de ADAM
                        self.layer_list[l].weights -= eta_k * m_hat_W / (np.sqrt(v_hat_W) + epsilon)
                        self.layer_list[l].bias -= eta_k * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
                else:
                    # GD Estandar 
                    for l in range(self.n_layers):
                        self.layer_list[l].weights -= (eta_k / batch_m) * weights_grad[l]
                        self.layer_list[l].bias -= (eta_k / batch_m) * bias_grad[l]
            
            Z_train, _ = self.forward_pass(X)
            epoch_loss = self.cross_entropy(y, Z_train[-1])
            loss_history.append(epoch_loss)
            
            if X_val is not None and y_val is not None:
                Z_val, _ = self.forward_pass(X_val)
                val_loss = self.cross_entropy(y_val, Z_val[-1])
                val_loss_history.append(val_loss)
                
                # Early Stopping
                if early_stopping:
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # guardo los pesos
                        best_weights = [np.copy(layer.weights) for layer in self.layer_list]
                        best_biases = [np.copy(layer.bias) for layer in self.layer_list]
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            # Restauro los pesos
                            for l, layer in enumerate(self.layer_list):
                                layer.weights = best_weights[l]
                                layer.bias = best_biases[l]
                            print(f"Early Stopping en epoch {k+1}!")
                            break
        
        if X_val is not None and y_val is not None:
            return loss_history, val_loss_history
        return loss_history

    def predict(self, X):
        Z, A = self.forward_pass(X)
        return Z[-1]
    
