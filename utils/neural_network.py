import random
import math
import numpy as np
from utils import tools


class LSTMNetwork:
    def __init__(self, animal):
        self.layer_dims = animal.nn_layer_dims  # Example: [6, 10, 8, output_dim]
        
        # Create weights for each LSTM layer
        self.weights = []
        for l_idx in range(len(self.layer_dims) - 2):  # all except output layer
            input_dim = self.layer_dims[l_idx]
            hidden_dim = self.layer_dims[l_idx + 1]
            layer = {
                'W_i': self.xavier_init(input_dim, hidden_dim),
                'W_f': self.xavier_init(input_dim, hidden_dim),
                'W_c': self.xavier_init(input_dim, hidden_dim),
                'W_o': self.xavier_init(input_dim, hidden_dim),
                'U_i': self.orthogonal_init(hidden_dim),
                'U_f': self.orthogonal_init(hidden_dim),
                'U_c': self.orthogonal_init(hidden_dim),
                'U_o': self.orthogonal_init(hidden_dim),
                'b_i': [0.0] * hidden_dim,
                'b_f': [0.1] * hidden_dim,  # forget bias initialized to 0.1
                'b_c': [0.0] * hidden_dim,
                'b_o': [0.0] * hidden_dim,
                'c': [0.0] * hidden_dim,  # initial cell state
                'h': [0.0] * hidden_dim   # initial hidden state
            }
            self.weights.append(layer)

        # Output layer: last hidden to output
        last_hidden_dim = self.layer_dims[-2]
        output_dim = self.layer_dims[-1]
        self.V = self.xavier_init(last_hidden_dim, output_dim)
        self.output_bias = [0.0] * output_dim

        # Learning rate and momentum for weight updates
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.0001
        
        # Store previous weight deltas for momentum
        self.velocity = self._init_velocity()
        
        # Cost tracking for adaptive learning
        self.cost = 0.0
        self.prev_cost = 0.0
        self.cost_history = []

    def _init_velocity(self):
        """Initialize velocity matrices for momentum"""
        velocity = []
        for layer in self.weights:
            v_layer = {}
            for key in ['W_i', 'W_f', 'W_c', 'W_o', 'U_i', 'U_f', 'U_c', 'U_o']:
                v_layer[key] = [[0.0 for _ in row] for row in layer[key]]
            velocity.append(v_layer)
        return velocity

    @staticmethod
    def xavier_init(in_dim, out_dim):
        """Xavier/Glorot initialization for better convergence"""
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        return [[random.uniform(-limit, limit) for _ in range(out_dim)] for _ in range(in_dim)]

    @staticmethod
    def orthogonal_init(dim):
        """Orthogonal initialization for recurrent weights"""
        mat = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(dim)]
        # Simple QR decomposition approximation
        for _ in range(10):
            for i in range(dim):
                norm = math.sqrt(sum(mat[i][j]**2 for j in range(dim)))
                if norm > 1e-10:
                    for j in range(dim):
                        mat[i][j] /= norm
        return mat

    @staticmethod
    def sigmoid(x):
        """Numerically stable sigmoid"""
        # Clamp to prevent overflow
        x = max(-500, min(500, x))
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    @staticmethod
    def tanh(x):
        return math.tanh(x)

    @staticmethod
    def softmax(vector):
        """Numerically stable softmax"""
        v_array = np.array(vector)
        e = np.exp(v_array - np.max(v_array))
        return e / e.sum()

    def matvec(self, mat, vec):
        return [sum(m_i * v_i for m_i, v_i in zip(m_row, vec)) for m_row in mat]

    def propagate(self, inputs):
        if len(inputs) != self.layer_dims[0]:
            raise ValueError("Input size does not match network input layer size.")

        sliding_input = inputs

        for layer in self.weights:
            # Input gate
            i = [self.sigmoid(x + y + b) for x, y, b in zip(
                self.matvec(layer['W_i'], sliding_input), 
                self.matvec(layer['U_i'], layer['h']), 
                layer['b_i'])]
            # Forget gate
            f = [self.sigmoid(x + y + b) for x, y, b in zip(
                self.matvec(layer['W_f'], sliding_input), 
                self.matvec(layer['U_f'], layer['h']), 
                layer['b_f'])]
            # Cell candidate
            g = [self.tanh(x + y + b) for x, y, b in zip(
                self.matvec(layer['W_c'], sliding_input), 
                self.matvec(layer['U_c'], layer['h']), 
                layer['b_c'])]
            # Output gate
            o = [self.sigmoid(x + y + b) for x, y, b in zip(
                self.matvec(layer['W_o'], sliding_input), 
                self.matvec(layer['U_o'], layer['h']), 
                layer['b_o'])]

            # Update cell state and hidden state
            layer['c'] = [f_j * c_j + i_j * g_j for f_j, c_j, i_j, g_j in zip(f, layer['c'], i, g)]
            layer['h'] = [o_j * self.tanh(c_j) for o_j, c_j in zip(o, layer['c'])]

            sliding_input = layer['h']

        # Final output
        logits = self.matvec(self.V, sliding_input)
        logits = [l + b for l, b in zip(logits, self.output_bias)]
        return self.softmax(logits)

    def adjust_weights(self):
        """Adaptive learning with momentum and weight decay"""
        self.prev_cost = self.cost
        
        # Normalize cost to [-1, 1] range for stable learning rates
        normalized_cost = max(-1.0, min(1.0, self.cost))
        
        # Adaptive learning rate: increase learning when loss improves, decrease otherwise
        if len(self.cost_history) > 0 and self.cost < self.cost_history[-1]:
            adaptive_lr = self.learning_rate * 1.05
        else:
            adaptive_lr = self.learning_rate * 0.95
        
        adaptive_lr = max(0.0001, min(0.1, adaptive_lr))  # Clamp learning rate
        
        self.cost_history.append(self.cost)
        if len(self.cost_history) > 100:
            self.cost_history.pop(0)

        def update_matrix(mat, velocity_mat, is_recurrent=False):
            """Update matrix with momentum and weight decay"""
            new_mat = []
            for i, row in enumerate(mat):
                new_row = []
                for j, weight in enumerate(row):
                    # Compute gradient: positive cost increases weight, negative decreases it
                    # Recurrent weights use smaller updates
                    scale = 0.5 if is_recurrent else 1.0
                    gradient = normalized_cost * scale
                    
                    # Momentum update
                    velocity_mat[i][j] = (self.momentum * velocity_mat[i][j] - 
                                         adaptive_lr * gradient)
                    
                    # Apply weight decay (L2 regularization)
                    new_weight = weight * (1 - self.weight_decay) + velocity_mat[i][j]
                    new_row.append(new_weight)
                new_mat.append(new_row)
            return new_mat

        # Create velocity matrix for output layer if needed
        if not hasattr(self, 'V_velocity'):
            self.V_velocity = [[0.0 for _ in row] for row in self.V]
        
        for layer_idx, layer in enumerate(self.weights):
            for key in ['W_i', 'W_f', 'W_c', 'W_o']:
                layer[key] = update_matrix(layer[key], self.velocity[layer_idx][key])
            
            for key in ['U_i', 'U_f', 'U_c', 'U_o']:
                layer[key] = update_matrix(layer[key], self.velocity[layer_idx][key], is_recurrent=True)
            
            # Update biases with smaller learning rate
            for bias_key in ['b_i', 'b_f', 'b_c', 'b_o']:
                layer[bias_key] = [b * (1 - self.weight_decay * 0.1) + 
                                  (normalized_cost * 0.01) for b in layer[bias_key]]

        self.V = update_matrix(self.V, self.V_velocity)
        self.output_bias = [b * (1 - self.weight_decay * 0.1) + (normalized_cost * 0.01) for b in self.output_bias]

    def map_input(self, self_agent):
        health_diff = self_agent.health - self_agent.start_health
        x_relative = abs(self_agent.x - self_agent.coords_focused.x) % self_agent.world.world_width / self_agent.world.world_width
        y_relative = abs(self_agent.y - self_agent.coords_focused.y) % self_agent.world.world_height / self_agent.world.world_height
        self.cost = health_diff
        sign = -1 if self_agent.priority == "predator" else 1
        return [x_relative, y_relative, sign, self_agent.food_need, self_agent.water_need, self_agent.reproduction_need]

    def think(self, self_agent):
        mapped_input = self.map_input(self_agent)
        values = list(self.propagate(mapped_input))
        self.output = max(range(len(values)), key=values.__getitem__)
        self.adjust_weights()
        return self.output


class DenseNetwork:
    def __init__(self, animal):
        self.layer_dims = animal.nn_layer_dims
        
        # Xavier initialization
        self.weights = []
        for l_idx in range(len(self.layer_dims) - 1):
            layer_weights = self.xavier_init(self.layer_dims[l_idx], self.layer_dims[l_idx + 1])
            self.weights.append(layer_weights)
        
        # Initialize biases
        self.biases = [[0.0] * self.layer_dims[l_idx + 1] for l_idx in range(len(self.layer_dims) - 1)]
        
        # Learning parameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.0001
        
        # Velocity for momentum
        self.weight_velocity = [[[0.0 for _ in row] for row in layer] for layer in self.weights]
        self.bias_velocity = [[0.0 for _ in bias] for bias in self.biases]
        
        # Cost tracking
        self.cost = 0.0
        self.cost_history = []

    @staticmethod
    def xavier_init(in_dim, out_dim):
        """Xavier/Glorot initialization"""
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        return [[random.uniform(-limit, limit) for _ in range(out_dim)] for _ in range(in_dim)]

    @staticmethod
    def sigmoid(x):
        """Numerically stable sigmoid"""
        x = max(-500, min(500, x))
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    @staticmethod
    def softmax(vector):
        """Numerically stable softmax"""
        v_array = np.array(vector)
        e = np.exp(v_array - np.max(v_array))
        return e / e.sum()

    def propagate(self, inputs):
        if len(inputs) != self.layer_dims[0]:
            raise ValueError("Input size does not match network input layer size.")

        sliding_layer = inputs
        for l_idx in range(len(self.layer_dims) - 1):
            out_layer = [0.0] * self.layer_dims[l_idx + 1]
            for n1_idx in range(self.layer_dims[l_idx]):
                for n2_idx in range(self.layer_dims[l_idx + 1]):
                    current_node = sliding_layer[n1_idx]
                    weight = self.weights[l_idx][n1_idx][n2_idx]
                    out_layer[n2_idx] += current_node * weight
            
            # Add biases
            out_layer = [out_layer[i] + self.biases[l_idx][i] for i in range(len(out_layer))]
            
            # Apply activation between layers (except output)
            if l_idx < len(self.layer_dims) - 2:
                sliding_layer = [self.sigmoid(v) for v in out_layer]
            else:
                sliding_layer = out_layer
        
        return self.softmax(sliding_layer)

    def adjust_weights(self):
        """Adaptive learning with momentum and weight decay"""
        normalized_cost = max(-1.0, min(1.0, self.cost))
        
        # Adaptive learning rate
        if len(self.cost_history) > 0 and self.cost < self.cost_history[-1]:
            adaptive_lr = self.learning_rate * 1.05
        else:
            adaptive_lr = self.learning_rate * 0.95
        
        adaptive_lr = max(0.0001, min(0.1, adaptive_lr))
        self.cost_history.append(self.cost)
        if len(self.cost_history) > 100:
            self.cost_history.pop(0)

        # Update weights with momentum
        for l_idx in range(len(self.weights)):
            for n1_idx in range(len(self.weights[l_idx])):
                for n2_idx in range(len(self.weights[l_idx][n1_idx])):
                    gradient = normalized_cost
                    
                    # Momentum update
                    self.weight_velocity[l_idx][n1_idx][n2_idx] = (
                        self.momentum * self.weight_velocity[l_idx][n1_idx][n2_idx] -
                        adaptive_lr * gradient
                    )
                    
                    # Apply weight decay and update
                    self.weights[l_idx][n1_idx][n2_idx] = (
                        self.weights[l_idx][n1_idx][n2_idx] * (1 - self.weight_decay) +
                        self.weight_velocity[l_idx][n1_idx][n2_idx]
                    )

        # Update biases with smaller learning rate
        for l_idx in range(len(self.biases)):
            for b_idx in range(len(self.biases[l_idx])):
                self.bias_velocity[l_idx][b_idx] = (
                    self.momentum * self.bias_velocity[l_idx][b_idx] -
                    adaptive_lr * 0.1 * normalized_cost
                )
                self.biases[l_idx][b_idx] = (
                    self.biases[l_idx][b_idx] * (1 - self.weight_decay * 0.1) +
                    self.bias_velocity[l_idx][b_idx]
                )

    def map_input(self, self_agent):
        health_diff = self_agent.health - self_agent.start_health
        x_relative = abs(self_agent.x - self_agent.coords_focused.x) % self_agent.world.world_width / self_agent.world.world_width
        y_relative = abs(self_agent.y - self_agent.coords_focused.y) % self_agent.world.world_height / self_agent.world.world_height
        self.cost = health_diff
        sign = -1 if self_agent.priority == "predator" else 1
        return [x_relative, y_relative, sign, self_agent.food_need, self_agent.water_need, self_agent.reproduction_need]

    def think(self, self_agent):
        mapped_input = self.map_input(self_agent)
        values = list(self.propagate(mapped_input))
        self.output = max(range(len(values)), key=values.__getitem__)
        self.adjust_weights()
        return self.output