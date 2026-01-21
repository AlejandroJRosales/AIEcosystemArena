import random
import math
import numpy as np
from abc import ABC, abstractmethod
from utils import tools


class Network(ABC):
    """Abstract base class for neural networks"""
    
    def __init__(self, layer_dims, learning_rate=10, momentum=0.9, weight_decay=0.0001):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cost = 0.0
        self.cost_history = []
    
    @staticmethod
    def sigmoid(x):
        """Numerically stable sigmoid"""
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
    
    @staticmethod
    def softmax(vector):
        """Numerically stable softmax"""
        v_array = np.array(vector)
        e = np.exp(v_array - np.max(v_array))
        return e / e.sum()
    
    @staticmethod
    def xavier_init(in_dim, out_dim):
        """Xavier/Glorot initialization"""
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        return [[random.uniform(-limit, limit) for _ in range(out_dim)] for _ in range(in_dim)]
    
    @staticmethod
    def matvec(mat, vec):
        """Matrix-vector multiplication"""
        return [sum(m_i * v_i for m_i, v_i in zip(m_row, vec)) for m_row in mat]
    
    def map_input(self, agent):
        """Map agent state to network input"""
        health_diff = agent.health - agent.start_health
        x_relative = abs(agent.x - agent.coords_focused.x) % agent.world.world_width / agent.world.world_width
        y_relative = abs(agent.y - agent.coords_focused.y) % agent.world.world_height / agent.world.world_height
        sign = -1 if agent.priority == "predator" else 1
        
        self.cost = health_diff
        return [x_relative, y_relative, sign, agent.food_need, agent.water_need, agent.reproduction_need]
    
    def think(self, agent):
        """Forward pass and weight update"""
        mapped_input = self.map_input(agent)
        values = list(self.propagate(mapped_input))
        self.output = max(range(len(values)), key=values.__getitem__)
        self.adjust_weights()
        return self.output
    
    @abstractmethod
    def propagate(self, inputs):
        """Forward pass - implemented by subclasses"""
        pass
    
    @abstractmethod
    def adjust_weights(self):
        """Weight update - implemented by subclasses"""
        pass
    
    def _get_adaptive_lr(self):
        """Calculate adaptive learning rate"""
        if self.cost_history and self.cost < self.cost_history[-1]:
            adaptive_lr = self.learning_rate * 1.05
        else:
            adaptive_lr = self.learning_rate * 0.95
        
        adaptive_lr = max(0.0001, min(0.1, adaptive_lr))
        self.cost_history.append(self.cost)
        if len(self.cost_history) > 100:
            self.cost_history.pop(0)
        return adaptive_lr
    
    def _blend_parent_weights(self, parent1, parent2, mutation_multi=0.5):
        """Common logic for blending parent weights"""
        lower_bound = max(1 - mutation_multi, 0.1)
        upper_bound = min(1 + mutation_multi, 4.0)
        mutation_factor = random.uniform(lower_bound, upper_bound)
        
        self.learning_rate = ((parent1.learning_rate + parent2.learning_rate) / 2) * random.uniform(0.5, 1.5)
        
        return mutation_factor


class DenseNetwork(Network):
    """Feedforward neural network with dense layers"""
    
    def __init__(self, animal, learning_rate=10, momentum=0.9, weight_decay=0.0001):
        layer_dims = animal.nn_layer_dims
        super().__init__(layer_dims, learning_rate, momentum, weight_decay)
        
        self.weights = [self.xavier_init(layer_dims[i], layer_dims[i + 1]) 
                       for i in range(len(layer_dims) - 1)]
        self.biases = [[0.0] * layer_dims[i + 1] for i in range(len(layer_dims) - 1)]
        
        self.weight_velocity = [[[0.0 for _ in row] for row in layer] for layer in self.weights]
        self.bias_velocity = [[0.0 for _ in bias] for bias in self.biases]
    
    def propagate(self, inputs):
        """Forward pass through network"""
        if len(inputs) != self.layer_dims[0]:
            raise ValueError("Input size does not match network input layer size.")
        
        sliding_layer = inputs
        for l_idx in range(len(self.layer_dims) - 1):
            out_layer = [self.biases[l_idx][n2] for n2 in range(self.layer_dims[l_idx + 1])]
            
            for n1_idx, node in enumerate(sliding_layer):
                for n2_idx in range(self.layer_dims[l_idx + 1]):
                    out_layer[n2_idx] += node * self.weights[l_idx][n1_idx][n2_idx]
            
            # Apply activation (except output layer)
            if l_idx < len(self.layer_dims) - 2:
                sliding_layer = [self.sigmoid(v) for v in out_layer]
            else:
                sliding_layer = out_layer
        
        return self.softmax(sliding_layer)
    
    def adjust_weights(self):
        """Update weights with momentum and adaptive learning rate"""
        normalized_cost = max(-1.0, min(1.0, self.cost))
        adaptive_lr = self._get_adaptive_lr()
        
        for l_idx in range(len(self.weights)):
            for n1_idx in range(len(self.weights[l_idx])):
                for n2_idx in range(len(self.weights[l_idx][n1_idx])):
                    velocity = self.momentum * self.weight_velocity[l_idx][n1_idx][n2_idx] - adaptive_lr * normalized_cost
                    self.weight_velocity[l_idx][n1_idx][n2_idx] = velocity
                    self.weights[l_idx][n1_idx][n2_idx] = self.weights[l_idx][n1_idx][n2_idx] * (1 - self.weight_decay) + velocity
        
        for l_idx in range(len(self.biases)):
            for b_idx in range(len(self.biases[l_idx])):
                velocity = self.momentum * self.bias_velocity[l_idx][b_idx] - adaptive_lr * 0.1 * normalized_cost
                self.bias_velocity[l_idx][b_idx] = velocity
                self.biases[l_idx][b_idx] = self.biases[l_idx][b_idx] * (1 - self.weight_decay * 0.1) + velocity
    
    def crossover(self, parent1, parent2, mutation_multi=0.5):
        """Blend weights from two parents"""
        mutation_factor = self._blend_parent_weights(parent1, parent2, mutation_multi)
        
        # Blend weights
        for layer_idx in range(len(self.weights)):
            for i in range(len(self.weights[layer_idx])):
                for j in range(len(self.weights[layer_idx][i])):
                    w1 = self._safe_get_weight(parent1, 'weights', layer_idx, i, j)
                    w2 = self._safe_get_weight(parent2, 'weights', layer_idx, i, j)
                    self.weights[layer_idx][i][j] = (w1 + w2) / 2 * mutation_factor
        
        # Blend biases
        for layer_idx in range(len(self.biases)):
            for i in range(len(self.biases[layer_idx])):
                b1 = self._safe_get_bias(parent1, layer_idx, i)
                b2 = self._safe_get_bias(parent2, layer_idx, i)
                self.biases[layer_idx][i] = (b1 + b2) / 2 * mutation_factor
        
        # Reset velocity
        self.weight_velocity = [[[0.0 for _ in row] for row in layer] for layer in self.weights]
        self.bias_velocity = [[0.0 for _ in bias] for bias in self.biases]
    
    @staticmethod
    def _safe_get_weight(network, attr, layer_idx, i, j):
        """Safely retrieve weight from network"""
        try:
            if hasattr(network, attr):
                weights = getattr(network, attr)
                if len(weights) > layer_idx and isinstance(weights[0], list):
                    return weights[layer_idx][i][j]
        except (IndexError, TypeError):
            pass
        return 0.0
    
    @staticmethod
    def _safe_get_bias(network, layer_idx, i):
        """Safely retrieve bias from network"""
        try:
            if hasattr(network, 'biases') and len(network.biases) > layer_idx:
                return network.biases[layer_idx][i]
        except (IndexError, TypeError):
            pass
        return 0.0


class LSTMNetwork(Network):
    """LSTM neural network"""
    
    def __init__(self, animal, learning_rate=10, momentum=0.9, weight_decay=0.0001):
        layer_dims = animal.nn_layer_dims
        super().__init__(layer_dims, learning_rate, momentum, weight_decay)
        
        self.weights = [self._init_lstm_layer(layer_dims[i], layer_dims[i + 1]) 
                       for i in range(len(layer_dims) - 2)]
        
        last_hidden_dim = layer_dims[-2]
        output_dim = layer_dims[-1]
        self.V = self.xavier_init(last_hidden_dim, output_dim)
        self.output_bias = [0.0] * output_dim
        
        self.velocity = self._init_velocity()
    
    @staticmethod
    def orthogonal_init(dim):
        """Orthogonal initialization for recurrent weights"""
        mat = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(dim)]
        for _ in range(10):
            for i in range(dim):
                norm = math.sqrt(sum(mat[i][j]**2 for j in range(dim)))
                if norm > 1e-10:
                    for j in range(dim):
                        mat[i][j] /= norm
        return mat
    
    @staticmethod
    def tanh(x):
        return math.tanh(x)
    
    def _init_lstm_layer(self, input_dim, hidden_dim):
        """Initialize single LSTM layer"""
        return {
            'W_i': self.xavier_init(input_dim, hidden_dim),
            'W_f': self.xavier_init(input_dim, hidden_dim),
            'W_c': self.xavier_init(input_dim, hidden_dim),
            'W_o': self.xavier_init(input_dim, hidden_dim),
            'U_i': self.orthogonal_init(hidden_dim),
            'U_f': self.orthogonal_init(hidden_dim),
            'U_c': self.orthogonal_init(hidden_dim),
            'U_o': self.orthogonal_init(hidden_dim),
            'b_i': [0.0] * hidden_dim,
            'b_f': [0.1] * hidden_dim,
            'b_c': [0.0] * hidden_dim,
            'b_o': [0.0] * hidden_dim,
            'c': [0.0] * hidden_dim,
            'h': [0.0] * hidden_dim
        }
    
    def _init_velocity(self):
        """Initialize velocity for momentum"""
        return [{key: [[0.0 for _ in row] for row in layer[key]] 
                 for key in ['W_i', 'W_f', 'W_c', 'W_o', 'U_i', 'U_f', 'U_c', 'U_o']}
                for layer in self.weights]
    
    def propagate(self, inputs):
        """Forward pass through LSTM layers"""
        if len(inputs) != self.layer_dims[0]:
            raise ValueError("Input size does not match network input layer size.")
        
        sliding_input = inputs
        
        for layer in self.weights:
            # Compute gates
            i = [self.sigmoid(x + y + b) for x, y, b in zip(
                self.matvec(layer['W_i'], sliding_input),
                self.matvec(layer['U_i'], layer['h']),
                layer['b_i'])]
            f = [self.sigmoid(x + y + b) for x, y, b in zip(
                self.matvec(layer['W_f'], sliding_input),
                self.matvec(layer['U_f'], layer['h']),
                layer['b_f'])]
            g = [self.tanh(x + y + b) for x, y, b in zip(
                self.matvec(layer['W_c'], sliding_input),
                self.matvec(layer['U_c'], layer['h']),
                layer['b_c'])]
            o = [self.sigmoid(x + y + b) for x, y, b in zip(
                self.matvec(layer['W_o'], sliding_input),
                self.matvec(layer['U_o'], layer['h']),
                layer['b_o'])]
            
            # Update cell and hidden state
            layer['c'] = [f_j * c_j + i_j * g_j for f_j, c_j, i_j, g_j in zip(f, layer['c'], i, g)]
            layer['h'] = [o_j * self.tanh(c_j) for o_j, c_j in zip(o, layer['c'])]
            
            sliding_input = layer['h']
        
        # Output layer
        logits = [l + b for l, b in zip(self.matvec(self.V, sliding_input), self.output_bias)]
        return self.softmax(logits)
    
    def adjust_weights(self):
        """Update LSTM weights with momentum and adaptive learning rate"""
        normalized_cost = max(-1.0, min(1.0, self.cost))
        adaptive_lr = self._get_adaptive_lr()
        
        def update_matrix(mat, velocity_mat, is_recurrent=False):
            scale = 0.5 if is_recurrent else 1.0
            gradient = normalized_cost * scale
            
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    velocity_mat[i][j] = self.momentum * velocity_mat[i][j] - adaptive_lr * gradient
                    mat[i][j] = mat[i][j] * (1 - self.weight_decay) + velocity_mat[i][j]
        
        # Update weights
        for layer_idx, layer in enumerate(self.weights):
            for key in ['W_i', 'W_f', 'W_c', 'W_o']:
                update_matrix(layer[key], self.velocity[layer_idx][key])
            
            for key in ['U_i', 'U_f', 'U_c', 'U_o']:
                update_matrix(layer[key], self.velocity[layer_idx][key], is_recurrent=True)
            
            # Update biases
            for bias_key in ['b_i', 'b_f', 'b_c', 'b_o']:
                layer[bias_key] = [b * (1 - self.weight_decay * 0.1) + (normalized_cost * 0.01) 
                                   for b in layer[bias_key]]
        
        # Update output layer
        for i in range(len(self.V)):
            for j in range(len(self.V[i])):
                v_velocity = getattr(self, 'V_velocity', None)
                if v_velocity is None:
                    self.V_velocity = [[0.0 for _ in row] for row in self.V]
                    v_velocity = self.V_velocity
                
                v_velocity[i][j] = self.momentum * v_velocity[i][j] - adaptive_lr * normalized_cost
                self.V[i][j] = self.V[i][j] * (1 - self.weight_decay) + v_velocity[i][j]
        
        self.output_bias = [b * (1 - self.weight_decay * 0.1) + (normalized_cost * 0.01) 
                           for b in self.output_bias]
    
    def crossover(self, parent1, parent2, mutation_multi=0.2):
        """Blend weights from two parents"""
        mutation_factor = self._blend_parent_weights(parent1, parent2, mutation_multi)
        
        weight_keys = ['W_i', 'W_f', 'W_c', 'W_o', 'U_i', 'U_f', 'U_c', 'U_o']
        bias_keys = ['b_i', 'b_f', 'b_c', 'b_o']
        
        # Blend LSTM weights
        for layer_idx in range(len(self.weights)):
            for key in weight_keys:
                for i in range(len(self.weights[layer_idx][key])):
                    for j in range(len(self.weights[layer_idx][key][i])):
                        w1 = self._safe_get_lstm_weight(parent1, layer_idx, key, i, j)
                        w2 = self._safe_get_lstm_weight(parent2, layer_idx, key, i, j)
                        self.weights[layer_idx][key][i][j] = (w1 + w2) / 2 * mutation_factor
            
            # Blend biases
            for key in bias_keys:
                for i in range(len(self.weights[layer_idx][key])):
                    b1 = self._safe_get_lstm_bias(parent1, layer_idx, key, i)
                    b2 = self._safe_get_lstm_bias(parent2, layer_idx, key, i)
                    self.weights[layer_idx][key][i] = (b1 + b2) / 2 * mutation_factor
        
        # Blend output layer
        for i in range(len(self.V)):
            for j in range(len(self.V[i])):
                v1 = parent1.V[i][j] if hasattr(parent1, 'V') else 0.0
                v2 = parent2.V[i][j] if hasattr(parent2, 'V') else 0.0
                self.V[i][j] = (v1 + v2) / 2 * mutation_factor
        
        for i in range(len(self.output_bias)):
            ob1 = parent1.output_bias[i] if hasattr(parent1, 'output_bias') else 0.0
            ob2 = parent2.output_bias[i] if hasattr(parent2, 'output_bias') else 0.0
            self.output_bias[i] = (ob1 + ob2) / 2 * mutation_factor
        
        self.velocity = self._init_velocity()
    
    @staticmethod
    def _safe_get_lstm_weight(network, layer_idx, key, i, j):
        """Safely retrieve LSTM weight"""
        try:
            if hasattr(network, 'weights') and len(network.weights) > layer_idx:
                if isinstance(network.weights[0], dict) and key in network.weights[layer_idx]:
                    return network.weights[layer_idx][key][i][j]
        except (IndexError, TypeError, KeyError):
            pass
        return 0.0
    
    @staticmethod
    def _safe_get_lstm_bias(network, layer_idx, key, i):
        """Safely retrieve LSTM bias"""
        try:
            if hasattr(network, 'weights') and len(network.weights) > layer_idx:
                if isinstance(network.weights[0], dict) and key in network.weights[layer_idx]:
                    return network.weights[layer_idx][key][i]
        except (IndexError, TypeError, KeyError):
            pass
        return 0.0