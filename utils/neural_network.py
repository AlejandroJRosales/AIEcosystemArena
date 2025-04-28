
import random
import math
import numpy as np
from utils import tools


# neural evolution augmenting topologies
class NEAT:
    pass

class DenseNetwork:
    def __init__(self, animal):
        # TODO: tune hyperparameters
        self.layer_dims = animal.weights_layers
        self.weights = [[[random.uniform(-1, 1) for weight in range(self.layer_dims[l_idx + 1])] for node in range(self.layer_dims[l_idx])] for l_idx in range(len(self.layer_dims) - 1)]
        # self.weights.append([[random.uniform(-1, 1) for weight in range(self.layer_dims[-2])] for node in range(self.layer_dims[-1])])
        # print([len(l) for l in self.weights])
        self.min_update = 0.9998
        self.max_update = 1.0002

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    @staticmethod
    def softmax(vector):
        e = np.exp(vector - np.max(vector))
        return e / e.sum()

    def propagate(self, inputs):
            if len(inputs) != self.layer_dims[0]:
                raise ValueError("Input size does not match network input layer size.")

            sliding_layer = inputs
            for l_idx in range(len(self.layer_dims) - 1):
                out_layer = [0] * self.layer_dims[l_idx + 1]
                for n1_idx in range(self.layer_dims[l_idx]):
                    for n2_idx in range(self.layer_dims[l_idx + 1]):
                        current_node = sliding_layer[n1_idx]
                        weight = self.weights[l_idx][n1_idx][n2_idx]
                        out_layer[n2_idx] += current_node * weight
                # Add activation between layers (except output if softmax)
                if l_idx < len(self.layer_dims) - 2:
                    sliding_layer = [self.sigmoid(v) for v in out_layer]
                else:
                    sliding_layer = out_layer
            return self.softmax(sliding_layer)

    def adjust_weights(self):
        alpha = self.sigmoid(self.cost) * .0001
        loc_min_update, loc_max_update = self.min_update + alpha, self.max_update + alpha
        # print(loc_min_update, loc_max_update)
        self.weights = [[[weight * random.uniform(loc_min_update, loc_max_update) for weight in node] for node in layer] for layer in self.weights]

    def map_input(self, self_agent):
        health_diff = self_agent.health - self_agent.start_health
        x_relative = abs(self_agent.x - self_agent.coords_focused.x) % self_agent.world.world_width / self_agent.world.world_width
        y_relative = abs(self_agent.y - self_agent.coords_focused.y) % self_agent.world.world_height / self_agent.world.world_height
        self.cost = health_diff
        sign = -1 if self_agent.priority == "predator" else 1
        return [x_relative, y_relative, sign, self_agent.food_need, self_agent.water_need, self_agent.reproduction_need]

    def think(self, self_agent):
        mapped_input = self.map_input(self_agent)
        # inputs = [self.distance_formula(self.focused_obj.x, self.focused_obj.y), self.focused_obj.x, self.focused_obj.y, self.x, self.y]
        values = list(self.propagate(mapped_input))
        self.output = max(range(len(values)), key=values.__getitem__)
        self.adjust_weights()
        # print(self.weights)
        # might need multiple weightss for multiple outputs?
        return self.output