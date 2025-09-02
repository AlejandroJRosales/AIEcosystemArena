import pygame
import numpy as np

class MyScene:
    def __init__(self, window, screen_size):
        self.window = window
        self.w_width = screen_size[0]
        self.w_height = screen_size[1]
        self.node_size = 15

    def draw(self, screen, nn, marked_node_idx):
        font = pygame.font.SysFont('arial', 36)
        if hasattr(nn, 'weights'):
            net_type = "DenseNetwork"
        elif hasattr(nn, 'lstm_layers'):
            net_type = "LSTMNetwork"
        else:
            net_type = "UnknownNetwork"
        text_surface = font.render(net_type, True, (255, 255, 255))  # White color
        text_rect = text_surface.get_rect(center=(self.w_width // 2, 30))  # Centered near top
        screen.blit(text_surface, text_rect)

        # --- Draw nodes and edges ---
        all_nodes_pos = self.draw_nodes(screen, nn, marked_node_idx)
        weights = self.extract_weights(nn)
        min_weight, max_weight = self.get_weight_range(weights)

        for l_idx in range(len(all_nodes_pos) - 1):
            for n1_idx in range(len(all_nodes_pos[l_idx])):
                for n2_idx in range(len(all_nodes_pos[l_idx + 1])):
                    current_node_pos = all_nodes_pos[l_idx][n1_idx]
                    next_node_pos = all_nodes_pos[l_idx + 1][n2_idx]
                    weight = weights[l_idx][n1_idx][n2_idx]
                    
                    scaled_weight = (weight - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 0.5
                    line_color = self.get_weight_color(weight, scaled_weight)
                    width = max(1, int(abs(weight) // 1))
                    pygame.draw.line(screen,
                                    line_color,
                                    (current_node_pos[0] + self.node_size, current_node_pos[1]),
                                    (next_node_pos[0] - self.node_size, next_node_pos[1]),
                                    width=width)

    def draw_nodes(self, screen, nn, marked_node_idx):
        layer_dims = self.get_layer_dims(nn)
        w_stride = self.w_width * 0.05
        h_stride = self.w_height * 0.0375
        self.node_size = self.w_height * 0.0175 * 0.5
        nn_len = len(layer_dims)
        all_nodes_pos = list()

        for l_idx in range(nn_len):
            l_nodes_pos = list()
            num_nodes = layer_dims[l_idx]
            for n_idx in range(num_nodes):
                adjust = num_nodes * h_stride // 2
                x = (self.w_width * .25) + l_idx * w_stride - 175
                y = (self.w_height * .25) - adjust + n_idx * h_stride
                node_color = (255, 255, 255)
                if l_idx == (nn_len - 1) and n_idx == marked_node_idx:
                    node_color = (255, 0, 0)
                pygame.draw.circle(screen, node_color, (x, y), self.node_size, width=3)
                l_nodes_pos.append((x, y))
            all_nodes_pos.append(l_nodes_pos)
        return all_nodes_pos

    def extract_weights(self, nn):
        """Return a dynamic format of weights: [layer][node][next_node]"""
        if hasattr(nn, 'weights'):  # DenseNetwork
            return nn.weights
        elif hasattr(nn, 'lstm_layers'):
            combined = []
            for lstm_layer in nn.lstm_layers:
                # Get the average of all gate weights for visualization
                avg_weight = np.mean([
                    lstm_layer['W_i'],
                    lstm_layer['W_f'],
                    lstm_layer['W_c'],
                    lstm_layer['W_o']
                ], axis=0)
                combined.append(avg_weight)
            # Output layer weights
            combined.append(nn.V)
            return combined
        else:
            raise ValueError("Unsupported network type for drawing.")

    def get_layer_dims(self, nn):
        """Get the node counts at each layer."""
        if hasattr(nn, 'layer_dims'):
            return nn.layer_dims
        else:
            raise ValueError("Unsupported network type for getting layer dimensions.")

    def get_weight_range(self, weights):
        """Return the min and max values of the weights across all layers."""
        all_weights = [w for layer in weights for row in layer for w in row]
        return min(all_weights), max(all_weights)

    @staticmethod
    def get_weight_color(weight, max_abs_weight=1.0):
        """Maps a weight to a color: strong negative=red, strong positive=blue, zero=white.
        Grayish weights are returned with partial transparency (alpha < 255)."""
        if max_abs_weight == 0:
            max_abs_weight = 1.0

        normalized = weight / max_abs_weight
        normalized = max(-1.0, min(1.0, normalized))  # clamp between -1 and 1

        if normalized < 0:
            # fades toward red as weight becomes more negative
            r = 255
            g = int(255 * (1 + normalized))
            b = int(255 * (1 + normalized))
        else:
            # fades toward blue as weight becomes more positive
            r = int(255 * (1 - normalized))
            g = int(255 * (1 - normalized))
            b = 255

        # Detect grayish color
        is_grayish = abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20
        alpha = 80 if is_grayish else 255  # See-through if grayish

        return (r, g, b, alpha)

