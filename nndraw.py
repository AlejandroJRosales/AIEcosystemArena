import pygame


class MyScene:
    def __init__(self, window, screen_size):
        self.window = window
        self.w_width = screen_size[0]
        self.w_height = screen_size[1]
        self.node_size = 15

    def draw(self, nn):
        all_nodes_pos = self.draw_nodes(nn)
        for l_idx in range(len(all_nodes_pos) - 1):
            for n1_idx in range(len(all_nodes_pos[l_idx])):
                print(len(all_nodes_pos[l_idx + 1]))
                for n2_idx in range(len(all_nodes_pos[l_idx + 1])):
                    current_node_pos = all_nodes_pos[l_idx][n1_idx]
                    next_node_pos = all_nodes_pos[l_idx + 1][n2_idx]
                    weight = nn[l_idx][n1_idx][n2_idx]
                    width = int(abs(weight) // 1)
                    line_color = (173, 216, 230) if weight >= 0 else (255, 204, 203)
                    pygame.draw.line(self.window,
                                     line_color,
                                     (current_node_pos[0] + self.node_size, current_node_pos[1]),
                                     (next_node_pos[0] - self.node_size, next_node_pos[1]),
                                     width=width)

    def draw_nodes(self, nn):
        w_stride = 100
        h_stride = 40
        self.node_size = 13
        nn_len = len(nn)
        all_nodes_pos = list()
        for l_idx in range(nn_len + 1):
            l_nodes_pos = list()
            num_nodes = len(nn[l_idx]) if l_idx != nn_len else len(nn[-1][0])
            for n_idx in range(num_nodes):
                adjust = num_nodes * h_stride // 2
                x = (self.w_width * .2) + l_idx * w_stride - 175
                y = (self.w_height * .25) - adjust + n_idx * h_stride
                node_color = (255, 255, 255)
                pygame.draw.circle(self.window, node_color, (x, y), self.node_size, width=3)
                l_nodes_pos.append((x, y))
            all_nodes_pos.append(l_nodes_pos)
        return all_nodes_pos
