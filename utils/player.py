import time
import pygame


class Controller:
    def __init__(self):
        self.selected_obj = None
    
    def initialize(self, selected_obj):
        self.player_start_time = time.time()
        self.selected_obj = selected_obj

    def deinitialize(self):
        self.selected_obj.is_player = None
        self.selected_obj = None

    def handle_player_movement(self, keys):
        if self.selected_obj is None:
            return
        
        if keys[pygame.K_e] and time.time() - self.player_start_time > 0.5:
            self.selected_obj.is_player = not self.selected_obj.is_player
            self.player_start_time = time.time()

        if self.selected_obj.is_player:
            if keys[pygame.K_w]:
                self.selected_obj.output_idx = 3
                self.selected_obj.move()
            if keys[pygame.K_s]:
                self.selected_obj.output_idx = 2
                self.selected_obj.move()
            if keys[pygame.K_a]:
                self.selected_obj.output_idx = 1
                self.selected_obj.move()
            if keys[pygame.K_d]:
                self.selected_obj.output_idx = 0
                self.selected_obj.move()
                
        if not self.selected_obj.alive or (keys[pygame.K_ESCAPE]):
            self.deinitialize()

    def update(self, keys):
        self.handle_player_movement(keys)