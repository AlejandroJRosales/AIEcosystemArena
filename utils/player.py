import pygame


class Controller:
    def __init__(self):
        self.selected_obj = None
    
    def initialize(self, selected_obj):
        self.selected_obj = selected_obj

    def deinitialize(self):
        self.selected_obj.is_player = None
        self.selected_obj = None

    def handle_player_movement(self, keys):
        if self.selected_obj is None:
            return
        
        if keys[pygame.K_e]:
            self.selected_obj.is_player = True

        if self.selected_obj.is_player:
            if keys[pygame.K_w]:
                self.selected_obj.move(coord_idx=3)
            if keys[pygame.K_s]:
                self.selected_obj.move(coord_idx=2)
            if keys[pygame.K_a]:
                self.selected_obj.move(coord_idx=1)
            if keys[pygame.K_d]:
                self.selected_obj.move(coord_idx=0)
                
        if not self.selected_obj.alive or (keys[pygame.K_ESCAPE] and self.selected_obj.is_player):
            self.deinitialize()