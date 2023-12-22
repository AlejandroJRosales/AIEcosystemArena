import random
import species
import pygame
import math

import utils


class Environment:
	def __init__(self, world_size, proportion=0.1):
		print("\tSetting environment parameters...")
		self.world_width = world_size[0]
		self.world_height = world_size[1]
		self.proportion = proportion
		self.w_num_chunks = 50
		self.h_num_chunks = 50
		self.w_chunk_size = math.ceil(self.world_width / self.w_num_chunks)
		self.h_chunk_size = math.ceil(self.world_height / self.h_num_chunks)
		self.animals = species.Animal.__subclasses__()
		self.children = []
		self.unwalkable = []
		self.soil = []
		self.terrain_color = []
		self.coord = 0
		self.last_walk_idx = 0
		self.x = 0
		self.y = 0

	def generate_world(self, species_types=None, bodies_of_water=5, water_body_size="Large"):
		print("\tCreating world...")
		water = self.generate_terrain(bodies_of_water, water_body_size)
		living = self.generate_living(species_types)
		return water + living

	def generate_terrain(self, bodies_of_water, water_body_size):
		print("\t\tCreating terrain...")
		self.generate_water(bodies_of_water, water_body_size)
		water = self.unwalkable
		print(f"\t\t\t\tCreated {len(water)} objects of water")

		# do not need to return land because land to enhance display only and is not used by other objects
		self.generate_land()

		return water

	def generate_living(self, species_types):
		print("\t\tCreating living objects...")
		# todo: walkable blocks
		# loop through list of animals
		plants = list()
		animals = list()
		for class_type in species_types.keys():
			# generate as many of the type of animal as user specified
			# todo: possible to shorten this code
			if class_type == species.Plant:
				plants = [self.generate_plant() for i in range(species_types[class_type])]
			else:
				animals += [class_type(self, (self.world_width * random.random(), self.world_height * random.random()))
						   for i in range(species_types[class_type])]
		print(f"\t\t\tCreated {len(plants)} plant objects")
		print(f"\t\t\tCreated {len(animals)} animals objects")
		return plants + animals

	def generate_water(self, bodies_of_water, water_body_size):
		print("\t\t\tCreating bodies of water...")
		water = list()
		for w in range(bodies_of_water):
			body_size = random.randint(100, 200) if water_body_size != "Large" else random.randint(400, 500)
			self.generate_vain(Water, body_size)
		return water

	def generate_vain(self, class_type, counter):
		self.coord = (self.world_width * random.uniform(0, 0.75), self.world_height * random.uniform(0, 0.75))
		self.x = self.coord[0]
		self.y = self.coord[1]
		for step in range(counter):
			res = class_type((self.x, self.y), self.proportion)
			self.unwalkable.append(res)
			self.walk(res)

	def generate_land(self):
		print("\t\t\tCreating land...")
		water_rect = [obj.rect for obj in self.unwalkable]
		for row in range(self.w_num_chunks):
			for col in range(self.h_num_chunks):
				collided_water = [rect for rect in water_rect if
								  utils.in_range(
									  self.w_chunk_size * row,
									  self.h_chunk_size * col,
									  rect.x,
									  rect.y,
									  random.randint(20, 80))]
				if len(collided_water) > 0:
					temp_soil_rect = pygame.rect.Rect(self.w_chunk_size * row,
													  self.h_chunk_size * col,
													  self.w_chunk_size,
													  self.h_chunk_size)
					# closest = int(min([utils.distance_formula(temp_soil_rect.x, temp_soil_rect.y, water.x, water.y) for water in collided_water]))
					# self.terrain_color.append(
					# 	(random.randint(110 + closest, 150 + closest),
					# 	 random.randint(90 + closest, 110 + closest),
					# 	 random.randint(65 + closest, 85 + closest))
					# )
					self.terrain_color.append(
						(random.randint(200, 240),
						 random.randint(170, 190),
						 random.randint(150, 170))
					)
					# self.terrain_color.append(
					# 	(random.randint(110, 150),
					# 	 random.randint(90, 110),
					# 	 random.randint(65, 85))
					# )
					if not temp_soil_rect.collidelist(collided_water):
						self.soil.append(temp_soil_rect)
				else:
					self.terrain_color.append((
						random.randint(75, 90), random.randint(110, 120), random.randint(40, 50)))

	def walk(self, obj):
		walk_size = obj.size[0]
		# todo: make this cleaner
		options = [count for count in range(4)]
		options.pop(self.last_walk_idx)
		walk_options = {0: (0, walk_size), 1: (0, -walk_size), 2: (walk_size, 0), 3: (-walk_size, 0), 4: (walk_size, 0)}
		self.last_walk_idx = random.choice(options)
		coord_change = walk_options[self.last_walk_idx]
		self.x = (self.x + coord_change[0]) % self.world_width
		self.y = (self.y + coord_change[1]) % self.world_height
		
	def generate_plant(self):
		if random.random() <= 0.8:
			coords = random.choice(self.soil)
			# print(coords)
			plant_coords = (coords.x + random.uniform(-60, 60), coords.y + random.uniform(-60, 60))
			return species.Plant(self, plant_coords)
		else:
			return species.Plant(self, (self.world_width * random.random(), self.world_height * random.random()))

	def update(self, environment):
		# update objects
		for obj in environment:
			if any([isinstance(obj, entity) for entity in self.animals]):
				obj.update(self, environment)
			elif isinstance(obj, species.Plant):
				obj.update()

		if random.random() <= 0.75:
			environment.append(self.generate_plant())

		environment = [obj for obj in environment if obj.alive]
		environment += self.children
		self.children = []
		return environment


class Water(pygame.sprite.Sprite):
	def __init__(self, coord, proportion):
		pygame.sprite.Sprite.__init__(self)
		self.x = coord[0]
		self.y = coord[1]
		size = 25
		self.alive = True

		# img_file_name = "images/water.jpg"
		# self.img = pygame.image.load(img_file_name)
		# self.img = pygame.transform.scale(self.img, (size, size))
		# self.rect = pygame.rect.Rect((self.x, self.y, size, size))
		# self.rect.center = (self.x, self.y)
		self.rect = pygame.rect.Rect((self.x, self.y, size, size))

		self.position = (self.x, self.y)
		self.size = (size, size)

	def get_coords(self):
		return self.x, self.y
