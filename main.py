import ctypes
import argparse
import time
import math
import pygame
from utils import species
from utils import environment
from utils import tools
from utils import display
from utils import nndraw


"""
TODO:
"I'm gonna try to build this in my simulation. I can like place large boulders then the animals will hv to figure out how to pass them. Maybe I'll use ants instead so they can transfer information and create a better path. Almost like a path finding algorithm but with ants"
obstacles, and being able to add obstacles with touch
save button, on screen attribute adjustments for "new simulation" feature
SPEED attribute doesnt hv to be intilized for class even though it is being called by a function that runs
endurance. costs more food. pass on trait for when to run/ walk
group forming to hunt using nns w/ position of others and self
have plants grow only on soil
time of day, seasons, weather, and how they affect the world
add a "pause" button to stop the simulation and allow the user to interact with the world
"""


class EcosystemScene:
	def __init__(self, world_width, world_height, size="Small"):
		print(f"\tSetting world parameters for \"{size}\" sized simulation...")
		super(EcosystemScene, self).__init__()
		
		self.w = world_width
		self.h = world_height
		self.size = size  # "Large", "Medium", "Small", or "Custom"

		# Window-based scaling factor
		base_width = 1000
		base_height = 800
		scale = min(world_width / base_width, world_height / base_height)

		# Settings based on size
		if size == "Large":
			self.proportion = 0.6
			self.bodies_of_water = 2
			self.water_body_size = "Large"
			num_plants = int(160 * scale)
			num_deer = int(120 * scale)
			num_wolfs = max(4, int(12 * scale))

		elif size == "Medium":
			self.proportion = 1
			self.bodies_of_water = 2
			self.water_body_size = "Medium"
			num_plants = int(80 * scale)
			num_deer = int(60 * scale)
			num_wolfs = max(2, int(10 * scale))

		elif size == "Small":
			self.proportion = 1.75
			self.bodies_of_water = 2
			self.water_body_size = "Small"
			num_plants = int(40 * scale)
			num_deer = int(30 * scale)
			num_wolfs = max(2, int(5 * scale))
		
		elif size == "Custom":
			self.proportion = 0.9
			self.bodies_of_water = 2
			self.water_body_size = "Medium"
			num_plants = int(50 * scale)
			num_deer = int(40 * scale)
			num_wolfs = max(1, int(8 * scale))

		else:
			raise ValueError("Invalid size. Choose from 'Large', 'Medium', 'Small', or 'Custom'.")

		self.species_types = {
			species.Plant: num_plants,
			species.Deer: num_deer,
			species.Wolf: num_wolfs
		}

		self.envir_func = None
		self.world = None
		self.containerized = True
		self.stopwatch = time.time()
		self.pop_sizes_time = dict()
		self.iterations = 0
		self.displ_pop_diff = False
		self.current_size = None
		self.past_size = 0
		self.last_touch_time = time.time()

		self.setup()

	def setup(self):
		self.envir_func = environment.Environment((self.w, self.h), proportion=self.proportion)
		self.world = self.envir_func.generate_world(
			species_types=self.species_types,
			bodies_of_water=self.bodies_of_water,
			water_body_size=self.water_body_size
		)
		self.past_size = len(self.world)
		print("Setup complete!")
		time.sleep(2)
		print(f"Starting with {self.past_size} objects in world...")

		print("Starting world in...")
		for count in range(3, 0, -1):
			print(f"{count}...")
			time.sleep(0.5)
		# self.display_internals(self.world)

	def update(self):
		self.world = self.envir_func.update(self.world)
		for obj in self.world:
			# update objects positioning based on each objects updated x and y coords
			# because plants and water dont move
			if isinstance(obj, species.Animal):
				obj.position = (obj.x, obj.y)
		self.current_size = len(self.world)

		if self.displ_pop_diff and abs(self.current_size - self.past_size) >= 10:
			print(f"Simulating {self.past_size} objects...")
			self.past_size = self.current_size

	def select_obj(self, touch_coords):
		selected_obj = None
		closest_obj_dist = math.inf
		for obj in self.world:
			obj_dist = tools.distance_formula(touch_coords[0], touch_coords[1], obj.x, obj.y)
			if isinstance(obj, species.Living) and obj_dist < closest_obj_dist:
				selected_obj = obj
				closest_obj_dist = obj_dist
		return selected_obj

	def draw_transparent_circle(self, screen, selected_obj):
		if isinstance(selected_obj, species.Animal):
			# Define the center (x, y) and radius (vision_dist) from the selected object
			x, y = selected_obj.x, selected_obj.y
			radius = selected_obj.vision_dist

			# Define the color with alpha (0.3 for transparency)
			circle_color = (34, 139, 34, 0)

			# Create a new surface with the same size as the window and an alpha channel
			circle_surface = pygame.Surface((self.w, self.h), pygame.SRCALPHA)  # Use SRCALPHA for transparency

			# Draw the circle on this new surface
			pygame.draw.circle(circle_surface, circle_color, (x + (selected_obj.width / 2), y + (selected_obj.height / 2)), radius)

			# Blit the surface onto the main window
			screen.blit(circle_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

	def display_internals(self, structure):
		for k, v in self.envir_func.__dict__.items():
			print(f"{k}: {v}")
		for obj in structure:
			print(f"\n{type(obj)}")
			for k, v in obj.__dict__.items():
				print(f"{k}: {v}")
			method_list = [func for func in dir(obj) if callable(getattr(obj, func)) and func[0] != "_"]
			print(f"methods:\n{method_list}")

	def unethical_runtime_optimization(self):
		kill = True
		for living in self.world:
			if isinstance(living, species.Living):
				if kill:
					living.die()
				kill = not kill

def to_hex(c):
	return '{:X}{:X}{:X}'.format(c[0], c[1], c[2])


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the ecosystem simulation.")
parser.add_argument('--size', type=str, choices=['Small', 'Medium', 'Large'], default='Medium',
					help='Simulation size: Small, Medium, Large, or Custom (default: Medium)')
args = parser.parse_args()
sim_size = "Large"
sim_size = sim_size.capitalize()

pygame.init()
pygame.font.init()

# get monitor width and height for full screen mode
# user32 = ctypes.windll.user32
# world_width = user32.GetSystemMetrics(0)
# world_height = user32.GetSystemMetrics(1)

# preset window size
world_width = 850
world_height = 600

print("Starting creation of new world object...")
# simulation setup
ecosystem = EcosystemScene(world_width, world_height, size=sim_size)

# display setup
screen = pygame.display.set_mode((ecosystem.w, ecosystem.h))
nnd = nndraw.MyScene(screen, (ecosystem.w, ecosystem.h))
pygame.display.set_caption('Ecosystem Simulation')
clock = pygame.time.Clock()
internal_speed = 50
display_world = True
selected_obj = None
print("Now displaying world\n")
while True:
	if display_world:
		index = 0
		for row in range(ecosystem.envir_func.w_num_chunks):
			for col in range(ecosystem.envir_func.h_num_chunks):
				pygame.draw.rect(screen,
								 ecosystem.envir_func.terrain_color[index],
								 (ecosystem.envir_func.w_chunk_size * row,
								  ecosystem.envir_func.h_chunk_size * col,
								  ecosystem.envir_func.w_chunk_size,
								  ecosystem.envir_func.h_chunk_size)
								 )
				index += 1
		[screen.blit(obj.img, (obj.x, obj.y)) if not isinstance(obj, environment.Water) else pygame.draw.rect(screen, (40, 101, 201), obj.rect) for obj in ecosystem.world]

	# check for users key pressed
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_q:
				pygame.quit()
		if event.type == pygame.MOUSEBUTTONUP:
			# # get object closest to where user clicked
			selected_obj = ecosystem.select_obj(pygame.mouse.get_pos())
			
			# # display stats of object
			# display.analysis_mode(selected_obj)
			
			# selected_obj = ecosystem.select_obj(pygame.mouse.get_pos())
			# selected_obj.is_player = True

	# checking pressed held
	keys = pygame.key.get_pressed()
	if selected_obj is not None:
		ecosystem.draw_transparent_circle(screen, selected_obj)
		if not selected_obj.alive:
			# if living died
			selected_obj = None
		# if keys[pygame.K_w]:
		# 	selected_obj.move(coord_change=(0, -1))
		# if keys[pygame.K_s]:
		# 	selected_obj.move(coord_change=(0, 1))
		# if keys[pygame.K_a]:
		# 	selected_obj.move(coord_change=(-1, 0))
		# if keys[pygame.K_d]:
		# 	selected_obj.move(coord_change=(1, 0))

		# if keys[pygame.K_ESCAPE]:
		# 	selected_obj.is_player = None
		# 	selected_obj = None

	if keys[pygame.K_t]:
		ecosystem.unethical_runtime_optimization()
		print("Pruning population")
	if keys[pygame.K_n]:
		# quit current window
		pygame.display.quit()
		clock = pygame.time.Clock()

		num_living_objects = len([obj for obj in ecosystem.world if isinstance(obj, species.Living)])
		print(f"[TERMINATING {num_living_objects} LIVING OBJECTS ({ecosystem.current_size} OBJECTS TOTAL)]")
		del ecosystem
		print("[TERMINATION COMPLETE]")
		time.sleep(2)
		print("\n\n\nStarting creation of new world object...")
		ecosystem = EcosystemScene(world_width, world_height, size=sim_size)
		print("Now displaying world")

		# display new window
		screen = pygame.display.set_mode((world_width, world_height))
		pygame.display.set_caption('EcosystemScene')
	if keys[pygame.K_f]:
		internal_speed += 5
		print(f"Internal speed: {internal_speed}")

	# if keys[pygame.K_d]:
	# 	display_world = not display_world
	# 	print(f"Display paused...") if not display_world else print("Display running...")
	# if keys[pygame.K_s]:
	# 	internal_speed -= 5
	# 	print(f"Internal speed: {internal_speed}")
	# if keys[pygame.K_d]:
	# 	display_world = not display_world
	# 	print(f"Display paused...") if not display_world else print("Display running...")

	ecosystem.update()

	if display_world and selected_obj is not None and isinstance(selected_obj, species.Animal):
		nnd.draw(screen, selected_obj.brain, selected_obj.output_idx)

	pygame.display.update()

	clock.tick(internal_speed)
