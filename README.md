# AIEcosystemArena
AI Ecosystem Arena is a biologically-inspired simulation of life where autonomous agents explore, survive, and evolve in a continuously generated virtual ecosystem. Each creature operates with a self-controlled neural network brain, learning from its environment in real time. No scripts. No hand-coded paths. Just instinct, memory, and survival.

Agents hunt, drink, reproduce, and avoid predators — all driven by evolving traits and reinforcement learning. Health is their lifeblood and currency: eat to thrive, learn to survive, mutate to adapt.

The result? Emergent behavior in a world that grows with them.

## Dependencies

I used Python 3.9 during development. Run the following command to install library dependencies:

```
pip install -r assets\requirements.txt
```

## Making it an Executable

Packages `main.py` into a single executable with PyInstaller, bundling the `assets` and `assets/config` folders into the final build for the OS you run the command on (run on Windows for windows exe, etc.):

```
pyinstaller --onefile --add-data "assets;assets" --add-data "assets\config;assets\config" main.py
```

To install `pyinstaller` run the command `pip install pyinstaller` in the terminal. The final executable will be `dist\main.exe`. The executable works with arguments for setting simulation `level` like so:

```
./main.exe --size=Large
```

## Example AI Agent

Below is an print of the attributes for an example AI agent "living" in the world. (click on the AI Agent in the simulation to get the printed output and a visual of the objects dense neural network):

```
Wolf (id=2115695063152)
│   ├── species_info: {'sexes': {'male': {'size': [20, 30]}, 'female': {'size': [20, 30]}}, 'mate': <class 'utils.species.Wolf'>, 'predators': (None,), 'diet': (<class 'utils.species.Deer'>,), 'speed': 3.956887459274844, 'consumption_rate': 31}
│   ├── _Sprite__g: set()
│   ├── world: <utils.environment.Environment object at 0x000001EC98A6CAF0>
│   ├── birth_coord: (283.5514351704484, 188.0040631396275)
│   ├── assets_img_path: [READCTED]
│   ├── species_type: wolf
│   ├── sexes_info: {'male': {'size': [20, 30]}, 'female': {'size': [20, 30]}}
│   ├── diet: (<class 'utils.species.Deer'>,)
│   ├── predators: (None,)
│   ├── mate_pref: <class 'utils.species.Wolf'>
│   ├── speed: 3.956887459274844
│   ├── consumption_rate: 31
│   ├── x: 283.5514351704484
│   ├── y: 380.82844880437676
│   ├── sex: male
│   ├── width: 20
│   ├── height: 30
│   ├── img: <Surface(20x30x32 SW)>
│   ├── rect: <rect(274, 366, 20, 30)>
│   ├── memory: {'predator': None, 'food': (228.5906438830329, 410.64734625277526), 'water': (218.22835960848764, 457.91233337814015), 'mate': (368.2724503972995, 302.54876849483037)}
│   ├── priority_dict: {'food': 420.4512000000028, 'predator': 0, 'mate': 0, 'water': -2.4567999999999954}
│   ├── priority: food
│   ├── start_health: 124
│   ├── health: 235.00224899012397
│   ├── age_depl: 0.00248
│   ├── water_depl: 0.124
│   ├── food_depl: 0.124
│   ├── tob: 1745624146.6841803
│   ├── water_need: -2.4567999999999954
│   ├── food_need: 420.4512000000028
│   ├── reproduction_need: 0
│   ├── avoid_need: 0
│   ├── food_increment: 1.55
│   ├── water_increment: 0.012400000000000001
│   ├── reproduction_increment: 0.062
│   ├── last_child_tob: 1745624146.6841803
│   ├── child_grace_period: 52
│   ├── look_for_mate: False
│   ├── vision_dist: 49843
│   ├── mutation_multi: 0.2
│   ├── prob_mutation: 0.1
│   ├── alive: True
│   ├── is_focused: True
│   ├── is_exploring: False
│   ├── coords_focused: <utils.species.Coords object at 0x000001EC9940B8B0>
│   ├── is_player: False
│   ├── coord_changes: [(3.956887459274844, 0), (-3.956887459274844, 0), (0, 3.956887459274844), (0, -3.956887459274844), (3.956887459274844, 3.956887459274844), (3.956887459274844, -3.956887459274844), (-3.956887459274844, 3.956887459274844), (-3.956887459274844, -3.956887459274844)]
│   ├── num_inputs: 5
│   ├── weights_layers: [5, 5, 4, 5, 8]
│   ├── brain: DenseNetwork
│   │   ├── layer_dims: [5, 5, 4, 5, 8]
│   │   ├── weights:
│   │   │   ├── Layer 0
│   │   │   │   ├── Neuron 0: [0.4865, -0.0911, 0.2273, -0.7414, -0.2539]
│   │   │   │   ├── Neuron 1: [-0.9785, -0.2746, 0.7218, 0.8227, -0.2132]
│   │   │   │   ├── Neuron 2: [0.4850, 0.1626, 0.5183, -0.5498, -0.8212]
│   │   │   │   ├── ...
│   │   │   ├── Layer 1
│   │   │   │   ├── Neuron 0: [0.2611, -0.8692, 0.7788, -0.6624]
│   │   │   │   ├── Neuron 1: [0.1475, 0.1936, -0.6195, 0.1684]
│   │   │   │   ├── Neuron 2: [-0.2601, -0.5837, -0.9489, -0.7168]
│   │   │   │   ├── ...
│   │   │   ├── Layer 2
│   │   │   │   ├── Neuron 0: [0.9366, -0.9948, 0.4013, -0.3830, 0.4148]
│   │   │   │   ├── Neuron 1: [0.2479, 0.1329, -0.9497, -0.8987, 0.5936]
│   │   │   │   ├── Neuron 2: [-0.2479, -0.2483, -0.7369, -0.8184, 0.4060]
│   │   │   │   ├── ...
│   │   │   ├── Layer 3
│   │   │   │   ├── Neuron 0: [-0.6267, -0.0628, 0.2034, -0.3336, -0.0280, -0.5862, -0.4462, -0.8213]
│   │   │   │   ├── Neuron 1: [0.6280, -0.8437, -0.6952, -0.2746, 0.4045, 0.3878, 0.7788, -1.0051]
│   │   │   │   ├── Neuron 2: [-0.7094, 0.0561, 0.4727, -0.5799, -0.1686, -0.3019, -0.3521, 0.2273]
│   │   │   │   ├── ...
│   │   ├── min_update: 0.9998
│   │   ├── max_update: 1.0002
│   │   ├── cost: 111.00224899012397
│   │   ├── output: 2
│   ├── output: None
│   ├── obj_location: (228.5906438830329, 410.64734625277526)
│   ├── output_idx: 2
│   ├── position: (283.5514351704484, 380.82844880437676)
```