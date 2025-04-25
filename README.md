# AIEcosystemArena
AI Ecosystem Arena is a biologically-inspired simulation of life where autonomous agents explore, survive, and evolve in a continuously generated virtual ecosystem. Each creature operates with a self-controlled neural network brain, learning from its environment in real time. No scripts. No hand-coded paths. Just instinct, memory, and survival.

Agents hunt, drink, reproduce, and avoid predators — all driven by evolving traits and reinforcement learning. Health is their lifeblood and currency: eat to thrive, learn to survive, mutate to adapt.

The result? Emergent behavior in a world that grows with them.

## Making it an Executable

Packages `main.py` into a single executable with PyInstaller, bundling the `assets` and `assets/config` folders into the final build for the OS you run the command on (run on Windows for windows exe, etc.):

```
pyinstaller --onefile --add-data "assets;assets" --add-data "assets\config;assets\config" main.py
```

The final executable will be `dist\main.exe`. The executable works with arguments for setting simulation size/`level` like so:

```
./main.exe --level=Large
```

## Dependencies

I used Python 3.9 during development. Run the following command to install library dependencies:

```
pip install -r assets\requirements.txt
```