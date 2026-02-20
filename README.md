# ROSBot Maze Navigation

## Project Overview

This project implements an autonomous navigation system for the RosBot within the Webots simulation environment. The robot is designed to navigate complex environments, moving from a blue pillar to a yellow pillar in the shortest possible simulation time.

The navigation logic accounts for various environmental challenges, including narrow passages and dead ends marked by red walls, strict avoidance of green ground areas, and efficient path planning to minimize total traversal time.

## File Structure
```
.
├── Maze1/
│   ├── world/           # Webots world file for Maze 1
│   └── controllers/     # Custom navigation controller for Maze 1
├── Maze2/
│   ├── world/           # Webots world file for Maze 2
│   └── controllers/     # Custom navigation controller for Maze 2
├── Maze3/
│   ├── world/           # Webots world file for Maze 3
│   └── controllers/     # Custom navigation controller for Maze 3
├── Maze4/
│   ├── world/           # Webots world file for Maze 4
│   └── controllers/     # Custom navigation controller for Maze 4
├── Maze5/
│   ├── world/           # Webots world file for Maze 5
│   └── controllers/     # Custom navigation controller for Maze 5
└── README.md            # Project documentation
```

## Configuration & Setup

To run the simulation and verify the results, follow these steps:

1. **Software Requirements:** Ensure Webots is installed and configured to support RosBot simulations.
2. **Global Pose:** The controllers utilize the Webots Supervisor to retrieve the global world pose of the robot for precise localization.
3. **Running a Maze:**
   - Open Webots.
   - Load the desired world file from the `MazeX/world/` directory.
   - The associated controller in the `controllers/` folder will automatically initialize.
   - Press the Play button to start the simulation.

## Academic Integrity

All navigation solutions and code logic were implemented by the all group members. No changes were made to the robot model or the provided simulation environments.