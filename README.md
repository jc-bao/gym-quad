# gym-quad

## Description

This is a gym environment for quadrotor control. It is based on the PyBullet physics engine.

The environment is a 3D quadrotor with 4 rotors. The rotors are controlled by a 4D action space. The observation space is a 3D position and 3D velocity. The reward is the negative distance to a target position.

The main feature is that the quadrotor is hanging an object by a free joint. The object can be moved by the quadrotor. The object can also be moved by an external force. 

## Installation

```bash
pip install -e .
```

## Usage

```bash
cd gym_quad
python quad_bullet.py
```
