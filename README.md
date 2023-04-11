# gym-quad

## Description

This is a gym environment for quadrotor control. It is based on the PyBullet physics engine.

The environment is a 3D quadrotor with 4 rotors. The rotors are controlled by a 4D action space. The observation space is a 3D position and 3D velocity. The reward is the negative distance to a target position.

The main feature is that the quadrotor is hanging an object by a free joint. The object can be moved by the quadrotor. The object can also be moved by an external force. 

|Visualization|Multi Quad|Plot|
|-|-|-|
|![tmp](https://user-images.githubusercontent.com/60093981/231089933-e11f9e6c-9e10-406a-aed5-109ce2881d8a.gif)|![multi-quad](https://user-images.githubusercontent.com/60093981/231121359-4737dba0-174a-4285-b6b0-fc08b68af92f.gif)|<img width="240" alt="image" src="https://user-images.githubusercontent.com/60093981/231090020-a05d8195-8609-44b8-b51e-8c0dde97ab9d.png">|

## Installation

```bash
pip install -e .
```

## Usage

```bash
cd gym_quad
python quad_bullet.py
```
