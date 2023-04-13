# Quadrotor simulator using pybullet

import pybullet as p
import gym
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
import seaborn as sns
from icecream import ic
import time


class PIDController:
    """PID controller for attitude rate control

    Returns:
        _type_: _description_
    """

    def __init__(self, kp, ki, kd, ki_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ki_max = ki_max
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)

    def reset(self):
        self.integral = np.zeros(3)
        self.last_error = np.zeros(3)

    def update(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.ki_max, self.ki_max)
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class QuadBullet(gym.Env):
    """
    Quadrotor simulator using pybullet wrapped as a gym environment
    """

    def __init__(self) -> None:
        super().__init__()

        # motor parameters
        self.max_torque = np.array([9e-3, 9e-3, 2e-3])
        self.max_thrust = 0.60
        self.drone_mass = 0.027
        self.obj_mass = 0.01
        self.J = np.array(
            [[1.7e-5, 0.0, 0.0], [0.0, 1.7e-5, 0.0], [0.0, 0.0, 2.98e-5]])

        # Initialize pybullet
        self.CLIENT = p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        # set simulator parameters
        self.sim_dt = 4e-4
        self.ctl_substeps = 5
        self.ctl_dt = self.sim_dt * self.ctl_substeps
        self.step_substeps = 50
        self.step_dt = self.ctl_dt * self.step_substeps
        p.setTimeStep(self.sim_dt)
        # set ground plane
        p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, 0)
        # disable GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # set camera pose
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=45,
                                     cameraPitch=-30,
                                     cameraTargetPosition=[0, 0, 0])

        # Load quadrotor
        self.quad = p.loadURDF("assets/cf2x.urdf", [0, 0, 0.5])
        # get joint number
        self.num_joints = p.getNumJoints(self.quad)

        # Set up action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(12,))
        self.obs_keys = ['xyz_drone', 'quat_drone', 'vxyz_drone', 'vrpy_drone']

        # controller
        self.KP = np.array([64e2, 64e2, 32e2])
        self.KI = np.array([256e2, 256e2, 12e2])
        self.KD = np.array([2e-1, 2e-1, 0.0])
        self.KI_MAX = np.array([10000.0, 10000.0, 10000.0])
        self.controller = PIDController(self.KP, self.KI, self.KD, self.KI_MAX)

        # reset
        self.thrust = 0.0
        self.torque = np.zeros(3)
        self.target_rpy_rate = np.zeros(3)
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the simulation
        """
        # Reset controller
        self.controller.reset()
        # Reset quadrotor
        p.resetBasePositionAndOrientation(self.quad, [0, 0, 0.5], [0, 0, 0, 1])
        # step simulation
        p.stepSimulation()
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step the simulation by one time step
        """

        # convert action to rpm
        thrust = action[0]
        target_rpy_rate = action[1:]

        for _ in range(self.step_substeps):
            self.ctlstep(thrust, target_rpy_rate)

        # Get state
        state = self._get_state()
        obs = np.concatenate((state[k] for k in self.obs_keys))

        # Get reward
        reward = 0

        # Check if done
        done = False

        # Get info
        info = {}

        return obs, reward, done, info

    def ctlstep(self, thrust, target_rpy_rate):
        # run lower level attitude rate PID controller
        self.target_rpy_rate = target_rpy_rate
        rpy_rate_error = target_rpy_rate - self.vrpy_drone
        torque = self.J @ self.controller.update(rpy_rate_error, self.ctl_dt)
        thrust, torque = np.clip(thrust, 0.0, self.max_thrust), np.clip(
            torque, -self.max_torque, self.max_torque)
        for _ in range(self.ctl_substeps):
            self.simstep(thrust, torque)
        return self._get_state()

    def simstep(self, thrust, torque):
        self.thrust, self.torque = thrust, torque
        p.applyExternalForce(self.quad,
                             0,
                             forceObj=[0, 0, self.thrust],
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )
        p.applyExternalTorque(self.quad,
                              0,
                              torqueObj=self.torque,
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )
        # set zero joint force
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.quad,
                                    i,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=0,
                                    force=0,
                                    physicsClientId=self.CLIENT
                                    )
        # Step simulation
        p.stepSimulation()

    def _get_state(self) -> np.ndarray:
        """
        Get state of the quadrotor
        """

        self.xyz_drone, self.quat_drone, _, _, _, _, self.vxyz_drone, self.vrpy_drone = p.getLinkState(
            self.quad, 0, 1)
        self.xyz_obj, self.quat_obj, _, _, _, _, self.vxyz_obj, self.vrpy_obj = p.getLinkState(
            self.quad, 3, 1)
        self.rpy_drone = p.getEulerFromQuaternion(self.quat_drone)
        self.rpy_obj = p.getEulerFromQuaternion(self.quat_obj)
        self.rotmat_drone = np.array(
            p.getMatrixFromQuaternion(self.quat_drone)).reshape(3, 3)
        self.rotmat_obj = np.array(
            p.getMatrixFromQuaternion(self.quat_obj)).reshape(3, 3)

        state = {
            'xyz_drone': self.xyz_drone,
            'quat_drone': self.quat_drone,
            'rpy_drone': self.rpy_drone,
            'rotmat_drone': self.rotmat_drone,
            'vxyz_drone': self.vxyz_drone,
            'vrpy_drone': self.vrpy_drone,
            'vrpy_drone_error': self.target_rpy_rate - self.vrpy_drone,
            'xyz_obj': self.xyz_obj,
            'quat_obj': self.quat_obj,
            'rpy_obj': self.rpy_obj,
            'rotmat_obj': self.rotmat_obj,
            'vxyz_obj': self.vxyz_obj,
            'vrpy_obj': self.vrpy_obj,
            'thrust': self.thrust,
            'torque': self.torque,
            'thrust_normed': self.thrust/self.max_thrust,
            'torque_normed': self.torque/self.max_torque
        }
        # convert all to numpy arrays
        for key in state.keys():
            state[key] = np.array(state[key])
        return state


class Logger:
    def __init__(self) -> None:
        self.log_items = ['xyz_drone', 'rpy_drone', 'vxyz_drone', 'vrpy_drone', 'thrust_normed',
                          'torque_normed', 'vrpy_drone_error', 'xyz_drone_error', 'xyz_obj_error', 'rpy_drone_error', 'xyz_obj']
        self.log_dict = {item: [] for item in self.log_items}

    def log(self, state):
        for item in self.log_items:
            self.log_dict[item].append(np.array(state[item]))

    def plot(self, filename):
        # set seaborn theme
        sns.set_theme()
        # create figure
        fig, axs = plt.subplots(len(self.log_items), 1,
                                figsize=(10, 3*len(self.log_items)))
        # plot
        x_time = np.arange(len(self.log_dict[self.log_items[0]])) / 500.0
        for i, item in enumerate(self.log_items):
            axs[i].plot(x_time, self.log_dict[item])
            axs[i].set_title(item)
        # save
        fig.savefig(filename)


def test():
    logger = Logger()
    env = QuadBullet()
    state = env.reset()
    target_pos = np.array([0.5, 0.55, 0.6])
    # create a sphere with pybullet to visualize the target position
    target_sphere = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 0.5])
    target_sphere = p.createMultiBody(
        baseVisualShapeIndex=target_sphere, basePosition=target_pos)

    # set up controllers
    objpos_controller = PIDController(
        np.ones(3)*10.0, np.ones(3)*0.0, np.ones(3)*12.0, np.ones(3)*100.0)
    pos_controller = PIDController(
        np.ones(3)*12.0, np.ones(3)*0.3, np.ones(3)*0.0, np.ones(3)*100.0)
    attitude_controller = PIDController(
        np.ones(3)*7.0, np.ones(3)*0.1, np.ones(3)*0.05, np.ones(3)*100.0)
    pos_controller.reset()
    attitude_controller.reset()

    for i in range(100):
        # if i == 0:
        #     # give the object an initial velocity
        #     p.applyExternalForce(env.quad,
        #                             3,
        #                             forceObj=[20.0, 20.0, 0.0],
        #                             posObj=[0, 0, 0],
        #                             flags=p.LINK_FRAME,
        #                             physicsClientId=env.CLIENT
        #                             )

        # PID controller
        # Object-level controller
        delta_pos = np.clip(target_pos - state['xyz_obj'], -1.0, 1.0)
        target_force_obj = env.obj_mass * \
            objpos_controller.update(
                delta_pos, env.step_dt) + np.array([0.0, 0.0, 9.81*env.obj_mass])
        xyz_obj2drone = state['xyz_obj'] - state['xyz_drone']
        z_hat_obj = xyz_obj2drone / np.linalg.norm(xyz_obj2drone)
        target_force_obj_projected = np.dot(
            target_force_obj, z_hat_obj) * z_hat_obj
        # Drone-level controller
        xyz_drone_target = state['xyz_obj'] + target_force_obj / \
            np.linalg.norm(target_force_obj) * 0.2 + np.array([0.0, 0.0, 0.03])
        # TODO check target_force_obj_projected, make the mass of the object smaller
        target_force_drone = env.drone_mass*pos_controller.update(xyz_drone_target - state['xyz_drone'], env.step_dt) + np.array([
            0.0, 0.0, 9.81*env.drone_mass]) + target_force_obj_projected
        thrust_desired = np.dot(target_force_drone, state['rotmat_drone'])
        thrust = thrust_desired[2]
        rot_err = np.cross(
            np.array([0, 0, 1]), thrust_desired/np.linalg.norm(thrust_desired))
        rot_err[2] = 0.0 - state['rpy_drone'][2]
        rpy_rate_target = attitude_controller.update(rot_err, env.step_dt)

        for _ in range(env.step_substeps):
            state = env.ctlstep(thrust, rpy_rate_target)
            # Extra logging term
            state['xyz_drone_error'] = xyz_drone_target - state['xyz_drone']
            state['xyz_obj_error'] = target_pos - state['xyz_obj']
            state['rpy_drone_error'] = rot_err
            logger.log(state)
            time.sleep(env.ctl_dt)
    logger.plot('results/test.png')


if __name__ == "__main__":
    test()