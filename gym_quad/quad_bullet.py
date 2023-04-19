# Quadrotor simulator using pybullet

import pybullet as p
import gym
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt
import seaborn as sns
from icecream import ic
import pandas as pd


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

        self.logger = Logger(enable=True)

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
        self.obs_keys = ['xyz_drones', 'quat_drones', 'vxyz_drones', 'vrpy_drones']

        # controller
        self.KP = np.array([4e2, 4e2, 1e2])
        self.KI = np.array([1e3, 1e3, 3e2])
        self.KD = np.array([0.0, 0.0, 0.0])
        self.KI_MAX = np.array([10000.0, 10000.0, 10000.0])
        self.attirate_controller = PIDController(
            self.KP, self.KI, self.KD, self.KI_MAX)
        self.attitude_controller = PIDController(
            np.ones(3)*7.0, np.ones(3)*0.1, np.ones(3)*0.05, np.ones(3)*100.0)
        self.objpos_controller = PIDController(
            np.ones(3)*10.0, np.ones(3)*0.0, np.ones(3)*12.0, np.ones(3)*100.0)
        self.pos_controller = PIDController(
            np.ones(3)*12.0, np.ones(3)*0.3, np.ones(3)*0.0, np.ones(3)*100.0)

        # reset
        self.thrust = 0.0
        self.torque = np.zeros(3)
        self.vrpy_target = np.zeros(3)
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reset the simulation
        """
        # Reset controller
        self.attirate_controller.reset()
        self.attitude_controller.reset()
        self.pos_controller.reset()
        self.objpos_controller.reset()
        # Reset quadrotor
        p.resetBasePositionAndOrientation(self.quad, [0, 0, 0.5], [0, 0, 0, 1])
        # step simulation
        p.stepSimulation()
        return self._get_state()

    def close(self) -> None:
        """
        Close the simulation
        """
        p.disconnect()
        self.logger.plot('results/test')

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step the simulation by one time step
        """

        # convert action to rpm
        thrust = action[0]
        vrpy_target = action[1:]

        for _ in range(self.step_substeps):
            self.ctlstep(thrust, vrpy_target)

        # Get state
        state = self._get_state()
        obs = np.concatenate([state[k] for k in self.obs_keys])

        # Get reward
        reward = 0

        # Check if done
        done = False

        # Get info
        info = {}

        return obs, reward, done, info

    def ctlstep(self, thrust, vrpy_target):
        # run lower level attitude rate PID controller
        self.vrpy_target = vrpy_target
        vrpy_error = vrpy_target - self.vrpy_drones
        torque = self.J @ self.attirate_controller.update(
            vrpy_error, self.ctl_dt)
        thrust, torque = np.clip(thrust, 0.0, self.max_thrust), np.clip(
            torque, -self.max_torque, self.max_torque)
        for _ in range(self.ctl_substeps):
            self.simstep(thrust, torque)

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
        # log data
        self.logger.log(self._get_state())

    def _get_state(self) -> np.ndarray:
        """
        Get state of the quadrotor
        """

        self.xyz_drones, self.quat_drones, _, _, _, _, self.vxyz_drones, self.vrpy_drones = p.getLinkState(
            self.quad, 0, 1)
        self.xyz_obj, self.quat_obj, _, _, _, _, self.vxyz_obj, self.vrpy_obj = p.getLinkState(
            self.quad, 3, 1)
        self.rpy_drones = p.getEulerFromQuaternion(self.quat_drones)
        self.rpy_obj = p.getEulerFromQuaternion(self.quat_obj)
        self.rotmat_drones = np.array(
            p.getMatrixFromQuaternion(self.quat_drones)).reshape(3, 3)
        self.rotmat_obj = np.array(
            p.getMatrixFromQuaternion(self.quat_obj)).reshape(3, 3)

        state = {
            'xyz_drones': self.xyz_drones,
            'quat_drones': self.quat_drones,
            'rpy_drones': self.rpy_drones,
            'rotmat_drones': self.rotmat_drones,
            'vxyz_drones': self.vxyz_drones,
            'vrpy_drones': self.vrpy_drones,
            'vrpy_drones_error': self.vrpy_target - self.vrpy_drones,
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

    def policy_att(self, vec_target, thrust, extra_torque=np.zeros(3)):
        rot_err = np.cross(np.array([0, 0, 1]), vec_target)
        rot_err[2] = 0.0 - self.rpy_drones[2]
        rpy_rate_target = self.attitude_controller.update(
            rot_err, self.step_dt) + extra_torque
        return np.concatenate((np.array([thrust]), rpy_rate_target))

    def policy_pos(self, xyz_target, extra_force=np.zeros(3)):
        pos_err = xyz_target - self.xyz_drones
        target_force_drones = self.drone_mass*self.pos_controller.update(pos_err, self.step_dt) + np.array([
            0.0, 0.0, 9.81*self.drone_mass]) + extra_force
        thrust_desired = np.dot(target_force_drones, self.rotmat_drones)
        thrust = thrust_desired[2]
        vec_target = thrust_desired/np.linalg.norm(thrust_desired)
        return self.policy_att(vec_target, thrust)

    def policy_obj(self, xyz_target):
        pos_err = xyz_target - self.xyz_obj
        target_force_obj = self.obj_mass*self.pos_controller.update(pos_err, self.step_dt) + np.array([
            0.0, 0.0, 9.81*self.obj_mass])
        xyz_obj2drone = self.xyz_obj - self.xyz_drones
        z_hat_obj = xyz_obj2drone / np.linalg.norm(xyz_obj2drone)
        target_force_obj_projected = np.dot(
            target_force_obj, z_hat_obj) * z_hat_obj
        xyz_drones_target = self.xyz_obj + target_force_obj / \
            np.linalg.norm(target_force_obj) * 0.2 + np.array([0.0, 0.0, 0.03])
        return self.policy_pos(xyz_drones_target, target_force_obj_projected)


class Logger:
    def __init__(self, enable=True) -> None:
        self.enable = enable
        self.log_items = ['xyz_drones', 'rpy_drones',
                          'vxyz_drones', 'vrpy_drones', 'xyz_obj', 'vxyz_obj']
        self.log_dict = {item: [] for item in self.log_items}

    def log(self, state):
        if not self.enable:
            return
        for item in self.log_items:
            self.log_dict[item].append(np.array(state[item]))

    def plot(self, filename):
        if not self.enable:
            return
        # set seaborn theme
        sns.set_theme()
        # create figure
        fig, axs = plt.subplots(len(self.log_items), 1,
                                figsize=(10, 3*len(self.log_items)))
        # plot
        x_time = np.arange(len(self.log_dict[self.log_items[0]])) * 4e-4
        for i, item in enumerate(self.log_items):
            self.log_dict[item] = np.array(self.log_dict[item])
            axs[i].plot(x_time, self.log_dict[item])
            axs[i].set_title(item)
        # save
        fig.savefig(filename+'.png')
        # save the dict log_items as csv
        # first split all multi-dimensional arrays into single-dimensional arrays
        save_dict = {'time': x_time}
        for item in self.log_items:
            if len(self.log_dict[item].shape) == 1:
                save_dict[item] = self.log_dict[item]
            else:
                for i in range(self.log_dict[item].shape[1]):
                    save_dict[item+'_'+str(i)] = self.log_dict[item][:, i]
        df = pd.DataFrame(save_dict)
        df.to_csv(filename+'.csv', index=False)

def test():
    env = QuadBullet()
    state = env.reset()
    target_pos = np.array([0.5, 0.55, 0.6])
    # create a sphere with pybullet to visualize the target position
    target_sphere = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 0.5])
    target_sphere = p.createMultiBody(
        baseVisualShapeIndex=target_sphere, basePosition=target_pos)

    for i in range(5):
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
        # action = env.policy_obj(target_pos)

        # manual control
        action = np.array([(env.drone_mass + env.obj_mass)*9.81, 3.0, 0.0, 0.0])
        env.step(action)
    env.logger.plot('results/test')

if __name__ == "__main__":
    test()