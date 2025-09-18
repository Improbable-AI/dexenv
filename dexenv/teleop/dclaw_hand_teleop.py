import hydra
import isaacgym
import os
import time
from omegaconf import DictConfig

import dexenv
from dexenv.utils.common import make_dir, set_print_formatting, set_random_seed
from dexenv.utils.create_task_env import create_task_env

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

def load_dclaw_urdf():
    # Path to your DClaw URDF
    urdf_path = "/workspace/dexenv/dexenv/assets/dclaw_4f/dclaw_4f.urdf"
    model_path = os.path.dirname(urdf_path)

    # Load the model
    model = pin.buildModelFromUrdf(urdf_path)
    # data = model.createData()
    # Optionally wrap in RobotWrapper if you want higher-level utilities
    robot = RobotWrapper.BuildFromURDF(urdf_path, [model_path])
    
    q0 = pin.neutral(model)

@hydra.main(config_path=dexenv.PROJECT_ROOT.joinpath('conf').as_posix(), config_name="debug_dclaw")
def main(cfg: DictConfig):
    make_dir(os.environ['WANDB_DIR'])
    set_random_seed(cfg.alg.seed)
    set_print_formatting()

    env = create_task_env(cfg)
    obs = env.reset()

    # Load URDF & kinematics model
    robot = load_dclaw_urdf()

    # for t in range(cfg.control.max_steps):
    #     # === Step 1: Compute end-effector pose ===
    #     desired_pose = get_desired_pose_from_teleop()

    #     # === Step 2: Use IK solver to get joint angles ===
    #     desired_joint_angles = solve_ik(robot, desired_pose['position'])

    #     # Get the current joint angles from the environment
    #     current_joint_angles = 0 # []

    #     # Compute the difference between desired_joint_angles and current joint angles to get the difference
    #     relative_joint_angle = desired_joint_angles - current_joint_angles

    #     # === Step 3: Send relative joint angle changes to the environment ===
    #     action = relative_joint_angle
    #     obs, reward, done, info = env.step(action)

    #     if cfg.render:
    #         env.render()
    #     time.sleep(0.01)

    #     if done:
    #         print("Episode finished")
    #         obs = env.reset()

# Dummy placeholders
def get_desired_pose_from_teleop():
    # Return target position/orientation for the end-effector
    return {
        'position': [0.1, 0.0, 0.2],
        'orientation': [0.0, 0.0, 0.0, 1.0]  # Quaternion
    }

def ik_solver(desired_pose):
    # Use your IK module to return joint angles given pose
    # Example: return np.array([q1, q2, q3, ...])
    raise NotImplementedError("Insert IK solver here")

if __name__ == '__main__':
    main()
