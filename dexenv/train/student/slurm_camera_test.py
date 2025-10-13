from isaacgym import gymapi
import os

# Force EGL/headless for cluster
os.environ.pop("DISPLAY", None)
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["ISAAC_USE_EGL"] = "1"
os.environ["__GL_ALLOW_UNSUPPORTED_PLATFORM"] = "1"

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.use_gpu_pipeline = True
# ... set your usual physx params here ...

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)  # device 0, headless/EGL
assert sim

lower = gymapi.Vec3(-1, -1, -1)
upper = gymapi.Vec3( 1,  1,  1)
env = gym.create_env(sim, lower, upper, 1)
assert env

# (Optional) create one lightweight actor here if you normally do.

gym.prepare_sim(sim)

props = gymapi.CameraProperties()
props.width = 320
props.height = 240
props.horizontal_fov = 70.0
props.near_plane = 0.01
props.far_plane = 10.0
props.enable_tensors = True

cam = gym.create_camera_sensor(env, props)
print("Camera handle:", cam)
