from enum import Enum


class StateDim(Enum):
    pos = slice(0, 3)
    quat = slice(3, 7)
    linvel = slice(7, 10)
    angvel = slice(10, 13)
    vel = slice(7, 13)
    pose = slice(0, 7)
    all = slice(0, 13)


class RandKey(Enum):
    ob = 'observations'
    action = 'actions'
    sim_params = 'sim_params'
    dof_properties = 'dof_properties'
    scale = 'scale'
    color = 'color'
    rbp = 'rigid_body_properties'
    rsp = 'rigid_shape_properties'
    dof_prop = 'dof_properties'
    setup_only = ['scale', 'mass']
