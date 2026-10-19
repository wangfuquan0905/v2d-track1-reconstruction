"""Discrete SO(3) angular derivatives in the common camera coordinate frame."""
from pytorch3d.transforms import matrix_to_axis_angle


def angular_derivatives(rotations, fps):
    """Interval angular velocity (rad/s) and centred acceleration (rad/s^2).

    R maps object coordinates into camera coordinates. Log(R_next R_prev^T)
    therefore gives a spatial rotation vector in the SAME camera frame for
    every interval; its finite difference needs no body-frame transport.
    The principal log assumes true inter-frame rotation is less than pi.
    """
    relative = rotations[1:] @ rotations[:-1].transpose(-1, -2)
    velocity = matrix_to_axis_angle(relative) * fps
    acceleration = (velocity[1:]-velocity[:-1]) * fps
    return velocity, acceleration
