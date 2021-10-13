import numpy as np
from .realur5_utils import (
    connect,
    UR5State,
    Gripper)
from time import sleep, time
from copy import deepcopy


def clamp_angles(angle, up=np.pi, down=-np.pi):
    angle[angle > up] -= up
    angle[angle < down] += down
    return angle


class UR5MoveTimeoutException(Exception):
    def __init__(self):
        super().__init__('UR5 Move Timeout')


class UR5:
    # joint position and tool pose tolerance (epsilon) for blocking calls
    tool_pose_eps = np.array([1e-2]*3 + [1.0]*3)

    GROUPS = {
        'arm': ['shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint'],
        'gripper': ['finger_joint',
                    'left_inner_knuckle_joint',
                    'left_inner_finger_joint',
                    'right_outer_knuckle_joint',
                    'right_inner_knuckle_joint',
                    'right_inner_finger_joint']
    }

    GROUP_INDEX = {
        'arm': [1, 2, 3, 4, 5, 6],
        'gripper': [9, 11, 13, 14, 16, 18]
    }

    LINK_COUNT = 10

    LIE = [0, 0, 0, 0, 0, 0]
    UP = [0, -1.5707, 0, -1.5707, 0, 0]
    EE_LINK_NAME = 'ee_link'
    TIP_LINK = "ee_link"
    BASE_LINK = "base_link"
    ARM = "arm"
    GRIPPER = "gripper"
    EE_TIP_LINK = 7
    TOOL_LINK = 6
    HOME = [-np.pi, -np.pi / 2,
            np.pi / 2, -np.pi / 2,
            -np.pi / 2, 0]
    RESET = [-np.pi, -np.pi / 2,
             np.pi / 2, -np.pi / 2,
             -np.pi / 2, 0]

    # this is read from moveit_configs joint_limits.yaml
    MOVEIT_ARM_MAX_VELOCITY = [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]

    LOWER_LIMIT = np.array([-2, -2, -1, -2, -2, -2]) * np.pi
    UPPER_LIMIT = np.array([2, 2, 1, 2, 2, 2]) * np.pi

    JOINT_EPSILON = 1e-2

    def __init__(self,
                 tcp_ip,
                 velocity=1e-2,
                 acceleration=1e-2,
                 tcp_port=30002,
                 rtc_port=30003,
                 gripper: Gripper = None):
        self.tcp_ip = tcp_ip
        self.velocity = velocity
        self.acceleration = acceleration

        self.create_tcp_sock_fn = lambda: connect(tcp_ip, tcp_port)
        self.create_rtc_sock_fn = lambda: connect(tcp_ip, rtc_port)
        self.tcp_sock = self.create_tcp_sock_fn()
        self.rtc_sock = self.create_rtc_sock_fn()

        self.state = UR5State(
            self.create_tcp_sock_fn,
            self.create_rtc_sock_fn)

        self.gripper = gripper
        if self.gripper is not None:
            tcp_msg = 'set_tcp(p[%f,%f,%f,%f,%f,%f])\n'\
                % tuple(self.gripper.tool_offset)
            self.tcp_sock.send(str.encode(tcp_msg))
        self.use_pos = False
        self.time_start_command = None
        self.action_timeout = 10

    # Move joints to specified positions or move tool to specified pose

    def movej(self, **kwargs):
        return self.move('j', **kwargs)

    def movel(self, **kwargs):
        return self.move('l', **kwargs)

    def check_pose_reachable(self, pose):
        from real_world.setup import MIN_UR5_BASE_SAFETY_RADIUS
        norm = np.linalg.norm(pose[:2])
        return norm < 0.90\
            and norm > MIN_UR5_BASE_SAFETY_RADIUS

    def move(self, move_type, params,
             blocking=True,
             j_acc=None, j_vel=None,
             times=0.0, blend=0.0,
             clear_state_history=False, use_pos=False):
        self.use_pos = use_pos
        params = deepcopy(params)
        if not j_acc:
            j_acc = self.acceleration
        if not j_vel:
            j_vel = self.velocity
        multiple_params = any(isinstance(item, list) for item in params)
        if type(params) != np.array:
            params = np.array(params)
        if multiple_params:
            def match_param_len(var):
                if not isinstance(var, list):
                    return [var] * len(params)
                elif len(var) != len(params):
                    raise Exception()
                return var
            j_vel = match_param_len(j_vel)
            j_acc = match_param_len(j_acc)
            move_type = match_param_len(move_type)
            times = match_param_len(times)
            blend = match_param_len(blend)
        else:
            params = [params]
            assert not isinstance(j_vel, list) and \
                not isinstance(j_acc, list)
            j_vel = [j_vel]
            j_acc = [j_acc]
            move_type = [move_type]
            times = [times]
            blend = [blend]
        if use_pos:
            # check all poses are reachable
            if not all([self.check_pose_reachable(pose=param)
                        for param in params]):
                return False
        self.curr_targ = params[-1]
        if self.use_pos:
            self.curr_targ[-3:] = clamp_angles(self.curr_targ[-3:])

        # Move robot
        tcp_msg = 'def process():\n'
        tcp_msg += f' stopj({j_acc[0]})\n'
        for m, p, a, v, t, r in zip(
                move_type, params, j_acc, j_vel, times, blend):
            tcp_msg += ' move%s(%s[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=%f,r=%f)\n' \
                % (m, 'p' if use_pos else '',
                    p[0], p[1], p[2], p[3], p[4], p[5],
                    a, v, t, r)
        tcp_msg += 'end\n'
        if clear_state_history:
            self.state.clear()
            while not len(self.state):
                sleep(0.001)
        self.tcp_sock.send(str.encode(tcp_msg))

        # If blocking call, pause until robot stops moving
        if blocking:
            self.time_start_command = time()
            while True:
                print('\r ', end='')  # IO so scheduler prioritizes process
                if self.reached_target():
                    self.time_start_command = None
                    return True
                elif self.is_timed_out():
                    self.time_start_command = None
                    raise UR5MoveTimeoutException
        return True

    def is_timed_out(self):
        if self.time_start_command is None:
            return False
        return float(time()-self.time_start_command) > self.action_timeout

    def reached_target(self, only_check_pos=True):
        if not (self.state.get_j_vel() < 1e-1).all():
            return False
        if self.use_pos:
            tool_pose = self.state.get_ee_pose()
            tool_pose_mirror = np.asarray(list(tool_pose))
            tool_pose_mirror[-3:] = clamp_angles(tool_pose_mirror[-3:])
            tool_pose_mirror[3:6] = clamp_angles(-tool_pose_mirror[3:6])
            err = np.abs(tool_pose-self.curr_targ)
            err_mirror = np.abs(tool_pose_mirror-self.curr_targ)
            vel_residual = np.sum(np.abs(self.state.get_j_vel()))

            err_acceptable = err < self.tool_pose_eps
            err_mirror_acceptable = err_mirror < self.tool_pose_eps
            if only_check_pos:
                err_acceptable = err_acceptable[:3]
                err_mirror_acceptable = err_mirror_acceptable[:3]
            return (
                (err_acceptable).all() or
                (err_mirror_acceptable).all())\
                and vel_residual < 0.01
        else:
            return (np.abs((self.state.get_j_pos() - self.curr_targ))
                    < UR5.JOINT_EPSILON).all()

    # Move joints to home joint configuration
    def homej(self, **kwargs):
        self.movej(params=self.RESET, **kwargs)

    def reset(self):
        self.homej()
