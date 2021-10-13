from .setup import DEFAULT_ORN, WORKSPACE_SURFACE, DIST_UR5


def fling(ur5_pair,
          grasp_width: float = 0.36,  # Safe range 0.15 to 0.5
          height: float = 0.3,  # Safe range 0.0 to 0.4
          real_orn_1_e=[1.74, 1.74, -0.74],
          real_orn_2_e=[2.5, 2.5, 1.0],
          left_grasping=True,
          right_grasping=True,
          j_acc=[5.0, 4.0, 0.5],
          j_vel=[1.4, 1.4, 0.5],
          blend=[0.15, 0.099, 0.0],
          back_dist=0.30,
          front_dist=0.40,
          touchdown_1=0.20,
          touchdown_2=0.30):
    dx = (DIST_UR5 - grasp_width)/2
    if grasp_width > 0.5:
        j_vel = [1.0, 1.0, 0.5]
    if right_grasping and left_grasping:
        ur5_pair.movel(
            params=[
                # left
                [dx, back_dist, height, *real_orn_2_e],
                # right
                [dx, -back_dist, height, *real_orn_1_e]],
            blocking=True, use_pos=True)
        ur5_pair.move(
            move_type='l',
            params=[
                # left
                [[dx+0.02, -front_dist, height, *real_orn_1_e],
                 [dx, touchdown_1, WORKSPACE_SURFACE + \
                  0.02] + DEFAULT_ORN,
                 [dx, touchdown_2, WORKSPACE_SURFACE + 0.02] + DEFAULT_ORN],
                # right
                [[dx+0.02, front_dist, height,  *real_orn_2_e],
                 [dx, -touchdown_1, WORKSPACE_SURFACE + \
                  0.02] + DEFAULT_ORN,
                 [dx, -touchdown_2, WORKSPACE_SURFACE + 0.02] + DEFAULT_ORN]],
            blocking=True, j_acc=j_acc, j_vel=j_vel, blend=blend, use_pos=True)
        ur5_pair.open_grippers()
        ur5_pair.movel(
            params=[
                # left
                [dx, touchdown_2, WORKSPACE_SURFACE + \
                 0.02] + DEFAULT_ORN,
                # right
                [dx, -touchdown_2, WORKSPACE_SURFACE + 0.02] + DEFAULT_ORN],
            blocking=True, use_pos=True)
    elif right_grasping and not left_grasping:
        ur5_pair.movel(
            params=[
                # left
                [0.31, 0, 0.2] + DEFAULT_ORN,
                # right
                [DIST_UR5/2, -back_dist, height, *real_orn_1_e]],
            blocking=True, use_pos=True)
        ur5_pair.move(
            move_type='l',
            params=[
                # left
                [[0.31, 0, 0.2] + DEFAULT_ORN]*3,
                # right
                [[DIST_UR5/2, front_dist, height,  *real_orn_2_e],
                 [DIST_UR5/2, -touchdown_1, WORKSPACE_SURFACE + \
                  0.02] + DEFAULT_ORN,
                 [DIST_UR5/2, -touchdown_2, WORKSPACE_SURFACE + 0.02] + DEFAULT_ORN]],
            blocking=True, j_acc=j_acc, j_vel=j_vel, blend=blend, use_pos=True)
        ur5_pair.open_grippers()
        ur5_pair.move(
            move_type='l',
            params=[
                # left
                [0.31, 0, 0.2] + DEFAULT_ORN,
                # right
                [DIST_UR5/2, -touchdown_2, WORKSPACE_SURFACE + 0.02] + DEFAULT_ORN],
            blocking=True, use_pos=True)
    elif not right_grasping and left_grasping:
        ur5_pair.movel(
            params=[
                # left
                [DIST_UR5/2, back_dist, height, *real_orn_2_e],
                # right
                [0.31, 0, 0.2] + DEFAULT_ORN],
            blocking=True, use_pos=True)
        ur5_pair.move(
            move_type='l',
            params=[
                # left
                [[DIST_UR5/2, -front_dist, height, *real_orn_1_e],
                 [DIST_UR5/2, touchdown_1, WORKSPACE_SURFACE + \
                  0.02] + DEFAULT_ORN,
                 [DIST_UR5/2, touchdown_2, WORKSPACE_SURFACE + 0.02] + DEFAULT_ORN],
                # right
                [[0.31, 0, 0.2] + DEFAULT_ORN]*3],
            blocking=True, j_acc=j_acc, j_vel=j_vel, blend=blend, use_pos=True)
        ur5_pair.open_grippers()
        ur5_pair.move(
            move_type='l',
            params=[
                # left
                [DIST_UR5/2, touchdown_2, WORKSPACE_SURFACE + \
                 0.02] + DEFAULT_ORN,
                # right
                [0.31, 0, 0.2] + DEFAULT_ORN],
            blocking=True, use_pos=True)
