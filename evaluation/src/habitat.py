import numpy as np
import quaternion
import habitat_sim
import supervision as sv
import magnum as mn
from .geom import get_collision_distance, IoU
from habitat_sim.utils.common import (
    quat_to_coeffs,
    quat_from_angle_axis,
    quat_from_two_vectors,
    quat_to_angle_axis,
)


def pos_normal_to_habitat(pts):
    # +90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))


def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def pose_habitat_to_normal(pose):
    # T_normal_cam = T_normal_habitat * T_habitat_cam
    return np.dot(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), pose
    )


def pose_normal_to_tsdf(pose):
    return np.dot(
        pose, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    )


def pose_habitat_to_tsdf(pose):
    return pose_normal_to_tsdf(pose_habitat_to_normal(pose))


def pose_normal_to_tsdf_real(pose):
    # This one makes sense, which is making x-forward, y-left, z-up to z-forward, x-right, y-down
    return pose @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def make_semantic_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    sim_cfg.load_semantic_mesh = True
    sim_cfg.enable_physics = False
    # sim_cfg.gpu_device_id = 1

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor, a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    rgb_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    depth_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    semantic_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    semantic_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [
        rgb_sensor_spec,
        depth_sensor_spec,
        semantic_sensor_spec,
    ]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    rgb_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = habitat_sim.geo.UP * settings["sensor_height"]
    depth_sensor_spec.orientation = [settings["camera_tilt"], 0.0, 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def get_quaternion(angle, camera_tilt):
    normalized_angle = angle % (2 * np.pi)
    if np.abs(normalized_angle - np.pi) < 1e-6:
        return quat_to_coeffs(
            quaternion.quaternion(0, 0, 1, 0)
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()

    return quat_to_coeffs(
        quat_from_angle_axis(angle, np.array([0, 1, 0]))
        * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
    ).tolist()


def get_navigable_point_from(
    pos_start,
    pathfinder,
    max_search=1000,
    min_dist=6,
    max_dist=999,
    prev_start_positions=None,
    min_dist_from_prev=3,
    max_samples=100,
):
    if min_dist < 0:
        return None, None, None
    if max_dist <= min_dist:
        max_dist = min_dist + 1

    pos_end_list = []

    # if no previous start positions, just find a random navigable point that is the farthest among the searches
    # otherwise, find the point that is the farthest from the previous start positions, and also satisfies the min_dist
    try_count = 0
    while True:
        try_count += 1
        if try_count > max_search:
            break

        pos_end_current = pathfinder.get_random_navigable_point()
        if (
            np.abs(pos_end_current[1] - pos_start[1]) > 0.4
        ):  # make sure the end point is on the same level
            continue

        path = habitat_sim.ShortestPath()
        path.requested_start = pos_start
        path.requested_end = pos_end_current
        found_path = pathfinder.find_path(path)
        if found_path:
            if min_dist < path.geodesic_distance < max_dist:
                if prev_start_positions is None:
                    pos_end_list.append(pos_end_current)
                else:
                    if np.all(
                        np.linalg.norm(prev_start_positions - pos_end_current, axis=1)
                        > min_dist_from_prev
                    ):
                        pos_end_list.append(pos_end_current)

        if len(pos_end_list) >= max_samples:
            break

    if len(pos_end_list) == 0:
        # if no point is found that satisfies the min_dist, then find again with shorter min_dist
        return get_navigable_point_from(
            pos_start,
            pathfinder,
            max_search,
            min_dist - 0.5,
            max_dist + 2,
            prev_start_positions,
            min_dist_from_prev,
            max_samples,
        )

    # sample a ramdom point from the list
    while True:
        pos_end = pos_end_list[np.random.randint(len(pos_end_list))]
        path = habitat_sim.ShortestPath()
        path.requested_start = pos_start
        path.requested_end = pos_end
        found_path = pathfinder.find_path(path)
        if found_path:
            break
    return pos_end, path.points, path.geodesic_distance


def get_navigable_point_to(
    pos_end,
    pathfinder,
    max_search=1000,
    min_dist=6,
    max_dist=999,
    prev_start_positions=None,
):
    pos_start, path_point, travel_dist = get_navigable_point_from(
        pos_end, pathfinder, max_search, min_dist, max_dist, prev_start_positions
    )
    if pos_start is None or path_point is None:
        return None, None, None

    # reverse the path_point
    path_point = path_point[::-1]
    return pos_start, path_point, travel_dist


def get_frontier_observation(
    agent,
    simulator,
    cfg,
    tsdf_planner,
    view_frontier_direction,
    init_pts,
    camera_tilt=0,
    max_try_count=10,
):
    agent_state = habitat_sim.AgentState()

    # solve edge cases of viewing direction
    default_view_direction = np.asarray([0.0, 0.0, -1.0])
    if np.linalg.norm(view_frontier_direction) < 1e-3:
        view_frontier_direction = default_view_direction
    view_frontier_direction = view_frontier_direction / np.linalg.norm(
        view_frontier_direction
    )

    # convert view direction to voxel space
    # since normal to voxel is just scale and shift, we don't need that conversion
    view_dir_voxel = pos_habitat_to_normal(view_frontier_direction)[:2]

    # set agent observation direction
    if (
        np.dot(view_frontier_direction, default_view_direction)
        / np.linalg.norm(view_frontier_direction)
        < -1 + 1e-3
    ):
        # if the rotation is to rotate 180 degree, then the quaternion is not unique
        # we need to specify rotating along y-axis
        agent_state.rotation = quat_to_coeffs(
            quaternion.quaternion(0, 0, 1, 0)
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
    else:
        agent_state.rotation = quat_to_coeffs(
            quat_from_two_vectors(default_view_direction, view_frontier_direction)
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()

    occupied_map = tsdf_planner.occupied_map_camera

    try_count = 0
    pts = init_pts.copy()
    valid_observation = None
    while try_count < max_try_count:
        try_count += 1

        # check whether current view is valid
        collision_dist = tsdf_planner._voxel_size * get_collision_distance(
            occupied_map, pos=tsdf_planner.habitat2voxel(pts), direction=view_dir_voxel
        )

        if collision_dist >= cfg.collision_dist:
            # set agent state
            agent_state.position = pts

            # get observations
            agent.set_state(agent_state)
            # _ = self.simulator.get_sensor_observations()
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]
            semantic_obs = obs["semantic_sensor"]

            # check whether the observation is valid
            keep_observation = True
            black_pix_ratio = np.sum(semantic_obs == 0) / (
                cfg.img_height * cfg.img_width
            )
            if black_pix_ratio > cfg.black_pixel_ratio_frontier:
                keep_observation = False
            positive_depth = depth[depth > 0]
            if (
                positive_depth.size == 0
                or np.percentile(positive_depth, 30) < cfg.min_30_percentile_depth
            ):
                keep_observation = False
            if keep_observation:
                valid_observation = rgb
                break

        # update pts
        pts = pts - view_frontier_direction * tsdf_planner._voxel_size

    if valid_observation is not None:
        return valid_observation

    # if no valid observation is found, then just get the observation at init position
    agent_state.position = init_pts
    agent.set_state(agent_state)
    # _ = self.simulator.get_sensor_observations()
    obs = simulator.get_sensor_observations()
    return obs["color_sensor"]
