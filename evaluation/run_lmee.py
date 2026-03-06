import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import math
import time
import json
import logging
import matplotlib.pyplot as plt
from PIL import Image
import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.scene import Scene
from src.utils import resize_image, calc_agent_subtask_distance, get_pts_angle_goatbench
from src.lmeebench_utils import prepare_lmeebench_navigation_goals
from src.query_mllm_lmee import query_mllm_for_response
from src.logger_lmeebench import Logger
from transformers import AutoModelForImageTextToText, AutoProcessor

def main(cfg, mllm_model, processor, start_ratio=0.0, end_ratio=1.0, split=1, answer_type="choice"):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    scene_data_list = os.listdir(cfg.test_data_dir)
    num_scene = len(scene_data_list)
    random.shuffle(scene_data_list)

    # split the test data by scene
    scene_data_list = scene_data_list[int(start_ratio * num_scene) : int(end_ratio * num_scene)]
    num_episode = 0
    for scene_data_file in scene_data_list:
        with open(os.path.join(cfg.test_data_dir, scene_data_file), "r") as f:
            num_episode += len(json.load(f))
    logging.info(
        f"Total number of episodes: {num_episode}"
    )
    logging.info(f"Total number of scenes: {len(scene_data_list)}")
    all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(
        cfg.scene_data_path + "/val"
    )

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size)

    for scene_data_file in scene_data_list:
        # load goatbench data
        scene_name_full = scene_data_file.split(".")[0]
        scene_name = scene_name_full.split('-')[1]

        scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
        scene_data = json.load(
            open(os.path.join(cfg.test_data_dir, scene_data_file), "r")
        )

        scene_goatbench = os.path.join(cfg.goat_data_dir, scene_name + ".json")
        scene_goatbench_data = json.load(open(scene_goatbench, "r"))

        # selecat the episodes according to the split
        total_episodes = len(scene_data)

        all_navigation_goals = scene_goatbench_data["goals"]  # obj_id to obj_data, apply for all episodes in this scene

        for episode_idx, episode in enumerate(scene_data):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode_idx
            task_instruct = episode['Task instruction']
            diff = episode['Difficulty']
            
            all_subtask_goal_types, all_subtask_goals = (
                prepare_lmeebench_navigation_goals(
                    scene_name=scene_name, # no index
                    episode=episode,
                    all_navigation_goals=all_navigation_goals,
                )
            )
            count_not_QA = sum(t == 'QA' for t in all_subtask_goal_types)
            # check whether this episode has been processed
            finished_subtask_ids = list(logger.success_by_snapshot.keys())
            finished_episode_subtask = [
                subtask_id
                for subtask_id in finished_subtask_ids
                if subtask_id.startswith(f"{scene_id}_{episode_id}_")
            ]
            if len(finished_episode_subtask) >= count_not_QA:
                logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                continue

            pts, angle = get_pts_angle_goatbench(episode["Start_position"][0], episode["Start_rotation"][0])

            # load scene
            try:
                scene.simulator.close()
                del scene
            except:
                pass
            
            scene = Scene(
                scene_id,
                cfg,
                cfg_cg,
                detection_model,
                sam_predictor,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
            )

            # initialize the TSDF
            floor_height = pts[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)
            # num_step = min(num_step, 5)
            logging.info(f"Scene max step: {num_step}")
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=floor_height,
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )

            episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )

            logging.info(f"\n\nScene {scene_id} initialization successful!")

            # run questions in the scene
            global_step = -1
            for subtask_idx, (goal_type, subtask_goal) in enumerate(
                zip(all_subtask_goal_types, all_subtask_goals)
            ):
                subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
                logging.info(
                    f"\nScene {scene_id} Episode {episode_id} Subtask {subtask_idx + 1}/{len(all_subtask_goals)}"
                )

                # if goal_type == "QA":
                #     continue
                    
                subtask_metadata = logger.init_subtask(
                    subtask_id=subtask_id,
                    goal_type=goal_type,
                    answer_type=answer_type,
                    diff=diff,
                    subtask_goal=subtask_goal,
                    task_instruct=task_instruct,
                    pts=pts,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                )

                # breakpoint()
                # mapping from the obj id in habitat to the id assigned by concept graph
                # this mapping/alignment is done by heuristic matching between object masks
                goal_obj_ids_mapping = {
                    obj_id: [] for obj_id in subtask_metadata["goal_obj_ids"]
                }

                # run steps
                task_success = False
                cnt_step = -1
                n_filtered_snapshots = 0
                pred_answer = None

                # reset tsdf planner
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
                max_point_choice = None

                if cfg.clear_up_memory_every_subtask and subtask_idx > 0:
                    scene.clear_up_detections()
                    tsdf_planner = TSDFPlanner(
                        vol_bnds=tsdf_bnds,
                        voxel_size=cfg.tsdf_grid_size,
                        floor_height=floor_height,
                        floor_height_offset=0,
                        pts_init=pts,
                        init_clearance=cfg.init_clearance * 2,
                        save_visualization=cfg.save_visualization,
                    )

                while cnt_step < num_step - 1:
                    cnt_step += 1
                    global_step += 1
                    logging.info(
                        f"\n== step: {cnt_step}, global step: {global_step} =="
                    )

                    # (1) Observe the surroundings, update the scene graph and occupancy map
                    # Determine the viewing angles for the current step
                    if cnt_step == 0 or global_step == 1:
                    # if cnt_step == 0 or cnt_step == 1:
                        angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_2
                    else:
                        angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_1
                    all_angles = [
                        angle + angle_increment * (i - total_views // 2)
                        for i in range(total_views)
                    ]
                    # Let the main viewing angle be the last one to avoid potential overwriting problems
                    main_angle = all_angles.pop(total_views // 2)
                    all_angles.append(main_angle)

                    rgb_egocentric_views = []
                    all_added_obj_ids = (
                        []
                    )  # Record all the objects that are newly added in this step
                    for view_idx, ang in enumerate(all_angles):
                        # For each view
                        # print("angle", ang)
                        obs, cam_pose = scene.get_observation(pts, angle=ang)
                        rgb = obs["color_sensor"]
                        depth = obs["depth_sensor"]
                        semantic_obs = obs["semantic_sensor"]

                        # collect all view features
                        obs_file_name = f"{global_step}-view_{view_idx}.png"
                        with torch.no_grad():
                            # Concept graph pipeline update
                            annotated_rgb, added_obj_ids, target_obj_id_mapping = (
                                scene.update_scene_graph(
                                    image_rgb=rgb[..., :3],
                                    depth=depth,
                                    intrinsics=cam_intr,
                                    cam_pos=cam_pose,
                                    pts=pts,
                                    pts_voxel=tsdf_planner.habitat2voxel(pts),
                                    img_path=obs_file_name,
                                    frame_idx=cnt_step * total_views + view_idx,
                                    semantic_obs=semantic_obs,
                                    gt_target_obj_ids=subtask_metadata["goal_obj_ids"],
                                )
                            )
                            scene.all_observations[obs_file_name] = rgb
                            rgb_egocentric_views.append(resize_image(rgb, cfg.prompt_h, cfg.prompt_w))
                            if cfg.save_visualization:
                                plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), annotated_rgb,)
                            else:
                                plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)
                            # if cfg.save_visualization:
                            #     img = Image.fromarray(annotated_rgb)
                            # else:
                            #     img = Image.fromarray(rgb)

                            # img = img.resize((360, 360))
                            # plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), img)

                            # update the mapping of hm3d object id to our detected object id
                            for ( gt_goal_id, det_goal_id) in target_obj_id_mapping.items():
                                goal_obj_ids_mapping[gt_goal_id].append(det_goal_id)
                            all_added_obj_ids += added_obj_ids

                        # Clean up or merge redundant objects periodically
                        scene.periodic_cleanup_objects(
                            frame_idx=cnt_step * total_views + view_idx,
                            pts=pts,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,
                        )

                        # Update depth map, occupancy map
                        tsdf_planner.integrate(
                            color_im=rgb,
                            depth_im=depth,
                            cam_intr=cam_intr,
                            cam_pose=pose_habitat_to_tsdf(cam_pose),
                            obs_weight=1.0,
                            margin_h=int(cfg.margin_h_ratio * img_height),
                            margin_w=int(cfg.margin_w_ratio * img_width),
                            explored_depth=cfg.explored_depth,
                        )
                    logging.info(f"Goal object mapping: {goal_obj_ids_mapping}")

                    # (2) Update Memory Snapshots with hierarchical clustering
                    # Choose all the newly added objects as well as the objects nearby as the cluster targets
                    all_added_obj_ids = [obj_id for obj_id in all_added_obj_ids if obj_id in scene.objects]
                    for obj_id, obj in scene.objects.items():
                        if (
                            np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                            < cfg.scene_graph.obj_include_dist + 0.5
                        ):
                            all_added_obj_ids.append(obj_id)
                    scene.update_snapshots(
                        obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
                    )
                    logging.info(
                        f"Step {cnt_step}, update memories, {len(scene.objects)} objects, {len(scene.snapshots)} memories"
                    )
                    # (3) Update the Frontier Snapshots
                    update_success = tsdf_planner.update_frontier_map(
                        pts=pts,
                        cfg=cfg.planner,
                        scene=scene,
                        cnt_step=cnt_step,
                        save_frontier_image=cfg.save_visualization,
                        eps_frontier_dir=eps_frontier_dir,
                        prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                    )
                    if not update_success:
                        logging.info("Warning! Update frontier map failed!")

                    # (4) Choose the next navigation point by querying the VLM
                    if cfg.choose_every_step:
                        # if we choose to query vlm every step, we clear the target point every step
                        if (tsdf_planner.max_point is not None and type(tsdf_planner.max_point) == Frontier):
                            # reset target point to allow the model to choose again
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None

                    # use the most common id in the mapped ids as the detected target object id
                    target_obj_ids_estimate = []
                    for obj_id, det_ids in goal_obj_ids_mapping.items():
                        if len(det_ids) == 0:
                            continue
                        target_obj_ids_estimate.append(
                            max(set(det_ids), key=det_ids.count)
                        )

                    if (
                        tsdf_planner.max_point is None
                        and tsdf_planner.target_point is None
                    ):
                        # query the VLM for the next navigation point, and the reason for the choice
                        vlm_response = query_mllm_for_response(
                            subtask_metadata=subtask_metadata,
                            scene=scene,
                            tsdf_planner=tsdf_planner,
                            rgb_egocentric_views=rgb_egocentric_views,
                            cfg=cfg,
                            model=mllm_model,
                            processor=processor,
                            verbose=True,
                        )
                        if vlm_response is None:
                            logging.info(
                                f"Subtask id {subtask_id} invalid: query_mllm_for_response failed!"
                            )
                            break
                        if goal_type == "QA":
                            max_point_choice, pred_answer, n_filtered_snapshots = vlm_response
                        else:
                            max_point_choice, n_filtered_snapshots = vlm_response

                        # set the vlm choice as the navigation target
                        update_success = tsdf_planner.set_next_navigation_point(
                            choice=max_point_choice,
                            pts=pts,
                            objects=scene.objects,
                            cfg=cfg.planner,
                            pathfinder=scene.pathfinder,
                        )
                        if not update_success:
                            logging.info(f"Subtask id {subtask_id} invalid: set_next_navigation_point failed!")
                            break

                    # (5) Agent navigate to the target point for one step
                    return_values = tsdf_planner.agent_step(
                        pts=pts,
                        angle=angle,
                        objects=scene.objects,
                        snapshots=scene.snapshots,
                        pathfinder=scene.pathfinder,
                        cfg=cfg.planner,
                        path_points=None,
                        save_visualization=cfg.save_visualization,
                    )
                    if return_values[0] is None:
                        logging.info(
                            f"Subtask id {subtask_id} invalid: agent_step failed!"
                        )
                        break

                    # update agent's position and rotation
                    pts, angle, pts_voxel, fig, _, target_arrived = return_values
                    logger.log_step(pts_voxel=pts_voxel)
                    logging.info(
                        f"Current position: {pts}, {logger.subtask_explore_dist:.3f}"
                    )

                    # sanity check about objects, scene graph, snapshots, ...
                    scene.sanity_check(cfg=cfg)

                    if cfg.save_visualization:
                        # save the top-down visualization
                        logger.save_topdown_visualization(
                            global_step=global_step,
                            subtask_id=subtask_id,
                            subtask_metadata=subtask_metadata,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,
                            fig=fig,
                        )
                        # save the visualization of vlm's choice at each step
                        logger.save_frontier_visualization(
                            global_step=global_step,
                            subtask_id=subtask_id,
                            tsdf_planner=tsdf_planner,
                            max_point_choice=max_point_choice,
                            global_caption=f"{subtask_metadata['question']}\n{subtask_metadata['task_type']}\n{subtask_metadata['class']}",
                        )

                    # (6) Check if the agent has arrived at the target to finish the question
                    if type(max_point_choice) == SnapShot and target_arrived:
                        # when the target is a snapshot, and the agent arrives at the target
                        # we consider the subtask is finished, take an observation and save the chosen target snapshot
                        obs, _ = scene.get_observation(pts, angle=angle)
                        rgb = obs["color_sensor"]
                        # img = Image.fromarray(rgb)
                        # img = img.resize((360, 360))
                        plt.imsave(os.path.join(logger.subtask_object_observe_dir, f"target.png"), rgb,)

                        snapshot_filename = max_point_choice.image.split(".")[0]
                        os.system(
                            f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(logger.subtask_object_observe_dir, f'snapshot_{snapshot_filename}.png')}"
                        )
                        task_success = True
                        break

                # get some statistics
                if task_success and np.any(
                    [
                        obj_id in max_point_choice.cluster
                        for obj_id in target_obj_ids_estimate
                    ]
                ):
                    success_by_snapshot = True
                    logging.info(f"Success: {target_obj_ids_estimate} in chosen memory {max_point_choice.image}!")
                else:
                    success_by_snapshot = False
                    logging.info(f"Fail: {target_obj_ids_estimate} not in chosen memory!")
                # calculate the distance to the nearest view point
                agent_subtask_distance = calc_agent_subtask_distance(
                    pts, subtask_metadata["viewpoints"], scene.pathfinder
                )
                if agent_subtask_distance < cfg.success_distance:
                    success_by_distance = True
                    logging.info(f"Success: agent reached the target viewpoint at distance {agent_subtask_distance}!")
                else:
                    success_by_distance = False
                    logging.info(f"Fail: agent failed to reach the target viewpoint at distance {agent_subtask_distance}!")

                logger.log_subtask_result(
                    success_by_snapshot=success_by_snapshot,
                    success_by_distance=success_by_distance,
                    pred_answer=pred_answer,
                    subtask_id=subtask_id,
                    subtask_metadata=subtask_metadata,
                    gt_subtask_explore_dist=subtask_metadata["gt_subtask_explore_dist"],
                    goal_type=goal_type,
                    task_step=cnt_step+1,
                    n_filtered_snapshots=n_filtered_snapshots,
                    n_total_snapshots=len(scene.snapshots),
                    n_total_frames=len(scene.frames),
                )

                logging.info(f"Scene graph of question {subtask_id}:")
                logging.info(f"Question: {subtask_metadata['question']}")
                logging.info(f"Task type: {subtask_metadata['task_type']}")
                logging.info(f"Answer: {subtask_metadata['class']}")
                # scene.print_scene_graph()

                if not cfg.save_visualization:
                    # clear up the stored images to save memory
                    os.system(
                        f"rm -r {os.path.join(str(cfg.output_dir), f'{subtask_id}')}"
                    )

            # save the results at the end of each episode
            logger.save_results()

            logging.info(f"Episode {episode_id} finish")
            if not cfg.save_visualization:
                os.system(f"rm -r {episode_dir}")

    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    logging.info(f"All scenes finish")

def load_model(cfg):
    pretrained = cfg.model.pretrained_path
    torch_dtype = cfg.model.torch_dtype
    device = cfg.model.device

    model = AutoModelForImageTextToText.from_pretrained(
        pretrained,
        torch_dtype=torch_dtype,
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(pretrained)
    model.eval()
    return model, processor

if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="cfg/eval_goatbench.yaml", type=str)
    parser.add_argument("--answer_type",help="answer type",default="open",type=str,choices=["choice", "open"])
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir),
        f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log",
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


    # load mllm
    mllm_model, processor = load_model(cfg)
    
    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, mllm_model, processor, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=args.split, answer_type=args.answer_type)
