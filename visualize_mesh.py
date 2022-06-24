import argparse
from pathlib import Path
import os

import open3d as o3d
import numpy as np


import util


def build_args():

    parser = argparse.ArgumentParser('3d Visualization', add_help=False)
    parser.add_argument('--scene_name', required=True, type=str)
    parser.add_argument('--base_dir', required=True, type=Path)
    parser.add_argument('--seg_out_dir', required=True, type=Path)
    parser.add_argument('--visualize_type', required=True, type=str)
    parser.add_argument('--skip_nums', required=False, default=1, type=int)
    parser.add_argument('--seg_type', required=True, default="pred", type=str)
    args = parser.parse_args()

    return args


def init_viz():

    viz = o3d.visualization.Visualizer()
    viz.create_window()

    return viz


def visualize_mesh(args, viz):

    scene_path = args.base_dir / Path(args.scene_name)
    scene_mesh = util.load_mesh(args.base_dir, args.scene_name)

    # start to show !
    viz.add_geometry(scene_mesh)
    viz.update_renderer()
    viz.run()

    # close the process by closing the window


def visualize_seg(args, viz):

    scene_path = args.base_dir / Path(args.scene_name)
    scene_mesh = util.load_mesh(args.base_dir, args.scene_name)
    scene_mesh_vertices = np.asarray(scene_mesh.vertices)

    if(args.seg_type == "pred"):
        seg_image_path = os.listdir(args.seg_out_dir / Path(args.scene_name))
    else:
        seg_image_path = os.listdir(os.path.join(scene_path, "label-filt"))

    for text_id in range(1, len(seg_image_path), args.skip_nums):

        # prepare some basic data
        depth_img = util.load_depth(scene_path, text_id)
        depth_intrinsic = util.load_intrinsic(scene_path, "depth")
        label_img = util.load_label((args.seg_out_dir / Path(args.scene_name)),
                                    text_id, depth_img.shape, args.seg_type)
        pose_matrix, camera_translation = util.load_pose(scene_path, text_id)

        # we first project the mesh vertices back to image plane
        vertex_projection, vertex_camera_coord = util.projectToFrame(
            scene_mesh_vertices, depth_intrinsic, pose_matrix, camera_translation)

        # camera visualization
        lines = util.visualizeCameraPosition(depth_intrinsic, pose_matrix,
                                             camera_translation, depth_img.shape)

        # finally ! let's back project the label, we actually can store a memory scene_label
        # array that we can further calculate maximum posterior probability label, but let's
        # leave it in the next intro
        scene_mesh = util.label_mesh(scene_mesh,
                                     depth_img, label_img, lines,
                                     vertex_projection, vertex_camera_coord,
                                     viz, text_id)

        # to make the back-projection progress more easy to observe
        # we specify the viewing extrinsic
        viz_extrinsic = np.array([[0.1589852858550839, 0.81266088242727896, -0.56062997516552193, 0.0],
                                  [0.98722197737634287, -0.12465113670926495, 0.099271655079542942, 0.0],
                                  [0.010791027139288546, -0.56924896511948497, -0.8220943798881456, 0.0],
                                  [-5.4297251120876489, -1.93594437975616, 8.4208078448793628, 1.0]]).T

        viz.add_geometry(scene_mesh)
        viz.add_geometry(lines)
        control = viz.get_view_control()
        param = control.convert_to_pinhole_camera_parameters()
        param.extrinsic = viz_extrinsic
        control.convert_from_pinhole_camera_parameters(param)

        viz.update_geometry(scene_mesh)
        viz.update_geometry(lines)
        viz.poll_events()
        viz.update_renderer()

        viz.remove_geometry(lines)

    # we stop at the final scene to let us have more observe
    viz.add_geometry(scene_mesh)
    viz.update_renderer()
    viz.run()


if __name__ == '__main__':

    args = build_args()

    viz = init_viz()

    if(args.visualize_type == "mesh"):
        visualize_mesh(args, viz)

    elif(args.visualize_type == "seg_mesh"):
        visualize_seg(args, viz)
