import csv
import numpy as np
from pathlib import Path


import open3d as o3d
import imageio
import cv2


def load_mesh(base_dir: Path, scene_name: str):
    """
    This is the function to load the mesh data
    base_dir: the Path of the root of all data
    scene_name: the string of the scene to visualize
    """

    mesh_suffix = scene_name + "_vh_clean_2.ply"
    mesh_path = base_dir / Path(scene_name) / Path(mesh_suffix)

    return o3d.io.read_triangle_mesh(str(mesh_path))


def load_depth(scene_name: Path, img_index: int):
    """
    This is the function to load depth image
    scene_name: the string of the scene to visualize
    img_index: image index to retrieve
    """
    depth_path = scene_name / "depth" / (str(img_index)+".png")

    return np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0


def load_color(scene_name: Path, img_index: int):
    """
    This is the function to load rgb image
    scene_name: the string of the scene to visualize
    img_index: image index to retrieve
    """

    color_path = scene_name / "color" / (str(img_index)+".jpg")

    return np.array(imageio.imread(color_path))


def load_intrinsic(scene_name: Path, intrinsic_type: str):
    """
    This is the function to load camera intrinsic info
    scene_name: the string of the scene to visualize
    intrinsic_type: type of camera to get, rgb or depth
    """

    intrinsic_path = scene_name / "intrinsic" / ("intrinsic_"+intrinsic_type+".txt")

    return np.loadtxt(intrinsic_path, dtype=np.float32)


def load_label(scene_name: Path, img_index: int, size, seg_type: str):
    """
    This is the function to load segmentation output image,
    it can be ground truth or predicted
    scene_name: the string of the scene to visualize
    img_index: image index to retrieve
    size: the size that match with depth image
    seg_type: the type of segmentation image, if the seg is of type gt, it need to be convert
                by tsv map, else just return the label

    TODO !!!:
        We now only support labelling the mesh by its segmentation id but not its instance id, although
        you can simply change the division of 1000 into modulus (%) to get instance, but it should be
        error there, try to fix it by applying Hungarian algorithm and we'll think how to evaluate
        in the future.
    """

    label_path = scene_name / (str(img_index)+".png")

    label_img = imageio.imread(label_path)
    label = np.array(label_img).astype(np.uint16)

    label = cv2.resize(label, dsize=(size[1], size[0]), interpolation=cv2.INTER_NEAREST)

    if(seg_type == "gt"):
        tsv_map = get_preprocessing_map("/home/aicenteruav/Sequential-DDETR/data/scannet/scannet-labels.combined.tsv")
        return rawCategory_to_nyu40(label, tsv_map)

    else:
        return label


def load_pose(scene_name: Path, img_index: int):
    """
    This is the function to load pose of the camera

    pose_matrix and camera_translation might be confuse to you,
    feel free to keep trace the code to check out what's the difference.

    scene_name: the string of the scene to visualize
    img_index: image index to retrieve
    """
    pose_path = scene_name / "pose" / (str(img_index)+".txt")
    pose_matrix = np.loadtxt(pose_path, dtype=np.float32)

    # calculate the true camera translation
    inv_pose = np.linalg.pinv(pose_matrix)
    camera_translation = -np.linalg.pinv(inv_pose[:3, :3])@inv_pose[:3, 3]

    return pose_matrix, camera_translation


def projectToFrame(vertexs, intrinsic, pose_matrix, translation):
    """
    This function is to back project all the mesh vertex back to image plane,
    why to do so ? tell me when we meet up !

    vertexs: mesh vertex of numpy array with shape (N, 3)
    intrinsic: numpy array of (3, 3)
    pose_matrix: numpy array of (4, 4)
    translation: numpy array of (3)
    """

    camera_translation_stack = np.repeat(translation[:, None].T,
                                         vertexs.shape[0], axis=0)

    vertexs_translated = vertexs.T - camera_translation_stack.T
    vertexs_camera_coordinate = np.linalg.pinv(pose_matrix[:3, :3])@vertexs_translated
    vertexs_camera_projection = intrinsic[:3, :3] @ vertexs_camera_coordinate

    vertexs_pixels = vertexs_camera_projection[:3, :] / vertexs_camera_projection[2, :]
    vertexs_pixels = vertexs_pixels[:2]

    return vertexs_pixels, vertexs_camera_coordinate


def visualizeCameraPosition(intrinsic, pose_matrix, translation, depth_img_shape):
    """
    This is the function to visualize how the current camera is position

    intrinsic: numpy array of (3, 3)
    pose_matrix: numpy array of (4, 4)
    translation: numpy array of (3)
    depth_image_shape: tuple of shape
    """

    camera_corner = np.array([[0, 0, 1],
                              [0, depth_img_shape[0], 1],
                              [depth_img_shape[1], depth_img_shape[0], 1],
                              [depth_img_shape[1], 0, 1]])

    camera_corner_3d = np.linalg.pinv(intrinsic[:3, :3])@camera_corner.T
    # camera_corner_3d = rotation_matrix()@camera_corner_3d
    camera_corner_3d = translation.reshape(3, 1) + pose_matrix[:3, :3]@camera_corner_3d
    camera_corner_3d = camera_corner_3d.T

    # 5 = 4 corners + 1 center
    visualized_points = np.ones((5, 3))
    visualized_points[:-1, :] = camera_corner_3d
    visualized_points[-1, :] = translation

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(visualized_points),
        lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    )

    color = [1, 0, 0]
    colors = np.tile(color, (8, 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def label_mesh(mesh,
               depth_img, label_img, linesets,
               vertex_on_CC, vertex_CC,
               viz, ids):
    """
    This is the main back projection code, if you are not fimiliar with it, please check out
    Abhijit Kundu, Xiaoqi Yin, Alireza Fathi, David Ross, Brian Brewington, Thomas Funkhouser, Caroline Pantofaru,
    "Virtual Multi-view Fusion for 3D Semantic Segmentation", in arXiv, 2020
    for more detail.

    mesh: mesh is the original scene mesh data
    depth_img: depth image that we will use to calculate whether object is in the camera image
    label_img: the predicted or gt label image
    linesets: line that visualize camera position
    viz: open3d visualize object
    id: image id
    """

    # start iterate over all vertex
    for i, vertex_each in enumerate(vertex_on_CC.T):

        label_colors = create_color_palette()

        # TODO
        # we simply rounded the vertex x and y component to get the coordinate on image. There's one
        # thing you need to notice, the way we did here is just simply nearest-neighbor assignment, while
        # might cause error due to its simplify, it's the easiest way to implement, try to us other way
        # like linear-interpolation in the future
        # there's another thing i want you to notice, why we assign to vertex index 1 while we assign at image
        # index 0 ? figure out the question and figure out why.

        vertex_each_v, vertex_each_u = round(vertex_each[0]), round(vertex_each[1])

        if(vertex_each_u >= 0 and vertex_each_u < depth_img.shape[0] and vertex_each_v >= 0 and vertex_each_v < depth_img.shape[1]):

            # retrieve the depth we calculate from mesh vertex and the depth we directly measure from sensor
            # compare these two to check whether vertex is on the current image plane
            Depth_Point = vertex_CC[2][i]
            Depth_Img = depth_img[vertex_each_u, vertex_each_v]

            # TODO
            # 0.5 is the acceptable error between these two depth data, is 0.5 a good parameter ?
            # we might need to test out by validation, please help me change this parameter into
            # parameter that argparse can manipulate.
            # while Depth Point > 0 on the other hand is a suitable parameter with no doubt since we
            # only want the point that is front us
            if(abs(Depth_Point - Depth_Img) < 0.5 and Depth_Point > 0):
                if(label_img[vertex_each_u, vertex_each_v] != 0 and label_img[vertex_each_u, vertex_each_v] != 40):
                    mesh.vertex_colors[i] = np.array(
                        label_colors[label_img[vertex_each_u, vertex_each_v] // 1000])/255

    return mesh


# color palette for nyu40 labels
def create_color_palette():
    return [
        (0, 0, 0),
        (174, 199, 232),		# wall
        (152, 223, 138),		# floor
        (31, 119, 180), 		# cabinet
        (255, 187, 120),		# bed
        (188, 189, 34), 		# chair
        (140, 86, 75),  		# sofa
        (255, 152, 150),		# table
        (214, 39, 40),  		# door
        (197, 176, 213),		# window
        (148, 103, 189),		# bookshelf
        (196, 156, 148),		# picture
        (23, 190, 207), 		# counter
        (178, 76, 76),
        (247, 182, 210),		# desk
        (66, 188, 102),
        (219, 219, 141),		# curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14), 		# refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),		# shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  		# toilet
        (112, 128, 144),		# sink
        (96, 207, 209),
        (227, 119, 194),		# bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  		# otherfurn
        (100, 85, 144)
    ]


def get_preprocessing_map(label_map):
    mapping = dict()
    with open(label_map) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:

            if not row["nyu40id"].isnumeric():
                mapping[int(row["id"])] = 40
            else:
                mapping[int(row["id"])] = int(row["nyu40id"])

    return mapping


def rawCategory_to_nyu40(label, map: dict):
    """Remap a label image from the 'raw_category' class palette to the 'nyu40' class palette """

    nyu40_label = label
    keys = np.unique(label).astype(np.int)

    for key in keys:
        print(label)
        print(key)
        if key != 0:
            nyu40_label[label == key] = map[key]
    return nyu40_label
