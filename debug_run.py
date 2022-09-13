"""This file implement an inference pipeline for Point-GNN on KITTI dataset"""

import os
import time
import argparse
import multiprocessing
from functools import partial

import numpy as np
import tensorflow as tf
import open3d
import cv2
from tqdm import tqdm

from dataset.kitti_dataset import KittiDataset, Points
from models.graph_gen import get_graph_generate_fn
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, get_encoding_len
from models import preprocess
from models import nms
from util.config_util import load_config, load_train_config
from util.summary_util import write_summary_scale

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# occlusion score function =============================================================
def occlusion(label, xyz):
    if xyz.shape[0] == 0:
        return 0
    normals, lower, upper = dataset.box3d_to_normals(label)
    projected = np.matmul(xyz, np.transpose(normals))
    x_cover_rate = (np.max(projected[:, 0])-np.min(projected[:, 0]))\
        / (upper[0] - lower[0])
    y_cover_rate = (np.max(projected[:, 1])-np.min(projected[:, 1]))\
        / (upper[1] - lower[1])
    z_cover_rate = (np.max(projected[:, 2])-np.min(projected[:, 2]))\
        / (upper[2] - lower[2])
    return x_cover_rate*y_cover_rate*z_cover_rate


# ArgParse ===============================================================
parser = argparse.ArgumentParser(description='Point-GNN inference on KITTI')
parser.add_argument('checkpoint_path', type=str,
                    help='Path to checkpoint')
parser.add_argument('-l', '--level', type=int, default=0,
                    help='Visualization level, 0 to disable,' +
                    '1 to nonblocking visualization, 2 to block.' +
                    'Default=0')
parser.add_argument('--test', dest='test', action='store_true',
                    default=False, help='Enable test model')
parser.add_argument('--no-box-merge', dest='use_box_merge',
                    action='store_false', default='True',
                    help='Disable box merge.')
parser.add_argument('--no-box-score', dest='use_box_score',
                    action='store_false', default='True',
                    help='Disable box score.')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                    help='Path to KITTI dataset. Default="../dataset/kitti/"')
parser.add_argument('--dataset_split_file', type=str,
                    default='',
                    help='Path to KITTI dataset split file.'
                    'Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"')
parser.add_argument('--output_dir', type=str,
                    default='',
                    help='Path to save the detection results'
                    'Default="CHECKPOINT_PATH/eval/"')
args = parser.parse_args()
    
# Parse Arguments ===============================================================
VISUALIZATION_LEVEL = args.level
IS_TEST = args.test
USE_BOX_MERGE = args.use_box_merge
USE_BOX_SCORE = args.use_box_score
DATASET_DIR = args.dataset_root_dir

if args.dataset_split_file == '':
    DATASET_SPLIT_FILE = os.path.join(DATASET_DIR, './3DOP_splits/val.txt')
else:
    DATASET_SPLIT_FILE = args.dataset_split_file

if args.output_dir == '':
    OUTPUT_DIR = os.path.join(args.checkpoint_path, './eval/')
else:
    OUTPUT_DIR = args.output_dir

CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH = os.path.join(CHECKPOINT_PATH, 'config')

assert os.path.isfile(CONFIG_PATH), 'No config file found in %s'
config = load_config(CONFIG_PATH)

# setup dataset ===============================================================
if IS_TEST: #Testing data
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/testing/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/testing/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/testing/calib/'),
        '',
        num_classes=config['num_classes'],
        is_training=False)
else: #Training data
    dataset = KittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes'])
NUM_TEST_SAMPLE = dataset.num_files
NUM_CLASSES = dataset.num_classes

# setup placeholder =================================================================
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])

if config['input_features'] == 'irgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'rgb':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 3])
elif config['input_features'] == '0000':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'i000':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])
elif config['input_features'] == 'i':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])
elif config['input_features'] == '0':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 1])
if config['input_features'] == 'ixyz':
    t_initial_vertex_features = tf.placeholder(dtype=tf.float32, shape=[None, 4])

t_vertex_coord_list = [tf.placeholder(dtype=tf.float32, shape=[None, 3])]
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_vertex_coord_list.append(tf.placeholder(dtype=tf.float32, shape=[None, 3]))

t_edges_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_edges_list.append(tf.placeholder(dtype=tf.int32, shape=[None, 2]))

t_keypoint_indices_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_keypoint_indices_list.append(tf.placeholder(dtype=tf.int32, shape=[None, 1]))

t_is_training = tf.placeholder(dtype=tf.bool, shape=[])

# model ==================================================================
model = get_model(config['model_name'])(num_classes=NUM_CLASSES,box_encoding_len=BOX_ENCODING_LEN, mode='test', **config['model_kwargs'])

t_logits, t_pred_box = model.predict(
    t_initial_vertex_features, t_vertex_coord_list, t_keypoint_indices_list,
    t_edges_list,
    t_is_training)
t_probs = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)

# optimizers ==================================================================
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
fetches = {
    'step': global_step,
    'predictions': t_predictions,
    'probs': t_probs,
    'pred_box': t_pred_box
}

# setup Visualizer ============================================================
if VISUALIZATION_LEVEL == 1:
    print("Configure the viewpoint as you want and press [q]")
    calib = dataset.get_calib(0)
    cam_points_in_img_with_rgb = dataset.get_cam_points_in_image_with_rgb(0, calib=calib)
    vis = open3d.Visualizer()
    vis.create_window()
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(cam_points_in_img_with_rgb.xyz)
    pcd.colors = open3d.Vector3dVector(cam_points_in_img_with_rgb.attr[:, 1:4])
    line_set = open3d.LineSet()
    graph_line_set = open3d.LineSet()
    box_corners = np.array([[0, 0, 0]])
    box_edges = np.array([[0, 0]])
    line_set.points = open3d.Vector3dVector(box_corners)
    line_set.lines = open3d.Vector2iVector(box_edges)
    graph_line_set.points = open3d.Vector3dVector(box_corners)
    graph_line_set.lines = open3d.Vector2iVector(box_edges)
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(graph_line_set)
    ctr = vis.get_view_control()
    ctr.rotate(0.0, 3141.0, 0)
    vis.run()
color_map = np.array([(211, 211, 211), (255, 0, 0), (255, 20, 147), (65, 244, 101),
                      (169, 244, 65), (65, 79, 244), (65, 181, 244), (229, 244, 66)],
                     dtype=np.float32)
color_map = color_map/255.0
gt_color_map = {
    'Pedestrian': (0, 255, 255),
    'Person_sitting': (218, 112, 214),
    'Car': (154, 205, 50),
    'Truck': (255, 215, 0),
    'Van': (255, 20, 147),
    'Tram': (250, 128, 114),
    'Misc': (128, 0, 128),
    'Cyclist': (255, 165, 0),
}

# runing network ==============================================================
time_dict = {}
saver = tf.train.Saver()
graph = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Initialize session
    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))
    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print('Restore from checkpoint %s' % model_path)
    saver.restore(sess, model_path)
    previous_step = sess.run(global_step)

    ### Edited ###
    NUM_TEST_SAMPLE = 1

    # Loop every frame
    for frame_idx in tqdm(range(0, NUM_TEST_SAMPLE)):
        start_time = time.time()
        if VISUALIZATION_LEVEL == 2:
            pcd = open3d.PointCloud()
            line_set = open3d.LineSet()
            graph_line_set = open3d.LineSet()
        
        # provide input ======================================================
        cam_rgb_points = dataset.get_cam_points_in_image_with_rgb(frame_idx, config['downsample_by_voxel_size'])
        calib = dataset.get_calib(frame_idx)
        image = dataset.get_image(frame_idx)

        if not IS_TEST:
            box_label_list = dataset.get_label(frame_idx)
        
        input_time = time.time()
        time_dict['fetch input'] = time_dict.get('fetch input', 0) + input_time - start_time

        # Graph Generation ====================================================
        graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
        (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(cam_rgb_points.xyz, **config['runtime_graph_gen_kwargs'])
        
        graph_time = time.time()
        time_dict['gen graph'] = time_dict.get('gen graph', 0) + graph_time - input_time
        
        """
            irgb = intensity, red, green, blue
            0rgb = 0 (zero), red, green, blue
            0000 = 0 (zero), 0 (zero), 0 (zero), 0 (zero)
            i000 = intensity, 0 (zero), 0 (zero), 0 (zero)
            i = inrensity
            0 = 0
        """
        if config['input_features'] == 'irgb':
            input_v = cam_rgb_points.attr
        elif config['input_features'] == '0rgb':
            input_v = np.hstack([np.zeros((cam_rgb_points.attr.shape[0], 1)), cam_rgb_points.attr[:, 1:]])
        elif config['input_features'] == '0000':
            input_v = np.zeros_like(cam_rgb_points.attr)
        elif config['input_features'] == 'i000':
            input_v = np.hstack([cam_rgb_points.attr[:, [0]], np.zeros((cam_rgb_points.attr.shape[0], 3))])
        elif config['input_features'] == 'i':
            input_v = cam_rgb_points.attr[:, [0]]
        elif config['input_features'] == '0':
            input_v = np.zeros((cam_rgb_points.attr.shape[0], 1))
        elif config['input_features'] == 'ixyz':
            input_v = np.hstack([cam_rgb_points.attr[:, [0]], cam_rgb_points.xyz])

        last_layer_graph_level = config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
        
        if config['label_method'] == 'yaw':
            label_map = {'Background': 0, 'Car': 1, 'Pedestrian': 3,'Cyclist': 5, 'DontCare': 7}
        if config['label_method'] == 'Car':
            label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}
        if config['label_method'] == 'Pedestrian_and_Cyclist':
            label_map = {'Background': 0, 'Pedestrian': 1, 'Cyclist': 3,'DontCare': 5}
        
        # run forwarding =====================================================
        feed_dict = {
            t_initial_vertex_features: input_v,
            t_is_training: True,
        }
        feed_dict.update(dict(zip(t_edges_list, edges_list)))
        feed_dict.update(dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
        feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))
        results = sess.run(fetches, feed_dict=feed_dict)
        
        gnn_time = time.time()
        time_dict['gnn inference'] = time_dict.get('gnn inference', 0) + gnn_time - graph_time


# Debug ===============================================================
print("\n-------------------- Start of Debugging --------------------\n")
print("VISUALIZATION_LEVEL: ", VISUALIZATION_LEVEL)
print("IS_TEST: ", IS_TEST)
print("USE_BOX_MERGE: ", USE_BOX_MERGE)
print("USE_BOX_SCORE: ", USE_BOX_SCORE)
print("DATASET_DIR: ", DATASET_DIR)
print("DATASET_SPLIT_FILE: ", DATASET_SPLIT_FILE)
print("OUTPUT_DIR: ", OUTPUT_DIR)
print("CHECKPOINT_PATH: ", CHECKPOINT_PATH)
print("CONFIG_PATH: ", CONFIG_PATH, "\n")

print("NUM_TEST_SAMPLE: ", NUM_TEST_SAMPLE)
print("NUM_CLASSES: ", NUM_CLASSES, "\n")

print("config['box_encoding_method']: ", config['box_encoding_method'])
print("BOX_ENCODING_LEN: ", BOX_ENCODING_LEN)
print("box_encoding_fn: ", box_encoding_fn)
print("box_decoding_fn: ", box_decoding_fn)
print("config['input_features']: ", config['input_features'])
print("t_initial_vertex_features: ", t_initial_vertex_features)
print("len(config['runtime_graph_gen_kwargs']['level_configs']): ", len(config['runtime_graph_gen_kwargs']['level_configs']))
print("t_vertex_coord_list: ", t_vertex_coord_list)
print("t_edges_list: ", t_edges_list)
print("t_keypoint_indices_list: ", t_keypoint_indices_list)
print("t_is_training: ", t_is_training, "\n")

print("model: ", model)
print("t_logits: ", t_logits)
print("t_pred_box: ", t_pred_box)
print("t_probs: ", t_probs)
print("t_predictions: ", t_predictions)
print("global_step: ", global_step, "\n")

print("len(cam_rgb_points): ", len(cam_rgb_points), " x ", len(cam_rgb_points[0]))
print("cam_rgb_points:\n", cam_rgb_points)
print("cam_rgb_points.attr:\n", cam_rgb_points.attr)
print("Combined ixyz:\n", np.hstack([cam_rgb_points.attr[:, [0]], cam_rgb_points.xyz]))
"""
    cam_rgb_points:
        xyz = points that are visible in image
        attr = image color to the points as attributes
"""
print("len(calib): ", len(calib), " x ", len(calib["P0"]))
# print("calib: ", calib)
print("len(image): ", len(image), " x ", len(image[0]))
# print("image: ", image)
print("len(box_label_list): ", len(box_label_list), " x ", len(box_label_list[0]), "\n")
# print("box_label_list: ", box_label_list)

print("graph_generate_fn: ", graph_generate_fn)
# print("config['runtime_graph_gen_kwargs']: ", config['runtime_graph_gen_kwargs'])

print()
print("len(input_v): ", len(input_v), " x ", len(input_v[0]))
print("input_v: ", input_v)
# print("fetches: ", fetches)
print("len(feed_dict): ", len(feed_dict))
# print("feed_dict: ", feed_dict)
# print("results: ", results)

# print processing time
print()
for key in time_dict:
    print(key + " time : " + str(time_dict[key]/NUM_TEST_SAMPLE))

print("\n-------------------- End of Debugging --------------------\n")


## Model ##
"""
    "results = sess.run(fetches, feed_dict=feed_dict)"
    fetches =
        'step'       : global_step   -> global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        'predictions': t_predictions -> t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)
        'probs'      : t_probs       -> t_probs = model.postprocess(t_logits)
        'pred_box'   : t_pred_box    -> t_logits, t_pred_box = model.predict(t_initial_vertex_features, t_vertex_coord_list, t_keypoint_indices_list, t_edges_list, t_is_training)
    feed_dict = 
        t_initial_vertex_features   : input_v (18630 x 1)
        t_is_training               : True (bool)
        t_edges_list                : edges_list (?, 2)
        t_keypoint_indices_list     : keypoint_indices_list (?, 2)
        t_vertex_coord_list         : vertex_coord_list (?, 3)

    "model = get_model(config['model_name'])(num_classes=NUM_CLASSES,box_encoding_len=BOX_ENCODING_LEN, mode='test', **config['model_kwargs'])"
        num_classes                 = NUM_CLASSES
        box_encoding_len            = BOX_ENCODING_LEN
        mode                        = 'test'
        **config['model_kwargs']    =
            layer_configs
            regularizer_kwargs
            regularizer_type
    
    Point-GNN Model = multi_layer_fast_local_graph_model_v2 -> MultiLayerFastLocalGraphModelV2
    MultiLayerFastLocalGraphModelV2:
        - predict
        - postprocess
        - loss
"""