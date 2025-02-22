"""This file defines classes for the graph neural network. """

import sys
from functools import partial

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def instance_normalization(features):
    with tf.variable_scope(None, default_name='IN'):
        mean, variance = tf.nn.moments(features, [0], name='IN_stats', keep_dims=True)
        features = tf.nn.batch_normalization(features, mean, variance, None, None, 1e-12, name='IN_apply')
    return(features)

# Calculate euclidean distance between two points
def euclideanDistance(x, y):
    dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1, keepdims = True))
    return dist

normalization_fn_dict = {
    'fused_BN_center': slim.batch_norm,
    'BN': partial(slim.batch_norm, fused=False, center=False),
    'BN_center': partial(slim.batch_norm, fused=False),
    'IN': instance_normalization,
    'NONE': None
}
activation_fn_dict = {
    'ReLU': tf.nn.relu,
    'ReLU6': tf.nn.relu6,
    'LeakyReLU': partial(tf.nn.leaky_relu, alpha=0.01),
    'ELU':tf.nn.elu,
    'NONE': None,
    'Sigmoid': tf.nn.sigmoid,
    'Tanh': tf.nn.tanh,
}

def multi_layer_fc_fn(sv, mask=None, Ks=(64, 32, 64), num_classes=4,
    is_logits=False, num_layer=4, normalization_type="fused_BN_center",
    activation_type='ReLU'):
    """A function to create multiple layers of neural network to compute
    features passing through each edge.

    Args:
        sv: a [N, M] or [T, DEGREE, M] tensor.
        N is the total number of edges, M is the length of features. T is
        the number of recieving vertices, DEGREE is the in-degree of each
        recieving vertices. When a [T, DEGREE, M] tensor is provided, the
        degree of each recieving vertex is assumed to be same.
        N is the total number of edges, M is the length of features. T is
        the number of recieving vertices, DEGREE is the in-degree of each
        recieving vertices. When a [T, DEGREE, M] tensor is provided, the
        degree of each recieving vertex is assumed to be same.
        mask: a optional [N, 1] or [T, DEGREE, 1] tensor. A value 1 is used
        to indicate a valid output feature, while a value 0 indicates
        an invalid output feature which is set to 0.
        num_layer: number of layers to add.

    returns: a [N, K] tensor or [T, DEGREE, K].
        K is the length of the new features on the edge.
    """
    assert len(sv.shape) == 2
    assert len(Ks) == num_layer-1
    if is_logits:
        features = sv
        for i in range(num_layer-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type],
                )
        features = slim.fully_connected(features, num_classes,
            activation_fn=None,
            normalizer_fn=None
            )
    else:
        features = sv
        for i in range(num_layer-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type],
                )
        features = slim.fully_connected(features, num_classes,
            activation_fn=activation_fn_dict[activation_type],
            normalizer_fn=normalization_fn_dict[normalization_type],
            )
    if mask is not None:
        features = features * mask
    return features

def multi_layer_neural_network_fn(features, Ks=(64, 32, 64), is_logits=False,
    normalization_type="fused_BN_center", activation_type='ReLU'):
    """A function to create multiple layers of neural network.
    """
    # print("features.shape:", features.shape)
    # print("features:", features)
    # print("Ks:", Ks)
    # print("is_logits:", is_logits)
    # print("normalization_type:", normalization_type)
    # print("activation_type:", activation_type)
    # print()

    assert len(features.shape) == 2
    if is_logits:
        for i in range(len(Ks)-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type])
        features = slim.fully_connected(features, Ks[-1],
            activation_fn=None,
            normalizer_fn=None)
    else:
        for i in range(len(Ks)):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type])
    return features

def graph_scatter_max_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_max(point_features, point_centers, num_centers, name='scatter_max')
    return aggregated

def graph_scatter_sum_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_sum(point_features, point_centers, num_centers, name='scatter_sum')
    return aggregated

def graph_scatter_mean_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_mean(point_features,
        point_centers, num_centers, name='scatter_mean')
    return aggregated

class ClassAwarePredictor(object):
    """A class to predict 3D bounding boxes and class labels."""

    def __init__(self, cls_fn, loc_fn):
        """
        Args:
            cls_fn: a function to classify labels.
            loc_fn: a function to predict 3D bounding boxes.
        """
        self._cls_fn = cls_fn
        self._loc_fn = loc_fn

    def apply_regular(self, features, num_classes, box_encoding_len,
        normalization_type='fused_BN_center',
        activation_type='ReLU'):
        """
        Args:
            input_v: input feature vectors. [N, M].
            output_v: not used.
            A: not used.
            num_classes: the number of classes to predict.

        returns: logits, box_encodings.
        """
        box_encodings_list = []
        with tf.variable_scope('predictor'):
            with tf.variable_scope('cls'):
                logits = self._cls_fn(
                    features, num_classes=num_classes, is_logits=True,
                    normalization_type=normalization_type,
                    activation_type=activation_type)
            with tf.variable_scope('loc'):
                for class_idx in range(num_classes):
                    with tf.variable_scope('cls_%d' % class_idx):
                        box_encodings = self._loc_fn(
                            features, num_classes=box_encoding_len,
                            is_logits=True,
                            normalization_type=normalization_type,
                            activation_type=activation_type)
                        box_encodings = tf.expand_dims(box_encodings, axis=1)
                        box_encodings_list.append(box_encodings)
            box_encodings = tf.concat(box_encodings_list, axis=1)
        return logits, box_encodings

class ClassAwareSeparatedPredictor(object):
    """A class to predict 3D bounding boxes and class labels."""

    def __init__(self, cls_fn, loc_fn):
        """
        Args:
            cls_fn: a function to classify labels.
            loc_fn: a function to predict 3D bounding boxes.
        """
        self._cls_fn = cls_fn
        self._loc_fn = loc_fn

    def apply_regular(self, features, num_classes, box_encoding_len,
        normalization_type='fused_BN_center',
        activation_type='ReLU'):
        """
        Args:
            input_v: input feature vectors. [N, M].
            output_v: not used.
            A: not used.
            num_classes: the number of classes to predict.

        returns: logits, box_encodings.
        """
        box_encodings_list = []
        with tf.variable_scope('predictor'):
            with tf.variable_scope('cls'):
                logits = self._cls_fn(
                    features, num_classes=num_classes, is_logits=True,
                    normalization_type=normalization_type,
                    activation_type=activation_type)
            features_splits = tf.split(features, num_classes, axis=-1)
            with tf.variable_scope('loc'):
                for class_idx in range(num_classes):
                    with tf.variable_scope('cls_%d' % class_idx):
                        box_encodings = self._loc_fn(
                            features_splits[class_idx],
                            num_classes=box_encoding_len,
                            is_logits=True,
                            normalization_type=normalization_type,
                            activation_type=activation_type)
                        box_encodings = tf.expand_dims(box_encodings, axis=1)
                        box_encodings_list.append(box_encodings)
            box_encodings = tf.concat(box_encodings_list, axis=1)
        return logits, box_encodings

class PointSetPooling(object):
    """A class to implement local graph netural network."""

    def __init__(self,
        point_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        output_fn=multi_layer_neural_network_fn):
        self._point_feature_fn = point_feature_fn
        self._aggregation_fn = aggregation_fn
        self._output_fn = output_fn

    def apply_regular(self,
        point_features,
        point_coordinates,
        keypoint_indices,
        set_indices,
        point_MLP_depth_list=None,
        point_MLP_normalization_type='fused_BN_center',
        point_MLP_activation_type = 'ReLU',
        output_MLP_depth_list=None,
        output_MLP_normalization_type='fused_BN_center',
        output_MLP_activation_type = 'ReLU'):
        """apply a features extraction from point sets.

        Args:
            point_features: a [N, M] tensor. N is the number of points.
            M is the length of the features.
            point_coordinates: a [N, D] tensor. N is the number of points.
            D is the dimension of the coordinates.
            keypoint_indices: a [K, 1] tensor. Indices of K keypoints.
            set_indices: a [S, 2] tensor. S pairs of (point_index, set_index).
            i.e. (i, j) indicates point[i] belongs to the point set created by
            grouping around keypoint[j].
            point_MLP_depth_list: a list of MLP units to extract point features.
            point_MLP_normalization_type: the normalization function of MLP.
            point_MLP_activation_type: the activation function of MLP.
            output_MLP_depth_list: a list of MLP units to embedd set features.
            output_MLP_normalization_type: the normalization function of MLP.
            output_MLP_activation_type: the activation function of MLP.

        returns: a [K, output_depth] tensor as the set feature.
        Output_depth depends on the feature extraction options that
        are selected.
        """
        # print("point_features: ", point_features)
        # print("point_coordinates: ", point_coordinates)
        # print("keypoint_indices: ", keypoint_indices)
        # print("set_indices: ", set_indices)
        # print("point_MLP_depth_list: ", point_MLP_depth_list)
        # print("point_MLP_normalization_type: ", point_MLP_normalization_type)
        # print("point_MLP_activation_type: ", point_MLP_activation_type)
        # print("output_MLP_depth_list: ", output_MLP_depth_list)
        # print("output_MLP_normalization_type: ", output_MLP_normalization_type)
        # print("output_MLP_activation_type: ", output_MLP_activation_type)
        # print()

        print("PointSetPooling Version: original")
        print("point_MLP_depth_list: ", point_MLP_depth_list)
        print("output_MLP_depth_list: ", output_MLP_depth_list)

        # point_features = tf.Print(point_features, [point_features], "\n1.point_features:", summarize=100)
        # point_coordinates = tf.Print(point_coordinates, [point_coordinates], "\n2.point_coordinates:", summarize=100)
        # keypoint_indices = tf.Print(keypoint_indices, [keypoint_indices], "\n3.keypoint_indices:", summarize=100)
        # set_indices = tf.Print(set_indices, [set_indices], "\n4.set_indices:", summarize=100)

        ### Gather the points in a set ###
        # point_features = tf.Print(point_features, [point_features], "point_features:", summarize=40)
        point_set_features = tf.gather(point_features, set_indices[:,0])
        # point_set_features = tf.Print(point_set_features, [point_set_features], "\n5.point_set_features:", summarize=100)
        
        point_set_coordinates = tf.gather(point_coordinates, set_indices[:,0])
        # point_set_coordinates = tf.Print(point_set_coordinates, [point_set_coordinates], "\n6.point_set_coordinates:", summarize=100)
        
        # print("point_set_features: ", point_set_features)
        # print("point_set_coordinates: ", point_set_coordinates)
        # print()

        ### Gather the keypoints for each set ###
        point_set_keypoint_indices = tf.gather(keypoint_indices, set_indices[:, 1])
        # point_set_keypoint_indices = tf.Print(point_set_keypoint_indices, [point_set_keypoint_indices], "\n7.point_set_keypoint_indices:", summarize=100)
        
        point_set_keypoint_coordinates = tf.gather(point_coordinates, point_set_keypoint_indices[:,0])
        # point_set_keypoint_coordinates = tf.Print(point_set_keypoint_coordinates, [point_set_keypoint_coordinates], "\n8.point_set_keypoint_coordinates:", summarize=100)
        
        # print("point_set_keypoint_indices: ", point_set_keypoint_indices)
        # print("point_set_keypoint_coordinates: ", point_set_keypoint_coordinates)
        # print()

        ### points within a set use relative coordinates to its keypoint ###
        point_set_coordinates = point_set_coordinates - point_set_keypoint_coordinates
        # point_set_coordinates = tf.Print(point_set_coordinates, [point_set_coordinates], "\n9.point_set_coordinates:", summarize=100)

        ### edited ###
        distance_calculation = False
        print("distance_calculation: ", distance_calculation)
        if distance_calculation:
            point_set_distance = tf.sqrt(tf.reduce_sum(tf.square(point_set_coordinates), 1, keepdims = True))
            # point_set_distance = tf.Print(point_set_distance, [point_set_distance], "\n9,5.point_set_distance:", summarize=100)
            point_set_features = tf.concat([point_set_features, point_set_coordinates, point_set_distance], axis=-1)
        else:
            point_set_features = tf.concat([point_set_features, point_set_coordinates], axis=-1)
        
        # point_set_features = tf.Print(point_set_features, [point_set_features], "\n10.point_set_features:", summarize=100)
        
        # print("point_set_features: ", point_set_features)
        # print()

        # point_set_features = tf.Print(point_set_features, [point_set_features], "\npoint_set_features:", summarize=40)
        
        with tf.variable_scope('extract_vertex_features'):
            # Step 1: Extract all vertex_features
            extracted_point_features = self._point_feature_fn(
                point_set_features,
                Ks=point_MLP_depth_list, 
                is_logits=False,
                normalization_type=point_MLP_normalization_type,
                activation_type=point_MLP_activation_type)
            
            # extracted_point_features = tf.Print(extracted_point_features, [extracted_point_features], "\nextracted_point_features:", summarize=100)
            
            set_features = self._aggregation_fn(
                extracted_point_features, 
                set_indices[:, 1],
                tf.shape(keypoint_indices)[0])
            
            # set_features = tf.Print(set_features, [set_features], "\nset_features:", summarize=100)
        
        with tf.variable_scope('combined_features'):
            set_features = self._output_fn(set_features,
                Ks=output_MLP_depth_list, 
                is_logits=False,
                normalization_type=output_MLP_normalization_type,
                activation_type=output_MLP_activation_type)
            
            # set_features = tf.Print(set_features, [set_features], "\nset_features:", summarize=100)
        
        return set_features

### Original ###
class GraphNetAutoCenter_ori(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    
    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: original")
        print("auto_offset: ", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])
        # print("s_vertex_features: ", s_vertex_features)
        # print("s_vertex_coordinates: ", s_vertex_coordinates)
        # print()

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
            # print("offset: ", offset)
            # print("input_vertex_coordinates: ", input_vertex_coordinates)
            # print()
                
        ### Gather the destination vertex of the edges ###
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        # print("d_vertex_coordinates: ", d_vertex_coordinates)
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], axis=-1)
        # print("edge_features: ", edge_features)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod version 1 ###
""" Simplified the edge feature function """
class GraphNetAutoCenter_ModV1(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v1")
        print("auto_offset: ", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])
        # print("s_vertex_features: ", s_vertex_features)
        # print("s_vertex_coordinates: ", s_vertex_coordinates)
        # print()

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
            # print("offset: ", offset)
            # print("input_vertex_coordinates: ", input_vertex_coordinates)
            # print()
        
        ### edited ###
        simple_v1 = False
        print("simple_v1: ", simple_v1)
        if simple_v1:
            ### Prepare initial edge features ###
            edge_features = tf.concat([s_vertex_features, s_vertex_coordinates], axis=-1)
            # print("edge_features: ", edge_features)
        else:
            ### Gather the destination vertex of the edges ###
            d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
            # print("d_vertex_coordinates: ", d_vertex_coordinates)
            
            ### Prepare initial edge features ###
            edge_features = tf.concat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], axis=-1)
            # print("edge_features: ", edge_features)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 2 ###
""" Add vertex feature (surce and destination) in the edge function """
class GraphNetAutoCenter_ModV2(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v2")
        print("auto_offset: ", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        ### Gather the destination vertex of the edges ###
        d_vertex_features = tf.gather(input_vertex_features, edges[:,1])
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([s_vertex_features, s_vertex_features - d_vertex_features, s_vertex_coordinates - d_vertex_coordinates], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 3 ###
""" Remove Residual Connection """
class GraphNetAutoCenter_ModV3(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    
    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v3")
        print("auto_offset: ", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
                
        ### Gather the destination vertex of the edges ###
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        ### Residual Connection removed ###
        output_vertex_features = update_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 4 ###
""" Add vertex feature (surce and destination) in the edge function (Edited from version 2) """
class GraphNetAutoCenter_ModV4(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v4")
        print("auto_offset: ", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        ### Gather the destination vertex of the edges ###
        d_vertex_features = tf.gather(input_vertex_features, edges[:,1])
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([d_vertex_features, s_vertex_features - d_vertex_features, s_vertex_coordinates - d_vertex_coordinates], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 5 ###
""" Add vertex feature (surce and destination) and distance in the edge function (Edited from version 4) """
class GraphNetAutoCenter_ModV5(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v5")
        print("auto_offset: ", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        ### Gather the destination vertex of the edges ###
        d_vertex_features = tf.gather(input_vertex_features, edges[:,1])
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])

        ### Calculate relative coordinates and relative distance ###
        relative_coordinates = s_vertex_coordinates - d_vertex_coordinates
        relative_distance = tf.sqrt(tf.reduce_sum(tf.square(relative_coordinates), 1, keepdims = True))
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([d_vertex_features, s_vertex_features - d_vertex_features, relative_coordinates, relative_distance], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 6 ###
""" Add distance in edge function """
class GraphNetAutoCenter_ModV6(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    
    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v6")
        print("auto_offset:", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        ### Gather the destination vertex of the edges ###
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])

        ### Calculate relative coordinates and relative distance ###
        relative_coordinates = s_vertex_coordinates - d_vertex_coordinates
        relative_distance = tf.sqrt(tf.reduce_sum(tf.square(relative_coordinates), 1, keepdims = True))
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([s_vertex_features, relative_coordinates, relative_distance], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 7 ###
""" Add vertex feature (surce and destination) in the edge function and enavle auto_offset"""
class GraphNetAutoCenter_ModV7(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v7")
        print("auto_offset:", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        ### Gather the destination vertex of the edges ###
        d_vertex_features = tf.gather(input_vertex_features, edges[:,1])
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([s_vertex_features, s_vertex_features - d_vertex_features, s_vertex_coordinates - d_vertex_coordinates], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 8 ###
""" Fix the auto_offset and add vertex feature (surce and destination) in the edge function (Edited from version 4) """
class GraphNetAutoCenter_ModV8(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v8")
        print("auto_offset: ", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        ### Gather the destination vertex of the edges ###
        d_vertex_features = tf.gather(input_vertex_features, edges[:,1])
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([d_vertex_features, s_vertex_features - d_vertex_features, s_vertex_coordinates - d_vertex_coordinates], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 9 ###
""" Add vertex feature (surce and destination) and distance in the edge function """
class GraphNetAutoCenter_ModV9(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v9")
        print("auto_offset: ", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        ### Gather the destination vertex of the edges ###
        d_vertex_features = tf.gather(input_vertex_features, edges[:,1])
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])

        ### Calculate relative coordinates and relative distance ###
        relative_coordinates = s_vertex_coordinates - d_vertex_coordinates
        relative_distance = tf.sqrt(tf.reduce_sum(tf.square(relative_coordinates), 1, keepdims = True))
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([s_vertex_features, s_vertex_features - d_vertex_features, relative_coordinates, relative_distance], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

### Mod Version 10 ###
""" Add vertex feature (surce and destination) in the edge function and enable auto_offset"""
class GraphNetAutoCenter(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """apply one layer graph network on a graph. .

        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
            edge_MLP_depth_list: a list of MLP units to extract edge features.
            edge_MLP_normalization_type: the normalization function of MLP.
            edge_MLP_activation_type: the activation function of MLP.
            update_MLP_depth_list: a list of MLP units to extract update
            features.
            update_MLP_normalization_type: the normalization function of MLP.
            update_MLP_activation_type: the activation function of MLP.
            auto_offset: boolean, use auto registration or not.
            auto_offset_MLP_depth_list: a list of MLP units to compute offset.
            auto_offset_MLP_normalization_type: the normalization function.
            auto_offset_MLP_feature_activation_type: the activation function.

        returns: a [N, M] tensor. Updated vertex features.
        """
        # print("input_vertex_features: ", input_vertex_features)
        # print("input_vertex_coordinates: ", input_vertex_coordinates)
        # print("edges: ", edges)
        # print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        # print("edge_MLP_normalization_type: ", edge_MLP_normalization_type)
        # print("edge_MLP_activation_type: ", edge_MLP_activation_type)
        # print("update_MLP_depth_list: ", update_MLP_depth_list)
        # print("update_MLP_normalization_type: ", update_MLP_normalization_type)
        # print("update_MLP_activation_type: ", update_MLP_activation_type)
        # print("auto_offset: ", auto_offset)
        # print("auto_offset_MLP_depth_list: ", auto_offset_MLP_depth_list)
        # print("auto_offset_MLP_normalization_type: ", auto_offset_MLP_normalization_type)
        # print("auto_offset_MLP_feature_activation_type: ", auto_offset_MLP_feature_activation_type)
        # print()

        ### Identity ###
        print("GNN Version: mod_v10")
        print("auto_offset:", auto_offset)
        print("edge_MLP_depth_list: ", edge_MLP_depth_list)
        print("update_MLP_depth_list: ", update_MLP_depth_list)

        ### Gather the source vertex of the edges ###
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])

        ### [optional] Compute the coordinates offset ###
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        ### Gather the destination vertex of the edges ###
        d_vertex_features = tf.gather(input_vertex_features, edges[:,1])
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        
        ### Prepare initial edge features ###
        edge_features = tf.concat([s_vertex_features, d_vertex_features, s_vertex_coordinates - d_vertex_coordinates], axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            ### Extract edge features ###
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            ### Aggregate edge features ###
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        ### Update vertex features ###
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, 
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        output_vertex_features = update_features + input_vertex_features

        # output_vertex_features = tf.Print(output_vertex_features, [output_vertex_features], "\noutput_vertex_features:", summarize=100)
        
        return output_vertex_features

