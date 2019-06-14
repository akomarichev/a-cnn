import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, acnn_module_rings

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    normals_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, normals_pl, cls_labels_pl

NUM_CATEGORIES = 16

def get_model(point_cloud, cls_label, normals, is_training, bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_normals = normals
    l0_points = normals

    # Set Abstraction layers
    l1_xyz, l1_points, l1_normals = acnn_module_rings(l0_xyz, l0_points, l0_normals, 512, [[0.0, 0.1], [0.1, 0.2]], [16,48], [[32,32,64], [64,64,128]], is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points, _ = acnn_module_rings(l1_xyz, l1_points, l1_normals, 128, [[0.1, 0.2], [0.3, 0.4]], [16,48], [[64,64,128], [128,128,256]], is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    cls_label_one_hot = tf.reshape(cls_label_one_hot, [batch_size, 1, NUM_CATEGORIES])
    cls_label_one_hot = tf.tile(cls_label_one_hot, [1,num_point,1])

    # Feature Propagation layers
    up_l3_points = pointnet_fp_module(l0_xyz, l3_xyz, None, l3_points, [64], is_training, bn_decay, scope='fa_layer1_up')
    up_l2_points = pointnet_fp_module(l0_xyz, l2_xyz, None, l2_points, [64], is_training, bn_decay, scope='fa_layer2_up')
    up_l1_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3_up')

    concat = tf.concat(axis=-1, values=[
                                     up_l3_points,
                                     up_l2_points,
                                     up_l1_points,
                                     cls_label_one_hot,
                                     l0_xyz
                                     ])

    net = tf_util.conv1d(concat, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp2')
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
    net = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc4')

    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
