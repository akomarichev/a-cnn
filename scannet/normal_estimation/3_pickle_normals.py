import pickle
import os
import sys
import numpy as np
from py3d import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from visualize_pc import visualize_pc, print_statistics

splits = ['train', 'test']
for split in splits:
	print("split: ", split)
	data_filename = '../scannet_%s.pickle'%(split)
	with open(data_filename,'rb') as fp:
		scene_points_list = pickle.load(fp, encoding='bytes')
		semantic_labels_list = pickle.load(fp, encoding='bytes')

	scene_points_with_normals_list = []
	for i in range(len(scene_points_list)):
		point_cloud = scene_points_list[i]

		path = '../scannet_%s/'%(split)
		normals_filename = path+str(i)+'_normals.txt'
		normals = np.loadtxt(normals_filename)

		scene_points_with_normals_list.append(np.concatenate([point_cloud, normals], 1))
		print("Point cloud "+normals_filename+" processed.")
	
	assert(len(scene_points_with_normals_list) == len(scene_points_list))
	pickle_filename = '../scannet_with_normals_%s.pickle'%(split)
	pickle_out = open(pickle_filename,'wb')
	pickle.dump(scene_points_with_normals_list, pickle_out)
	pickle.dump(semantic_labels_list, pickle_out)
	pickle_out.close()
