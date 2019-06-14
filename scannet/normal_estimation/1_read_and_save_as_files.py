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

	for i in range(len(scene_points_list)):
		point_cloud = scene_points_list[i]

		centered_point_cloud = np.copy(point_cloud)
		xyz_min = np.amin(centered_point_cloud, axis=0)[0:3]
		centered_point_cloud[:, 0:3] -= xyz_min

		xyz_max = np.amax(centered_point_cloud, axis=0)[0:3]
		centered_point_cloud[:, 0:3] -= (xyz_max / 2.0)

		path = '../scannet_%s/'%(split)
		out_filename = path+str(i)+'_xyz.pcd'
		fout = open(out_filename, 'w')
		fout.write('# .PCD v0.7 - Point Cloud Data file format\n')
		fout.write('VERSION 0.7\n')
		fout.write('FIELDS x y z\n')
		fout.write('SIZE 4 4 4\n')
		fout.write('TYPE F F F \n')
		fout.write('COUNT 1 1 1\n')
		fout.write('WIDTH '+str(centered_point_cloud.shape[0])+'\n')
		fout.write('HEIGHT 1\n')
		fout.write('VIEWPOINT 0 0 0 1 0 0 0\n')
		fout.write('POINTS '+str(centered_point_cloud.shape[0])+'\n')
		fout.write('DATA ascii\n')
		for i in range(centered_point_cloud.shape[0]):
			fout.write('%f %f %f\n' % \
						  (centered_point_cloud[i,0], centered_point_cloud[i,1], centered_point_cloud[i,2]))
		fout.close()

		print("Point cloud "+out_filename+" processed.")