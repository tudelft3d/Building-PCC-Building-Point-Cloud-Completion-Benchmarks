import argparse
import os
import numpy as np
import sys
import open3d as o3d
from tqdm import tqdm
from datasets.io import IO
from scipy.spatial.distance import pdist, squareform
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_root', type=str, default='', help='ground truth root')
    args = parser.parse_args()

    assert args.gt_root is not None
    return args

def get_pc_scale(ptcloud):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(ptcloud)

    # Compute the oriented bounding box (OBB)
    obb = point_cloud.get_oriented_bounding_box()
    scale = np.linalg.norm(np.array(obb.get_max_bound()) - np.array(obb.get_min_bound()))
    return scale

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    assert m != 0
    return pc, centroid, m

def main():
    args = get_args()
    max_scale = 0
    max_pc_path = ''
    if args.gt_root != '':
        pc_file_list = os.listdir(args.gt_root)
        for pc_file in tqdm(pc_file_list, desc="Processing Files"):
            pc_file = os.path.join(args.gt_root, pc_file)
            pc_ndarray = IO.get(pc_file).astype(np.float64)
            #scale_pc = get_pc_scale(pc_ndarray)
            _, _, scale_pc = pc_norm(pc_ndarray)
            if scale_pc > max_scale:
                max_scale = scale_pc
                max_pc_path = pc_file

        print("max_scale = " + str(max_scale) + ", REFERENCE_POINTS_PATH: " + max_pc_path)

if __name__ == '__main__':
    main()