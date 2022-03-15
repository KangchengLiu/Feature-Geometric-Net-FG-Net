import numpy as np
from numpy.core.records import fromfile
import open3d as o3d
import argparse

def get_pcd_from_np_data(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(data[:,3:]/255.)
    return pcd

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    args = parser.parse_args()

    txt_data = np.fromfile(args.input_file, sep=' ').reshape(-1,6)
    pcd = get_pcd_from_np_data(txt_data)

    o3d.io.write_point_cloud(args.output_file, pcd)