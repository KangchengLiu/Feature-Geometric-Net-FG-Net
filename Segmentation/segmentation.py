import open3d.ml.torch as ml3d
import shutil
import open3d as o3d
import argparse
import math
import numpy as np
import os
import random
import sys
import torch
from os.path import exists, join
import open3d.ml as _ml3d
from load_datas import load_model, get_custom_data, open3d_pcd
import matplotlib.pyplot as plt

s3dis_labels = {
    0: 'unlabeled',
    1: 'ceiling',
    2: 'floor',
    3: 'wall',
    4: 'beam',
    5: 'column',
    6: 'window',
    7: 'door',
    8: 'table',
    9: 'chair',
    10: 'sofa',
    11: 'bookcase',
    12: 'board',
    13: 'clutter'
}

cfg_file = "../configs/randlanet_s3dis.yml"
ckpt_path = load_model()

'''
save_planar_clouds extracts 4 labels of clouds (walls, floors, ceiling and others) 
from the given point cloud and stores them individually in the output directory.
'''
def save_planar_clouds(out_path, pc, pipeline_r):
    
    results_r = pipeline_r.run_inference(pc)
    print (set(results_r['predict_labels']))
    pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
    pred_label_r[0] = 0

    pts = pc['point']
    feature = pc['feat']
    label = pc['label']

    labels = results_r['predict_labels']
    labels = labels[:, None]
    pts_ceil, feat_ceil, pts_floor, feat_floor, pts_wall, feat_wall, pts_others, feat_others = ([] for i in range(8))
    for j, label in enumerate(results_r['predict_labels']):
        if label == 0:
            pts_ceil.append(pts[j])
            feat_ceil.append(feature[j])
        elif label == 1:
            pts_floor.append(pts[j])
            feat_floor.append(feature[j])    
        elif label == 2:
            pts_wall.append(pts[j])
            feat_wall.append(feature[j])
        else:
            pts_others.append(pts[j])
            feat_others.append(feature[j])

    if len(pts_floor) != 0:
        pcd_floor, pts_floor, feat_floor = open3d_pcd(pts_floor, feat_floor)
        o3d.io.write_point_cloud(os.path.join(out_path,'floor.ply'), pcd_floor)

    if len(pts_wall) != 0: 
        pcd_wall, pts_wall, feat_wall = open3d_pcd(pts_wall, feat_wall)
        o3d.io.write_point_cloud(os.path.join(out_path,'walls.ply'), pcd_wall)

    if len(pts_ceil) != 0:
        pcd_ceil, pts_ceil, feat_ceil = open3d_pcd(pts_ceil, feat_ceil)
        o3d.io.write_point_cloud(os.path.join(out_path,'ceiling.ply'), pcd_ceil)

    if len(pts_others) != 0:
        pcd_others, pts_others, feat_others = open3d_pcd(pts_others, feat_others)
        o3d.io.write_point_cloud(os.path.join(out_path,'others.ply'), pcd_others)


def save_floor_plan(pc_names, pcs, pipeline_r, vis_open3d):
    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]
        results_r = pipeline_r.run_inference(data)
        pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        pred_label_r[0] = 0

        pts = data['point']
        feature = data['feat']
        label = data['label']

        vis_d = {
            "name": name,
            "points": pts,
            "labels": label,
            "features": feature,
            'pred': pred_label_r,
        }
        vis_points.append(vis_d)

        labels = results_r['predict_labels']
        labels = labels[:, None]
        pts_ceil, feat_ceil, pts_floor, feat_floor, pts_wall, feat_wall, pts_column, feat_column, pts_window, feat_window, pts_door, feat_door = ([] for i in range(12))
        for j, label in enumerate(results_r['predict_labels']):
            if label == 0:
                pts_ceil.append(pts[j])
                feat_ceil.append(feature[j])
            if label == 1:
                pts_floor.append(pts[j])
                feat_floor.append(feature[j])    
            if label == 2:
                pts_wall.append(pts[j])
                feat_wall.append(feature[j])
            if label == 4:
                pts_column.append(pts[j])
                feat_column.append(feature[j])
            if label == 5:
                pts_window.append(pts[j])
                feat_window.append(feature[j])
            if label == 6:
                pts_door.append(pts[j])
                feat_door.append(feature[j])

        plt.axis('off')
        if len(pts_floor) !=0 :
            pcd_floor, pts_floor, feat_floor = open3d_pcd(pts_floor, feat_floor)
            plane_model, inliers = pcd_floor.segment_plane(distance_threshold=0.01,
                                            ransac_n=3,
                                            num_iterations=1000)
            [a, b, c, d] = plane_model
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            inlier_cloud = pcd_floor.select_by_index(inliers)
            outlier_cloud = pcd_floor.select_by_index(inliers, invert=True)

        if len(pts_wall) !=0 :
            pcd_wall, pts_wall, feat_wall = open3d_pcd(pts_wall, feat_wall)
            #pts_wall[:,2] = -plane_model[3] / plane_model[2]
            plt.scatter(pts_wall[:,0], pts_wall[:,1], color = 'black', s = 20)

        if len(pts_column) !=0 :
            pcd_column, pts_column, feat_column = open3d_pcd(pts_column, feat_column)
            plt.scatter(pts_column[:,0], pts_column[:,1], color = 'red', s = 20)

        if len(pts_window) !=0 :
            pcd_window, pts_window, feat_window = open3d_pcd(pts_window, feat_window)
            plt.scatter(pts_window[:,0], pts_window[:,1], color = 'blue', s = 20)

        if len(pts_door) != 0:
            pcd_door, pts_door, feat_door = open3d_pcd(pts_door, feat_door)
            plt.scatter(pts_door[:,0], pts_door[:,1], color = 'green', s = 20)


        if vis_open3d == True:
            o3d.visualization.draw_geometries([inlier_cloud] + [pcd_wall] + [pcd_column] + [pcd_window] + [pcd_door], zoom=0.8, front=[-0.4999, -0.1659, -0.8499], lookat=[2.1813, 2.0619, 2.0999], up=[0.1204, -0.9852, 0.1215])

        
        #inlier_points = np.asarray(inlier_cloud.points)

        plt.savefig(name)
        #im = Image.open('testplot.png').convert('RGB').save('testplot.jpg','JPEG')
        #rgb_im = im.convert('RGB')
        #rgb_im.save('testplot.jpg','JPEG')
    return vis_points



def main(data_path, out_path, visualize_prediction=False, vis_open3d=False, task='extraction'):

    cfg = _ml3d.utils.Config.load_from_file(cfg_file)

    cfg.model.ckpt_path = ckpt_path
    model = ml3d.models.RandLANet(**cfg.model)
    pipeline = ml3d.pipelines.SemanticSegmentation(model, **cfg.pipeline)
    pipeline.load_ckpt(model.cfg.ckpt_path)
    pcs = get_custom_data(data_path)
    if task == 'floor_plan':
        pcs_with_pred = save_floor_plan([os.path.join(out_path, 'floor_plan.png')], [pcs], pipeline, vis_open3d)
    elif task == 'extraction':
        save_planar_clouds(out_path, pcs, pipeline)
    else:
        print("No such task!")

    if visualize_prediction==True:
        v = ml3d.vis.Visualizer()
        lut = ml3d.vis.LabelLUT()
        for val in sorted(s3dis_labels.keys()):
            lut.add_label(s3dis_labels[val], val)
        v.set_lut("pred",lut)
        v.visualize(pcs_with_pred)
    for file in os.listdir("."):
        if file.endswith(".png"):
            shutil.copy(file, out_path)
            os.remove(file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" 3D semantic segmentation of point clouds and processing the segmentation result for floor plan")
    parser.add_argument("--data_path", help="The path to input PLY mesh file")
    parser.add_argument("--vis_prediction", action='store_true', help="Option to visualize semantic segmentation result. DEFAULT = FALSE")
    parser.add_argument("--vis_open3d", action='store_true',help="Option to visualize floor plan in 3D from open3d. DEFAULT = FALSE")
    parser.add_argument("--output_dir", help="The path to store floor plan images")
    parser.add_argument("--task", default='extraction', help='extraction (extract planar meshes), floor_plan')
    args = parser.parse_args()

    input_path = args.data_path
    output_path = args.output_dir
    visualize_prediction = args.vis_prediction
    vis_open3d =  args.vis_open3d
    task = args.task


    if not args.data_path or not args.output_dir:
        parser.print_help(sys.stderr)
        sys.exit(1)

    main(input_path, output_path, visualize_prediction, vis_open3d, task)