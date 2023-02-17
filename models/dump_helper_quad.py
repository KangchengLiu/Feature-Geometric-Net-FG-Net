import numpy as np
import torch
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util

DUMP_CONF_THRESH = 0.5 # Dump boxes with obj prob larger than that.

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs



def dump_results_quad(end_points, dump_dir, config, inference_switch=False):
    ''' Dump results.

    Args:
        end_points: dict
            {..., pred_mask}
            pred_mask is a binary mask array of size (batch_size, num_proposal) computed by running NMS and empty box removal
    Returns:
        None
    '''
    if not os.path.exists(dump_dir):
        os.system('mkdir %s'%(dump_dir))

    # INPUT
    point_clouds = end_points['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]


    objectness_scores = end_points['last_quad_scores'].detach().cpu().numpy() # (B,K,2)
    pred_center = end_points['last_quad_center'].detach().cpu().numpy() # (B,K,3)
    normal_vector =  end_points['last_normal_vector'].detach().cpu().numpy()
    pred_size = end_points['last_quad_size'].detach().cpu().numpy()

    scan_names = end_points['scan_name']
    aggregated_vote_xyz = end_points['aggregated_vote_xyz'].detach().cpu().numpy()
 
    # OTHERS
    if 'pred_quad_mask' in end_points:
        pred_mask = end_points['pred_quad_mask']
    else:
        pred_mask = end_points['pred_mask'] # B,num_proposal
    idx_beg = 0

    for i in range(batch_size):
        pc = point_clouds[i,:,:]
        objectness_prob = softmax(objectness_scores[i,:,:])[:,1] # (K,)

        # Dump various point clouds
        #pc_util.write_ply(pc, os.path.join(dump_dir, '%s_pc.ply'%(scan_names[i])))
        #pc_util.write_ply(aggregated_vote_xyz[i,:,:], os.path.join(dump_dir, '%s_aggregated_fps_pc.ply'%(scan_names[i])))
        # Dump predicted bounding boxes
        if np.sum(objectness_prob>DUMP_CONF_THRESH)>0:
            num_proposal = pred_center.shape[1]
            obbs = []
            for j in range(num_proposal):
                cos_theta = torch.cosine_similarity(torch.tensor(normal_vector[i,j,:]),torch.tensor([0,1,0]),dim=0)
                heading_angle = torch.arccos(cos_theta) 
                cos_theta1 = torch.cosine_similarity(torch.tensor(normal_vector[i,j,:]),torch.tensor([1,0,0]),dim=0)
                if cos_theta1>0:
                  heading_angle = np.pi*2 - heading_angle               
                obb = np.zeros((7,))
                obb[0:3] = pred_center[i,j]
                obb[3] = pred_size[i,j,0]
                obb[4] = 0.1
                obb[5] = pred_size[i,j,1]
                obb[6] = heading_angle
                obbs.append(obb)
            if len(obbs)>0:
                obbs = np.vstack(tuple(obbs)) # (num_proposal, 7)
                #print(obbs)
                if len(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:])>0:
                    pc_util.write_oriented_bbox(obbs[np.logical_and(objectness_prob>DUMP_CONF_THRESH, pred_mask[i,:]==1),:], os.path.join(dump_dir, '%s_pred_confident_nms_quad.ply'%(scan_names[i])))
                    pc_util.write_oriented_bbox(obbs[objectness_prob>DUMP_CONF_THRESH, :], os.path.join(dump_dir, '%s_pred_confident_quad.ply'%(scan_names[i])))
                    pc_util.write_oriented_bbox(obbs[pred_mask[i,:]==1,:], os.path.join(dump_dir, '%s_pred_nms_quad.ply'%(scan_names[i])))

    # Return if it is at inference time. No dumping of groundtruths
    if inference_switch:
        return

    # LABELS
    center_label = end_points['gt_quad_centers'].detach().cpu().numpy() 
    size_label = end_points[f'gt_quad_sizes'].detach().cpu().numpy() 
    vector_label =  end_points[f'gt_normal_vectors'].detach().cpu().numpy() 
    num_gt_quads = end_points['num_gt_quads'].detach().cpu().numpy() 

    for i in range(batch_size):
        # Dump GT bounding boxes
        obbs = []
        for j in range(num_gt_quads[i,0]):
            cos_theta = torch.cosine_similarity(torch.tensor(vector_label[i,j,:]),torch.tensor([0,1,0]),dim=0)
            heading_angle = torch.arccos(cos_theta)  
            cos_theta1 = torch.cosine_similarity(torch.tensor(vector_label[i,j,:]),torch.tensor([1,0,0]),dim=0)
            if cos_theta1>0:
              heading_angle = np.pi*2 - heading_angle               
            obb = np.zeros((7,))
            obb[0:3] = center_label[i,j]
            obb[3] = size_label[i,j,0]
            obb[4] = 0.1
            obb[5] = size_label[i,j,1]
            obb[6] = heading_angle
            obbs.append(obb)
        if len(obbs)>0:
            obbs = np.vstack(tuple(obbs)) # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%s_gt_quad.ply'%(scan_names[i])))
    