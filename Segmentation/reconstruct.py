import open3d as o3d
from argparse import ArgumentParser
import numpy as np
from glob import glob

def planar_decimation(mesh, target_face_count = 800):
    return o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, target_face_count)

def construct_planar_mesh(pcd, algo='poisson'):
    pcd.estimate_normals(fast_normal_computation=False)
    pcd.orient_normals_consistent_tangent_plane(100)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        if algo == 'poisson':
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, linear_fit=False)
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        elif algo == 'alpha_shapes':
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 1.0, tetra_mesh, pt_map)
        
        elif algo == 'ball_pivot':
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.25 * avg_dist   
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([0.005, 0.01, 0.02, 0.04]))
            # [0.01*radius, 0.02*radius, 0.04*radius]
    # o3d.visualization.draw_geometries([mesh])
    return mesh

def project_pcd_orthogonal(plane_model, pcd):
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # planar projection
    in_pts = np.asarray(pcd.points)
    flat_pts = []
    for k in range(in_pts.shape[0]):
        x_1, y_1, z_1 = in_pts[k]
        t = (a*x_1+b*y_1+c*z_1+d)/(a**2+b**2+c**2)
        flat_pts.append([x_1+a*t, y_1+b*t, z_1+c*t])
    flat_pcd = o3d.geometry.PointCloud()
    flat_pcd.points = o3d.utility.Vector3dVector(np.asarray(flat_pts).reshape(-1,3))
    flat_pcd.colors = pcd.colors
    return flat_pcd

def segment_walls(pcd, no_projection=False):
    count = 0
    # colored_cloud = []
    meshes = []
    while np.asarray(pcd.points).shape[0]>100:
        plane, ind = o3d.geometry.PointCloud.segment_plane(pcd, 0.025, np.asarray(pcd.points).shape[0]//10, 10000)
        wall = pcd.select_by_index(ind)
        pcd = pcd.select_by_index(ind, True)
        # o3d.visualization.draw_geometries([wall])
        # o3d.visualization.draw_geometries([pcd])
        # o3d.io.write_point_cloud("output/wall_"+str(count)+".ply", wall)
        if not no_projection:
            wall = project_pcd_orthogonal(plane, wall)
        
        meshes.append(wall)
        # wall.paint_uniform_color(np.random.rand(1,3).reshape(-1).tolist())
        # colored_cloud.append(wall)
        # count += 1
    # o3d.visualization.draw_geometries(colored_cloud)
    return meshes

def merge_meshes(meshes):
    merged_mesh = o3d.geometry.TriangleMesh()
    for mesh in meshes:
        merged_mesh += mesh
    # merged_mesh.orient_triangles()
    o3d.io.write_triangle_mesh('output/merged_walls.ply', merged_mesh)

def orient_normals_towards_origin(pcd):
    pcd.estimate_normals()

    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    pcd.orient_normals_consistent_tangent_plane(100)
    pcd.normals = o3d.utility.Vector3dVector(-1*np.asarray(pcd.normals))

    # normals = np.asarray(pcd.normals)
    # points = np.asarray(pcd.points)

    # print("Normals shape: ", normals.shape)
    # print("Points shape: ", points.shape)

    # rand_idx1, rand_idx2 = np.random.randint(low=points.shape[0], size=2)

    # if (np.dot(normals[rand_idx1], points[rand_idx2])<0):
    #     pcd.normals = o3d.utility.Vector3dVector(-1*normals)

    return pcd

def reconstruction_pipeline(args):
    from os.path import isfile
    # Initialize
    wall_pcd = o3d.geometry.PointCloud()
    floor_pcd = o3d.geometry.PointCloud()
    ceiling_pcd = o3d.geometry.PointCloud()
    others_pcd = o3d.geometry.PointCloud() 
    merged_mesh = o3d.geometry.TriangleMesh()

    # Load from files
    if isfile(args.walls_file):
        wall_pcd = o3d.io.read_point_cloud(args.walls_file)
    if isfile(args.floor_file):
        floor_pcd = o3d.io.read_point_cloud(args.floor_file)
    if isfile(args.ceiling_file):
        ceiling_pcd = o3d.io.read_point_cloud(args.ceiling_file)
    if isfile(args.others_file):
        others_pcd = o3d.io.read_point_cloud(args.others_file)
    
    wall_pcd = orient_normals_towards_origin(wall_pcd)
    floor_pcd = orient_normals_towards_origin(floor_pcd)
    ceiling_pcd = orient_normals_towards_origin(ceiling_pcd)
    others_pcd = orient_normals_towards_origin(others_pcd)

    if args.no_segmentation:
        print("Not performing segmentation")
        combined_pcd = wall_pcd + floor_pcd + ceiling_pcd
        combined_mesh = construct_planar_mesh(combined_pcd, args.meshing_algorithm)
        merged_mesh += planar_decimation(combined_mesh, 4000)
    else:
        # Process the walls
        wall_meshes_segmented = segment_walls(wall_pcd, args.no_projection)
        for wall in wall_meshes_segmented:
            wall_mesh = construct_planar_mesh(wall)
            #o3d.io.write_triangle_mesh("output/wall_mesh_"+str(count)+".ply", wall_mesh)
            merged_mesh += planar_decimation(wall_mesh)
        
        # Process the floor
        floor_mesh = construct_planar_mesh(floor_pcd, args.meshing_algorithm)
        merged_mesh += planar_decimation(floor_mesh)

        # Process the ceiling
        ceiling_mesh = construct_planar_mesh(ceiling_pcd, args.meshing_algorithm)
        merged_mesh += planar_decimation(ceiling_mesh)
    #o3d.io.write_triangle_mesh("output/wall_mesh_decimated_"+str(count)+".ply", wall_mesh_decimated)

    # Process the rest
    others_mesh = construct_planar_mesh(others_pcd, args.meshing_algorithm)
    merged_mesh += planar_decimation(others_mesh, 10000)

    o3d.io.write_triangle_mesh(args.output, merged_mesh)
    return

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-iw', '--walls_file', help = 'Input point cloud of just walls')
    parser.add_argument('-if','--floor_file', default='None', help='Input point cloud of floor')
    parser.add_argument('-ic','--ceiling_file', default='None', help='Input point cloud of ceiling')
    parser.add_argument('-io', '--others_file', default='None', help='Input point cloud of other points')
    parser.add_argument('-o', '--output', help='Output mesh file path')
    parser.add_argument('-m','--meshing_algorithm', default='poisson', help='poisson(default), ball_pivot, alpha_shapes')
    parser.add_argument('--no_projection', action='store_true', help='Do not project all points to plane')
    parser.add_argument('--no_segmentation', action='store_true', help='Do not process walls individually')
    args = parser.parse_args()

    reconstruction_pipeline(args)