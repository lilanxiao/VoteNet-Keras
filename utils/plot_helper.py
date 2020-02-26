import open3d as o3d
import numpy as np

def create_bbox(corners, color=[0,1,0]):
    assert corners.shape==(8,3)
    '''
    corners in camera coordinate
    x,y,z: right, downwards, forwards
    points: 
        0,1,2,3 on the top. 
        4,5,6,7 on the bottom.

        |z
        |
    3---|----0
    |   |____|______x
    |        |
    2--------1
    '''
    box = o3d.geometry.LineSet()
    box.points = o3d.utility.Vector3dVector(np.array(corners))
    box.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                                            [4, 5], [5, 6], [6, 7], [7, 4],
                                            [0, 4], [1, 5], [2, 6], [3, 7]]))
    box.colors = o3d.utility.Vector3dVector(np.array(
            [color for i in range(12)]))
    return box

def create_pointcloud(points, color=[0.5,0.5,0.5]):
    assert points.shape[1] ==3
    assert len(color) ==3
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(color,0),len(points),axis=0))
    return pcd