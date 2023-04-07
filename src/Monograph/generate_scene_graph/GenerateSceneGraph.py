import generate_scene_graph.config as config
import numpy as np
import open3d as o3d
from generate_scene_graph.Graph import Graph

# TODO: adapt graph output depending on the next step

class GenerateSceneGraph:
    def __init__(self, depth, semantic, viz=True):
        self._depth = depth
        self._semantic = semantic
        self._viz = viz
        self.height, self.width = depth.shape
        self.generate()

    def generate(self):
        self.find_focal_lengths()
        self.find_uniq_sem_ids()
        self.generate_point_cloud()
        self.find_medians()
        self.generate_graph()
        if self._viz:
            self.visualize()

    def find_focal_lengths(self):
        near = config.FOV_NEAR
        fov_x = config.FOV_X
        fov_y = 2.0 * np.arctan(
            self.height * np.tan(fov_x/2.0) / self.width
        )
        self.f_h = np.tan(fov_y/2.0)*near
        self.f_w = self.f_h*self.width/self.height
    
    def find_uniq_sem_ids(self):
        self.sem_uniq = list(set(np.matrix.flatten(self._semantic)))
        if -1 in self.sem_uniq:
            self.sem_uniq.remove(-1)
    
    def generate_point_cloud(self):
        """
        generate point cloud using intrinsic parameters of camera
        """
        x0 = self.width//2
        y0 = self.height//2

        self.ids_3d_points = {}
        for id in self.sem_uniq:
            self.ids_3d_points[id] = []

        self.pcd = []
        for i in range(self.height):
            for j in range(self.width):
                z = self._depth[i][j]
                x = (j - x0) * z / self.f_w
                y = (i - y0) * z / self.f_h
                self.pcd.append([x, y, z])

                sem_id = self._semantic[i][j]
                if sem_id != -1:
                    self.ids_3d_points[sem_id].append([x,y,z])
    
    def find_medians(self):
        # find medians of each object in image
        self.medians = {}
        for id in self.sem_uniq:
            xs = np.array(self.ids_3d_points[id])[:,0]
            ys = np.array(self.ids_3d_points[id])[:,1]
            zs = np.array(self.ids_3d_points[id])[:,2]
            self.medians[id] = [np.median(xs), np.median(ys), np.median(zs)]
    
    def generate_graph(self):
        # find distance between 2 medians and generate graph based on threshold
        self.graphs = []
        for fr_idx, fr_id in enumerate(self.sem_uniq):
            for to_idx, to_id in enumerate(self.sem_uniq):
                if to_idx>fr_idx:
                    sq_dist = np.sum((np.array(self.medians[fr_id]) - np.array(self.medians[to_id]))**2)
                    dist = np.sqrt(sq_dist)
                    if dist < config.DIST_THRESH:
                        graph = Graph(self.medians[fr_id], self.medians[to_id], fr_id, to_id)
                        self.graphs.append(graph)
    
    def visualize(self):
        o3d_obj = []
        # plot point cloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.pcd)
        o3d_obj.append(pcd_o3d)

        # plot medians
        medians_o3d = o3d.geometry.PointCloud()
        medians_o3d.points = o3d.utility.Vector3dVector(list(self.medians.values()))
        med_color = [0,0,1]
        median_colors = [med_color for i in range(len(self.medians.values()))]
        medians_o3d.colors = o3d.utility.Vector3dVector(median_colors)
        o3d_obj.append(medians_o3d)

        # draw lines 
        for graph in self.graphs:
            point1 = graph.from_coord
            point2 = graph.to_coord
            points = [point1, point2]
            lines = [[0,1]]
            colors = [[1,0,0]]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            o3d_obj.append(line_set)

        # visualize all
        o3d.visualization.draw_geometries(o3d_obj)
            
    def get_graph(self):
        return self.graphs
