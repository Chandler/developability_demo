import open3d
import time
import numpy as np
from numpy.linalg import norm

yellow = [1, 0.706, 0]
GLFW_KEY_TAB = 258
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT = 263
GLFW_KEY_P = 80

class CallbackContainer(object):
    def __init__(self, verts, faces):
        self.faces = faces
        self.verts = verts
        self.NUM_VERTS = len(verts)
        
        self.iterations = [verts.reshape(self.NUM_VERTS*3)]
        self.energies = []
        self.grads = []
        self.toggle = False

    def __call__(self, I):
        print("iteration: {}".format(len(self.iterations)))
        self.iterations.append(I)
        return False

    def save(self, file):
        np.save(file, np.array([self.verts, self.faces, self.energies, self.grads, self.iterations]))

    @classmethod
    def load(cls, file):
        verts, faces, energies, grads, iterations = np.load(file, allow_pickle=True)
        inst = cls(verts, faces)
        inst.iterations = iterations
        inst.energies = energies
        inst.grads = grads
        return inst

    def compute(self, energy_and_grad):
        self.energies = []
        self.grads = []

        print("begining recompute of grad/energy ")
        for i, I in enumerate(self.iterations):
            energy, grad    = energy_and_grad(I)
            print("{} - energy: {} grad norm: {}".format(i, energy, np.linalg.norm(grad)))

            self.energies.append(energy)
            self.grads.append(grad)

    def visualize_mesh(self, color=yellow):
        """"
            utility::LogInfo("  -- Render mode control --");
            utility::LogInfo("    tab/arrow    : next frame");
            utility::LogInfo("    L            : Turn on/off lighting.");
            utility::LogInfo("    +/-          : Increase/decrease point size.");
            utility::LogInfo("    Ctrl + +/-   : Increase/decrease width of geometry::LineSet.");
            utility::LogInfo("    N            : Turn on/off point cloud normal rendering.");
            utility::LogInfo("    S            : Toggle between mesh flat shading and smooth shading.");
            utility::LogInfo("    W            : Turn on/off mesh wireframe.");
            utility::LogInfo("    B            : Turn on/off back face rendering.");
            utility::LogInfo("    I            : Turn on/off image zoom in interpolation.");
            utility::LogInfo("    T            : Toggle among image render:");
            utility::LogInfo("                   no stretch / keep ratio / freely stretch.");
            utility::LogInfo("");
        """

        T           = open3d.geometry.TriangleMesh()
        T.triangles = open3d.utility.Vector3iVector(self.faces)
        T.vertices  = open3d.utility.Vector3dVector(self.verts)
        T.paint_uniform_color(color)
        T.compute_vertex_normals()

        P = open3d.geometry.PointCloud()
        P.points = T.vertices
        
        attached_normals = open3d.geometry.LineSet()
        origin_normals  = open3d.geometry.LineSet()

        def custom_draw_geometry_with_key_callback(pcd):
            self.current_frame_idx = 0
            
            def previous_frame(vis):
                if self.current_frame_idx == 0 :
                    previous_frame_idx = len(self.iterations) - 1
                else:
                    previous_frame_idx = self.current_frame_idx - 1

                show_frame(vis, previous_frame_idx)
                self.current_frame_idx = previous_frame_idx
                                
            def next_frame(vis):
                if self.current_frame_idx == len(self.iterations) - 1:
                    next_frame_idx = 0
                else:
                    next_frame_idx = self.current_frame_idx + 1

                show_frame(vis, next_frame_idx)
                self.current_frame_idx = next_frame_idx
                
            def show_frame(vis, frame_idx):
                if len(self.energies) > 0:
                    print("rendering iteration: {}  E: {} - G: {}".format(
                        frame_idx, self.energies[frame_idx], norm(self.grads[frame_idx])))
                else:
                    print("rendering iteration: {}".format(frame_idx))

                # grab an iteration vector and reshape it back
                # into a list of verts
                I = self.iterations[frame_idx]
                verts = I.reshape(self.NUM_VERTS, 3)
                
                # update the `TriangleMesh` verts and recompute
                # the normals            
                T.vertices = open3d.utility.Vector3dVector(verts)
                T.compute_vertex_normals()
                T.compute_triangle_normals()
                
                # update the point cloud
                P.points = T.vertices

                attached_normal_points = []
                origin_normal_points   = []
                lines                  = [] # indices of points =
                
                i = 0
                for n, point in zip(np.array(T.triangle_normals), np.array(T.vertices)):
                    attached_normal_points.extend(np.add([[0,0,0], n], [point]))
                    origin_normal_points.extend([[0,0,0], n*3])
                    lines.append([i, i+1])
                    i = i+2

                attached_normals.lines = open3d.utility.Vector2iVector(lines)
                attached_normals.points = open3d.utility.Vector3dVector(attached_normal_points)

                origin_normals.lines = open3d.utility.Vector2iVector(lines)
                origin_normals.points = open3d.utility.Vector3dVector(origin_normal_points)
                
                # update the animation page
                vis.update_geometry()
                vis.update_renderer()
                vis.poll_events()

            def switch(vis):
                if self.toggle:
                    vis.add_geometry(T)
                    vis.add_geometry(P)
                    vis.add_geometry(attached_normals)
                    vis.remove_geometry(origin_normals)
                else:
                    vis.remove_geometry(T)
                    vis.remove_geometry(P)
                    vis.remove_geometry(attached_normals)
                    vis.add_geometry(origin_normals)

                vis.update_renderer()
                vis.poll_events()

                self.toggle = not self.toggle
            
            key_to_callback = {}
            key_to_callback[GLFW_KEY_P] = switch
            key_to_callback[GLFW_KEY_TAB] = next_frame
            key_to_callback[GLFW_KEY_RIGHT] = next_frame
            key_to_callback[GLFW_KEY_LEFT] = previous_frame
            open3d.visualization.draw_geometries_with_key_callbacks(pcd, key_to_callback)

        custom_draw_geometry_with_key_callback([T])
