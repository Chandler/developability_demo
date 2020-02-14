# from scipy.linalg import eigh as eigendecomp
import util
import numpy as np
# import jax.numpy as np
# from jax import lax
from util import assert_shape

def hinge_energy(
    input_vector, 
    faces,
    lookup_vertex_face_neighbors,
    NUM_VERTS,
    gradient_mode):
    """
    faces[fi]        => [vi, vj, vk] // indices into verts
    face_normals[fi] => [nx, ny, nz] // 3 vector
    face_areas[fi]   => A            // scaler
    face_angles[fi]  => [ti, tj, tk] // scaler angles
    """
  
    util.assert_shape(input_vector, (NUM_VERTS*3,))

    # reshape input vector into a list of points
    verts = input_vector.reshape(NUM_VERTS, 3)

    face_normals, face_areas = util.get_face_normals(verts, faces)

    face_angles = util.get_face_angles(verts, faces)

    jacobian = np.zeros((NUM_VERTS, 3))

    def compute_energy(v_index):
        # the indices of the faces on this vertex star
        vertex_face_neighbors = lookup_vertex_face_neighbors[v_index]

        # for each face, the index of the center vertex of our star
        v_indices_in_faces = faces[vertex_face_neighbors] == v_index

        angles  = face_angles[vertex_face_neighbors][v_indices_in_faces]
        normals = face_normals[vertex_face_neighbors]
        areas   = face_areas[vertex_face_neighbors]

        scaled_normals = normals * np.sqrt(angles[:, np.newaxis])
        
        normal_covariance_matrix = scaled_normals.T.dot(scaled_normals)

        eigenvalues, eigenvectors = np.linalg.eigh(normal_covariance_matrix)
        
        smallest_eigenvalue = eigenvalues[0]

        # start computing the gradient if it's needed
        if gradient_mode == 1 or gradient_mode == 2:
            x = eigenvectors[:,0]

            # # TODO this is pretty hard to explain
            # # we're getting all the faces in JKI order
            # roll = 2-np.where(v_indices_in_faces)[1]
            # fk_index, fj_index, fi_index = util.roll_matrix(faces[vertex_face_neighbors], roll).T
            # all_fi = verts[fi_index] # 6 copies of the center vert
            # all_fj = verts[fj_index] # for each face, the j vert
            # all_fk = verts[fk_index] # for each face, the i vert
            
            # for every face touching our vertex (vertex star)
            for f_index in lookup_vertex_face_neighbors[v_index]:                            
                face = faces[f_index]
                
                # fi is v
                fi_index = v_index
                fj_index, fk_index = [j for j in face if fi_index != j]

                # the three vertices of the current face
                fi = verts[fi_index]
                fj = verts[fj_index]
                fk = verts[fk_index]
                
                # theta is the angle of the corner
                # of the face at fi
                theta = util.angle_between(fj-fi, fk-fi)

                # the face normal
                N = face_normals[f_index]

                # scalar, double the area
                A = face_areas[f_index] * 2.0

                # derivatives of the normal
                dNdi = np.outer(np.cross(fk-fj, N), N)/A
                dNdj = np.outer(np.cross(fi-fk, N), N)/A
                dNdk = np.outer(np.cross(fj-fi, N), N)/A
                assert_shape(dNdi, (3,3))

                # angle derivatives math from paper
                dThetadj = np.cross(N, (fi-fj)/np.linalg.norm(fi-fj))
                dThetadk = np.cross(N, (fk-fi)/np.linalg.norm(fk-fi))
                dThetadi = -1*(dThetadj + dThetadk)
                assert_shape(dThetadj, (3,))
                
                xdotN = x.dot(N)

                # a 3 vector pointing in the direction the i vertex should move
                # jacobian[fi_index] += xdotN*xdotN*dThetadi + 2.0*theta*xdotN * x.dot(dNdi)
                # jacobian[fj_index] += xdotN*xdotN*dThetadj + 2.0*theta*xdotN * x.dot(dNdj)
                # jacobian[fk_index] += xdotN*xdotN*dThetadk + 2.0*theta*xdotN * x.dot(dNdk)

                Ui = np.cross(fk - fj, N)/2.0
                Uj = np.cross(fi - fk, N)/2.0
                Uk = np.cross(fj - fi, N)/2.0

                jacobian[fi_index] += Ui
                jacobian[fj_index] += Uj
                jacobian[fk_index] += Uk

        return np.mean(areas)
        return smallest_eigenvalue
        
    eigenvalues = list(map(compute_energy, range(0, NUM_VERTS)))

    K = np.sum(eigenvalues)

    # return energy, gradient or both
    if gradient_mode == 0:
        return K
    elif gradient_mode == 1:
        return jacobian.reshape(NUM_VERTS*3)
    elif gradient_mode == 2:
        return (K, jacobian.reshape(NUM_VERTS*3))


        #================================================


            # # derivatives of the normal
            # dNdi = np.outer(np.cross(fk-fj, N), N)/A
            # dNdj = np.outer(np.cross(fi-fk, N), N)/A
            # dNdk = np.outer(np.cross(fj-fi, N), N)/A
            # assert_shape(dNdi, (3, 3))

            # # angle derivatives math from paper
            # dThetadj = np.cross(N, (fi-fj)/norm(fi-fj))
            # dThetadk = np.cross(N, (fk-fi)/norm(fk-fi))
            # dThetadi = -1*(dThetadj + dThetadk)
            # assert_shape(dThetadj, (3,))