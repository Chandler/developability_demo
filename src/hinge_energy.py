from scipy.linalg import eigh as eigendecomp
import copy
import numpy as np
from util import assert_shape
import util

def hinge_energy(input_vector, faces, lookup_vertex_face_neighbors, gradient_mode, NUM_VERTS):
    """
    The covariance energy for minimizing Gaussian
    curvature and its gradient

    Its expected that you curry this method into a function
    of a single variable `input_vector` to make it optimizable
    see `hinge_energy_and_grad`
    
    input_vector: a 1D array representing a flattened list
    of positions

    trimesh: a TriMesh object (see local trimesh.py) containing
    all the immutable topology of the mesh. This object
    can be muted with updated vertices and used to recompute
    normals and areas.
    
    gradient_mode sets the return value and gates gradient computation:
      0: return energy
      1: return grad
      2: return (energy, grad)

    NUM_VERTS: how many vertices are in the mesh total
    """
    assert_shape(input_vector, (NUM_VERTS*3,))

    # reshape input vector into a list of points
    verts = input_vector.reshape(NUM_VERTS, 3)

    face_normals, face_areas = util.get_face_normals(verts, faces)
    face_angles = util.get_face_angles(verts, faces)
    energy = []

    jacobian = np.zeros((NUM_VERTS, 3))

    # for every vertex compute an energy
    for v_index in range(0, len(verts)):

        normal_covariance_matrix = np.zeros((3,3))

        # for every face touching our vertex (vertex star)
        for f_index in lookup_vertex_face_neighbors[v_index]:            
            face = faces[f_index]

            theta = face_angles[f_index][face == v_index]
   
            # the normal of the face
            N = face_normals[f_index]

            # NN^T gives us a 3x3 covariance matrix
            # of the vector. Scale it by Theta
            # and add it to the running covariance matrix
            # being built up for every normal
            normal_covariance_matrix += theta*np.outer(N, N)

        # now you have the covariance matrix for every face
        # normal surrounding this vertex star. 
        # the first eigenvalue of this matrix is our energy
        #
        #
        # ``` (eq 4)
        # Since Equation 3 is just the
        # variational form of an eigenvalue problem, λi can also be expressed
        # as the smallest eigenvalue of the 3 × 3 normal covariance matrix
        # ```
        eigenvalues, eigenvectors = eigendecomp(normal_covariance_matrix)

        smallest_eigenvalue = eigenvalues[0]

        associated_eigenvector = eigenvectors[:,0]
        
        # the first eigenvalue is the smallest one
        energy.append(smallest_eigenvalue)

        # start computing the gradient if it's needed
        if gradient_mode == 1 or gradient_mode == 2:

            x = associated_eigenvector

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
                assert_shape(dNdi, (3))

                # angle derivatives math from paper
                dThetadj = np.cross(N, (fi-fj)/np.linalg.norm(fi-fj))
                dThetadk = np.cross(N, (fk-fi)/np.linalg.norm(fk-fi))
                dThetadi = -1*(dThetadj + dThetadk)
                assert_shape(dThetadj, (3,))

                xdotN = x.dot(N)

                # a 3 vector pointing in the direction the i vertex should move
                jacobian[fi_index] += xdotN*xdotN*dThetadi + 2.0*theta*xdotN * x.dot(dNdi)
                jacobian[fj_index] += xdotN*xdotN*dThetadj + 2.0*theta*xdotN * x.dot(dNdj)
                jacobian[fk_index] += xdotN*xdotN*dThetadk + 2.0*theta*xdotN * x.dot(dNdk)

    # squared sum of the energies is our final cost value
    K = np.sum(energy)

    # return energy, gradient or both
    if gradient_mode == 0:
        return K
    elif gradient_mode == 1:
        return jacobian.reshape(NUM_VERTS*3)
    elif gradient_mode == 2:
        return (K, jacobian.reshape(NUM_VERTS*3))
