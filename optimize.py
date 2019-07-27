# -*- coding: utf-8 -*-
from trimesh import TriMesh
from numpy.linalg import norm
from scipy import optimize
from scipy.linalg import eigh as eigendecomp
import copy
import numpy as np
from util import assert_shape
import util

counter = 0

def hinge_energy(input_vector, trimesh, gradient_mode, NUM_VERTS):
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

    # mutate the trimesh w/ the new verts
    # and update the normals and areas (
    # these are the non-topological properties
    # that change w/ vertices)
    trimesh.vs = verts
    trimesh.update_face_normals_and_areas()
    face_areas   = trimesh.get_face_areas()
    face_normals = trimesh.get_face_normals()

    energy = []

    jacobian = np.zeros((NUM_VERTS, 3))

    # for every vertex compute an energy
    for v_index in range(0, len(verts)):

        normal_covariance_matrix = np.zeros((3,3))

        # for every face touching our vertex (vertex star)
        for f_index in trimesh.vertex_face_neighbors(v_index):            
            face = trimesh.faces[f_index]

            # get the indices of the face in proper ijk
            # order where is is the center of the vertex star
            fi_index = v_index
            fj_index, fk_index = [j for j in face if fi_index != j]

            fi = verts[fi_index]
            fj = verts[fj_index]
            fk = verts[fk_index]
            
            # theta is the angle of the corner
            # of the face at a
            eij = np.subtract(fj, fi)
            eik = np.subtract(fk, fi)
            theta = util.angle_between(eij, eik)
   
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
        eigenvalues, eigenvectors = \
            eigendecomp(normal_covariance_matrix)

        smallest_eigenvalue = eigenvalues[0]
        associated_eigenvector = eigenvectors[:,0]

        # the first eigenvalue is the smallest one
        energy.append(smallest_eigenvalue)

        # start computing the gradient if it's needed
        if gradient_mode == 1 or gradient_mode == 2:

            x = associated_eigenvector

            # for every face touching our vertex (vertex star)
            for f_index in trimesh.vertex_face_neighbors(v_index):            
                face = trimesh.faces[f_index]
                
                # fi is v
                fi_index = v_index
                fj_index, fk_index = [j for j in face if fi_index != j]

                # the three vertices of the current face
                fi = verts[fi_index]
                fj = verts[fj_index]
                fk = verts[fk_index]
                
                # theta is the angle of the corner
                # of the face at fi
                eij = np.subtract(fj, fi)
                eik = np.subtract(fk, fi)
                theta = util.angle_between(eij, eik)

                # the face normal
                N = face_normals[f_index]

                # scalar, double the area
                A = face_areas[f_index] * 2

                # oriented edges
                ejk = np.subtract(fk, fj)
                eki = np.subtract(fi, fk)
                eij = np.subtract(fj, fi)
                assert_shape(ejk, (3,))

                # derivatives of the normal
                dNdi = np.outer(np.cross(ejk, N), N)/A
                dNdj = np.outer(np.cross(eki, N), N)/A
                dNdk = np.outer(np.cross(eij, N), N)/A
                assert_shape(dNdi, (3, 3))

                # derivatives of the angle
                dThetadj = -1 * np.cross(N, eij/norm(eij))
                dThetadk = -1 * np.cross(N, eki/norm(eki))
                dThetadi = np.cross(N, eij+eki)
                assert_shape(dThetadj, (3,))

                xdotN = x.dot(N)
                assert_shape(xdotN, ())

                # a 3 vector pointing in the directly the i vertex should move
                jacobian[fi_index] += xdotN*xdotN*dThetadi + 2*theta*xdotN * x.dot(dNdi)
                jacobian[fj_index] += xdotN*xdotN*dThetadj + 2*theta*xdotN * x.dot(dNdj)
                jacobian[fk_index] += xdotN*xdotN*dThetadk + 2*theta*xdotN * x.dot(dNdk)

    # squared sum of the energies is our final cost value
    K = np.sum(energy)**2

    # print cost at every invocation for debugging
    global counter
    if counter % 2 == 0:
        print "cost: {}, count: {}".format(K, counter)
    counter = counter + 1

    # return energy, gradient or both
    if gradient_mode == 0:
        return K
    elif gradient_mode == 1:
        return jacobian.reshape(NUM_VERTS*3)
    elif gradient_mode == 2:
        return (K, jacobian.reshape(NUM_VERTS*3))

if __name__ == '__main__':
    mesh_obj_path = "bunny.obj"

    trimesh = TriMesh.FromOBJ_FileName(mesh_obj_path)

    # a lot of meshes have duplicate vertices associated
    # with adjacent faces, clean that up.
    trimesh = util.deduplicate_trimesh_vertices(trimesh)

    NUM_VERTS = len(trimesh.vs)

    # curry different versions of the energy function and the starting mesh
    # these are now functions of a single variable that can be optimized
    hinge_energy_only     = lambda x: hinge_energy(x, trimesh, gradient_mode=0, NUM_VERTS=NUM_VERTS)
    hinge_grad_only       = lambda x: hinge_energy(x, trimesh, gradient_mode=1, NUM_VERTS=NUM_VERTS)
    hinge_energy_and_grad = lambda x: hinge_energy(x, trimesh, gradient_mode=2, NUM_VERTS=NUM_VERTS)

    # the first guess at the solution is the object itself
    first_guess = trimesh.vs.reshape(NUM_VERTS*3)

    # when this is false let scipy determine gradient for you
    # when it's true, use our numerical gradient approximation
    use_jac = False

    print "starting optimization"
    result = \
        optimize.minimize(
            fun=hinge_energy_and_grad if use_jac else hinge_energy_only,
            x0=first_guess,
            options={'disp': True},
            method="L-BFGS-B",
            constraints=None,
            jac=use_jac)

    final_mesh_verts = result.x.reshape(NUM_VERTS, 3)
    
    trimesh.vs = final_mesh_verts
    trimesh.write_OBJ("optimized_bunny.obj")
