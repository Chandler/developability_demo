demo of covariance energy for minimizing guassian curvature in a triangular mesh

based on

http://www.cs.columbia.edu/cg/developability/developability-of-triangle-meshes.pdf
https://github.com/odedstein/DevelopabilityOfTriangleMeshes

# run using scipy numerical gradient
python cli.py \
    --in_obj=data/bunny.obj \
    --out_obj=data/optimized_bunny.obj \
    --energy=hinge \
    --action=optimize

# run using exact gradient
python cli.py \
	--exact_grad \
    --in_obj=data/bunny.obj \
    --out_obj=data/optimized_bunny.obj \
    --energy=hinge \
    --action=optimize

# debug exact gradien
python cli.py \
	--in_obj=data/bunny.obj \
	--energy=hinge \
	--action=check_gradient

