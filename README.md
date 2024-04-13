# TensoSDF
TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction

The method is based on NeRO, please refer to it to setup the environment.

### Datasets
Download the [TensoSDF synthetic datasets](https://drive.google.com/file/d/1JI2kMvi_79JIUBGbUBckxeCAgrWEW0kl/view?usp=drive_link) and the [ORB real datasets](https://stanfordorb.github.io/). For the ORB dataset, We use the *blender_LDR.tar.gz* for training and *ground_truth.tar.gz* for evaluation.

### Geometry reconstruction

Below take the "compressor" scene as an example:

```
# you need to modify the "dataset_dir" in configs/shape/syn/compressor first.

# reconstruct the geometry
python run_training.py --cfg configs/shape/syn/compressor.yaml

# evaluate the geometry reconstruction results via normal MAE metric
python eval_geo.py --cfg configs/shape/syn/compressor.yaml

# extract mesh from the model
python extract_mesh.py --cfg configs/shape/syn/compressor.yaml
```

Intermediate results will be saved at ```data/train_vis```. Models will be saved at ```data/model```. NVS results will be saved at ```data/nvs```. Extracted mesh will be saved at ```data/meshes```.

### Material reconstruction

```
# you need to modify the "dataset_dir" in configs/mat/syn/compressor first.

# estimate the material
python run_training.py --cfg configs/mat/syn/compressor.yaml

# evaluate the relighting results using the estimated materials via PSNR, SSIM and LPIPS metrics
python eval_mat.py --cfg configs/shape/syn/compressor.yaml
```
Intermediate results will be saved at ```data/train_vis```. Models will be saved at ```data/model```. Extracted materials will be saved at ```data/materials```. Relighting results will be saved at ```data/relight```.
