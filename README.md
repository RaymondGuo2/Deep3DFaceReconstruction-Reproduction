# Reproduction of Deep 3D Face Reconstruction 

This is a reproduction of the original paper titled 'Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set'. The citation is shown below:

```
Y. Deng, J. Yang, S. Xu, D. Chen, Y. Jia, and X. Tong, Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set, IEEE Computer Vision and Pattern Recognition Workshop (CVPRW) on Analysis and Modeling of Faces and Gestures (AMFG), 2019.
```

The purpose of this reproduction is to facilitate a separate pipeline, while evaluating the success of 3D Morphable Model fitting mechanisms on face reconstruction. Given the pre-trained models, this repository therefore differs from the original in helping to facilitate a pipeline for generating custom images to produce outputs. The steps for the reproduction are detailed carefully below, and makes significant reference to the original repository found [here](https://github.com/sicxu/Deep3DFaceRecon_pytorch).

## Step 1: Install Dependencies

To install the dependencies, one can refer to the steps in the [official repository](https://github.com/sicxu/Deep3DFaceRecon_pytorch). To sum up, run the `environment.yml` file in a conda environment like so:

```
conda env create -f environment.yml
conda activate deep3d_pytorch
```

While the dependencies should run fairly straightforwardly, it is possible that they might fail. In that case, make sure to first activate the environment `deep3d_pytorch`, and manually install the rest of the dependencies. As detailed in the original repo, the next step is to download the `nvdiffrast` library. To do so, the easiest way is to clone that repository and pip installing from source. You will also require pip installing `dlib` and `MTCNN`. This is done by:

```
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast    # ./Deep3DFaceRecon_pytorch/nvdiffrast
pip install .
```

The final requirement is to install Arcface PyTorch:
```
cd ..    # ./Deep3DFaceRecon_pytorch
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/
```

The remaining work is to set up the CUDA environment, which can potentially be tricky owing to the specific packages used in the repo being deprecated. This implementation was tested on CUDA 10.0.130 with CUDNN 7.6.4.38. To initialise this, the `.bashrc` file will need to be updated and sourced to apply the changes, with the following lines added to the script.

```
export CUDA_HOME=/vol/cuda/10.0.130-cudnn7.6.4.38
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Extra Consideration**

In the event that after running the test file an error with regards to the compiler is mentioned, it may be the case given the dependencies and CUDA set-up, that a specific version of GCC and G++ are required in order to run the scripts. In that case, one first needs to conda install the correct dependency. It was found in the trial of this code, that a version of at most GCC and G++ 7 was necessary. Firstly, then, one can conda install that dependency by:

```
conda install gcc_linux-64=7.5.0 gxx_linux-64=7.5.0
```

If after installing the dependency and running the code, chances are that the model did not look in the right place for the compiler. As a consequence, you can type:
```
export CC=/vol/bitbucket/rqg23/anaconda/envs/deep3d_pytorch/bin/x86_64-conda-linux-gnu-gcc
export CXX=/vol/bitbucket/rqg23/anaconda/envs/deep3d_pytorch/bin/x86_64-conda-linux-gnu-g++
```

Another potential issue might occur in the detection of the library GL and EGL. This is quite tricky to establish, so an alternative solution is to state "--use_opengl False", as stated in an issue thread [here](https://github.com/sicxu/Deep3DFaceRecon_pytorch/issues/108).

## Step 2: Prepare Requisite Models

The steps for preparation can be found in the original repository (https://github.com/sicxu/Deep3DFaceRecon_pytorch) under the section titled "Inference with a Pre-trained Model". The first step is to download the model from the [Basel Face Model](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) project, which can be found [here](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads). Once you have agreed the conditions, download the file named "01_MorphableModel.mat". To obtain the expression basis by [Guo et al.](https://github.com/Juyong/3DFace) titled "Exp_Pca.bin", install from this [link](https://drive.google.com/file/d/1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6/view). These files must be placed inside the `BFM` subdirectory like so:

```
3DMM-Fitting-Reproduction
│
└─── BFM
    │
    └─── 01_MorphableModel.mat
    │
    └─── Exp_Pca.bin
    |
    └─── ...
```

Next, the file for the trained model titled 'epoch_20.pth' must be installed from this [link](https://drive.google.com/drive/folders/1liaIxn9smpudjjqMaWWRpP0mXRW_qRPP). Create a folde titled `checkpoints`, and within that folder create another subfolder titled a name of your choice. Place `epoch_20.pth` inside that folder, yielding a structure like so:

```
3DMM-Fitting-Reproduction
│
└─── checkpoints
    │
    └─── <model_name>
        │
        └─── epoch_20.pth
```

Finally, create a folder to store the custom images that you would like to test, which should look like the below. Note that detections refers to a folder with 5x2 landmark detections on the face. In the original repository, the authors did not detail a method to obtain these detections which is crucial for running the model. However, a solution was provided in a thread [here](https://github.com/sicxu/Deep3DFaceRecon_pytorch/issues/85#issuecomment-1137456892), and is thus used in the `mtcnn_detector.py`. This folder should be structured like so.

```
3DMM-Fitting-Reproduction
│
└─── <folder_to_test_images>
    │
    └─── *.jpg/*.png
    |
    └─── detections
        |
	    └─── *.txt
```

## Step 3: Preprocess Test Images

As stated above, the input images need to have associated landmark detections in order for the model to work. Furthermore, the provision of "in-the-wild" test images can often lead to issues in the reliability of the pipeline. This section is therefore dedicated as an extension of the repository to enhance the comprehensibility and reproducibility of the original research paper. In `process_images.py`, a folder of images are cleaned and standardised for facial detection, which is the role of `mtcnn_detector.py`. To simplify the process, a shell script is prepared that automates the pipeline for you, simply requiring one argument pointing to the folder which you want to clean and prepare landmarks for. Execute this below, if `test_images` is the folder with your images:

```
./process_images.sh <target_folder> # e.g. ./process_images.sh ./test_images
```

## Step 4: Run the Test Script

Having prepared the custom data (including its detected landmarks), you can run the function below on your test folder:

```
python test.py --name=<model_name> --epoch=20 --img_folder=<folder_to_test_images>
```

## Citation

Please cite the original paper if you have used this repository.

```
@inproceedings{deng2019accurate,
    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2019}
}
```