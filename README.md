
# Feature-Geometric-Net-FG-Net

**Comparisons of Running Time of Our Method with SOTA methods RandLA and KPConv:**<br />
**Comparisons on Sequence 12:** <br />

![cardinal](./fig/Sequence_12.gif) <br />

**Comparisons on Sequence 13:** <br />

![cardinal](./fig/Sequence_13.gif) <br />

**Comparisons on Sequence 14:** <br />

![cardinal](./fig/Sequence_14.gif) <br />

**Comparisons on Sequence 15:** <br />

![cardinal](./fig/Sequence_15.gif) <br />

**Comparisons on Sequence 16:** <br />

![cardinal](./fig/Sequence_16.gif) <br />

**Semantic Semgmentation Results on Lille_1_1 of NPM3D Benchmark:**<br />

![cardinal](./fig/Lille_1_1.gif) 

**Semantic Semgmentation Results on Lille_1_2 of NPM3D Benchmark:**<br />

![cardinal](./fig/Lille_1_2.gif) 

**Semantic Semgmentation Results on Lille_2 of NPM3D Benchmark:**<br />

![cardinal](./fig/Lille_2.gif) 

**Semantic Semgmentation Results on Paris of NPM3D Benchmark:**<br />

![cardinal](./fig/Paris.gif) 

**Semantic Semgmentation Results on Area 1 of S3DIS Benchmark:**<br />

![cardinal](./fig/Area_1.gif) 

**Semantic Semgmentation Results on Area 2 of S3DIS Benchmark:**<br />

![cardinal](./fig/Area_2.gif)

**Semantic Semgmentation Results on Area 3 of S3DIS Benchmark:**<br />

![cardinal](./fig/Area_3.gif) 

**Semantic Semgmentation Results on Area 4 of S3DIS Benchmark:**<br />

![cardinal](./fig/Area_4.gif) 

**Semantic Semgmentation Results on Area 5 of S3DIS Benchmark:**<br />

![cardinal](./fig/Area_5.gif) 

**Semantic Semgmentation Results on Area 6 of S3DIS Benchmark:**<br />

![cardinal](./fig/Area_6.gif) 



**Semantic Semgmentation Results on Semantic3D Benchmark:**<br />

**Results on Birdfountain_station1_xyz_intensity_rgb**<br />

![cardinal](./fig/Birdfountain_station1_xyz_intensity_rgb.gif) 

**Results on Castleblatten_station_1_intensity_rgb**<br />

![cardinal](./fig/Castleblatten_station_1_intensity_rgb.gif) 

**Results on Marketplacefeldkirch_station1_intensity_rgb**<br />

![cardinal](./fig/Marketplacefeldkirch_station1_intensity_rgb.gif) 

**Results on Marketplacefeldkirch_station4_intensity_rgb**<br />

![cardinal](./fig/Marketplacefeldkirch_station4_intensity_rgb.gif) 

**Results on Marketplacefeldkirch_station7_intensity_rgb**<br />

![cardinal](./fig/Marketplacefeldkirch_station7_intensity_rgb.gif) 

**Results on Sg27_Station10_rgb_intensity**<br />

![cardinal](./fig/Sg27_Station10_rgb_intensity-reduced.gif) 

**Results on Sg28_Station2_rgb_intensity**<br />

![cardinal](./fig/Sg28_Station2_rgb_intensity-reduced.gif)

**Results on StGallenCathedral_station1_rgb_intensity**<br />

![cardinal](./fig/StGallenCathedral_station1_rgb_intensity.gif)

**Results on StGallenCathedral_station3_rgb_intensity**<br />

![cardinal](./fig/StGallenCathedral_station3_rgb_intensity.gif)

**Results on StGallenCathedral_station6_rgb_intensity**<br />

![cardinal](./fig/StGallenCathedral_station6_rgb_intensity.gif)


**Semantic Semgmentation Results on SemanticKITTI Benchmark:**<br />

**Results on Sequence 11-14 of SemanticKITTI Benchmark**<br />
![cardinal](./fig/Semantic-KITTI_11_14.gif)

**Results on Sequence 15-18 of SemanticKITTI Benchmark**<br />
![cardinal](./fig/Semantic-KITTI_15_18.gif)

**Results on Sequence 08 (Validation Set) of SemanticKITTI Benchmark**<br />
![cardinal](./fig/Semantic-KITTI_08.gif)

**Visualizations of Kernel Deformations on S3DIS 1**<br />
![cardinal](./fig/Kernel_deformation_1.gif)

**Visualizations of Kernel Deformations on S3DIS 2**<br />
![cardinal](./fig/Kernel_deformation_2.gif)

<!-- [[**Visualizations of Kernel Deformations on S3DIS 3**<br />
![cardinal](./fig/Kernel_deformation_3.gif)](url)](url) -->
**Comparisons of Our Proposed FG-Net on S3DIS with Current SOTA Methods**<br />

![cardinal](./fig/S3DIS_Compared_Final.png)
    
**Comparisons of Our Proposed FG-Net on S3DIS with Current SOTA Methods**<br />

![cardinal](./fig/S3DIS_Compared_Final_2.png)


**Comparisons of Our Proposed FG-Net on SemanticKITTI with Current SOTA Methods**<br />

![cardinal](./fig/SemanticKITTI_Compare_Results.png)

**Comparisons of Our Proposed FG-Net on Semantic3D with Current SOTA Methods**<br />

![cardinal](./fig/Semantic3D_Compare_2.png)

**Semantic Semgmentation Results on S3DIS Benchmark Whole Areas**<br />

![cardinal](./fig/s3dis_results_whole.png)

**Detailed Semantic Semgmentation Results on S3DIS Benchmark**<br />

![cardinal](./fig/s3dis_results_detailed.png)

**Semantic Semgmentation Results on NPM3D Benchmark**<br />

![cardinal](./fig/NPM3D_results.png)

**Detailed Semantic Semgmentation Results on NPM3D Benchmark**<br />
![cardinal](./fig/NPM3D_results_2.png)

**Detailed Semantic Semgmentation Results on S3DIS Benchmark**<br />

![cardinal](./fig/s3dis_results_detailed.png)

**Detailed Semantic Semgmentation Results on SemanticKITTI Benchmark**<br />

![cardinal](./fig/semantic_kitti_results.png)

**Detailed Semantic Semgmentation Results on Semantic3D Benchmark**<br />

![cardinal](./fig/semantic3d_final_result.png)

**Detailed Semantic Semgmentation Results on PartNet Benchmark**<br />

![cardinal](./fig/PartNet_results.png)

**Detailed Semantic Semgmentation Results on SemanticKITTI Benchmark**<br />

![cardinal](./fig/semantic_kitti_results.png)

# Summary of Work

This work presents FG-Net, a general deep learning framework for large-scale point clouds understanding without voxelizations, which achieves accurate and real-time performance with a single NVIDIA GTX 1080 GPU and an i7 CPU. First, a novel noise and outlier filtering method is designed to facilitate the subsequent high-level understanding tasks. For effective understanding purpose, we propose a novel plug-and-play module consisting of correlated feature mining and deformable convolution based geometric-aware modelling, in which the local feature relationships and point clouds geometric structures can be fully extracted and exploited. For the efficiency issue, we put forward a new composite inverse density sampling based and learning based operation and a feature pyramid based residual learning strategy to save the computational cost and memory consumption respectively. Compared with current methods which are only validated on limited datasets, we have done extensive experiments on eight real-world challenging benchmarks, which demonstrates that our approaches outperform state-of-the-art approaches in terms of both accuracy and efficiency. Moreover, weakly supervised transfer learning is also conducted to demonstrate the generalization capacity of our method. Source code, and representative results on the public benchmarks of our work are made publicly available to benefit the community.

### Acknowledgment
A portion of the code refers to <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> and the popular point based network Kernel Point Convolution <a href="https://github.com/HuguesTHOMAS/KPConv">KPConv</a>. Thank their contributions.


### License
The license is under the MIT license, see [LICENSE](./LICENSE).



### Citations

```
@article{liu2022fg,
  title={Fg-net: A fast and accurate framework for large-scale lidar point cloud understanding},
  author={Liu, Kangcheng and Gao, Zhi and Lin, Feng and Chen, Ben M},
  journal={IEEE Transactions on Cybernetics},
  volume={53},
  number={1},
  pages={553--564},
  year={2022},
  publisher={IEEE}
}

@inproceedings{liu2021fg,
  title={FG-Conv: Large-Scale LiDAR Point Clouds Understanding Leveraging Feature Correlation Mining and Geometric-Aware Modeling},
  author={Liu, Kangcheng and Gao, Zhi and Lin, Feng and Chen, Ben M},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={12896--12902},
  year={2021},
  organization={IEEE}
}

```