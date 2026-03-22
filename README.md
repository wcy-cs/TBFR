# A Natural Language Guided Approach for Blind Face Restoration: Methodology and Dataset 

<img width="2063" height="912" alt="model20" src="https://github.com/user-attachments/assets/6221800d-84db-4cf5-894a-d54e97d8c419" />

[paper](https://ieeexplore.ieee.org/document/11397781)

Our implementation of the TBFR module is based on [ResShift](https://github.com/zsyOAOA/ResShift).  
You can download the descriptions corresponding to CelebaHQ from [Descriptions](https://huggingface.co/datasets/wangchy29/TBFR_descriptions/blob/main/selected_descriptions.zip).

## Training

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 --nnodes=1 main.py \
  --cfg_path configs/blind_face_restoration256.yaml \
  --save_dir [logging_folder]
````

## Inference

```bash
python inference_TBFR.py \
  -i [input_image_path_or_folder] \
  -o [output_folder] \
  --steps 15 \
  --scale 4 \
  --seed 12345 \
  --chop_size 512
```
##Citation
```bash
@ARTICLE{11397781,
  author={An, Wenjie and Wang, Chenyang and Jiang, Junjun and Jiang, Kui and Liu, Xianming and Nie, Liqiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Natural Language Guided Approach for Blind Face Restoration: Methodology and Dataset}, 
  year={2026},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2026.3665736}}
```
