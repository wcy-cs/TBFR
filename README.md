# A Natural Language Guided Approach for Blind Face Restoration: Methodology and Dataset

<img width="2063" height="912" alt="model20" src="https://github.com/user-attachments/assets/6221800d-84db-4cf5-894a-d54e97d8c419" />

Our implementation of the TBFR module is based on [ResShift](https://github.com/zsyOAOA/ResShift).  
You can download the descriptions corresponding to CelebaHQ from [Descriptions](https://huggingface.co/datasets/wangchy29/TBFR_descriptions/blob/main/selected_descriptions.zip).

训练代码：
> CUDA_VISIBLE_DEVICES=0,  torchrun --standalone --nproc_per_node=1  --nnodes=1 main.py --cfg_path configs/blind_face_restoration256.yaml --save_dir [Logging Folder]

推理代码：
> python inference_resshift.py -i [输入图像路径或文件夹] -o [输出文件夹] --steps 15 --scale 4 --seed 12345 --chop_size 512
