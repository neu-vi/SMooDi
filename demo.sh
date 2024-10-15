module load cuda/11.8
module load gcc/11.1.0
module load ffmpeg/20190305
python demo_cmld.py --cfg ./configs/config_cmld_humanml3d.yaml --cfg_assets ./configs/assets.yaml --example ./demo/test.txt
