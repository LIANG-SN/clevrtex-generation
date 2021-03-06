export PYTHONPATH=~/anaconda3/envs/p37/bin/python
export PYTHONHOME=~/anaconda3/envs/p37
blender --background \
     --python-use-system-env --python generate.py -- \
     --render_tile_size 8 \
     --width 128 --height 128 \
		 --start_idx 0 \
		 --filename_prefix LOCAL \
		 --variant debug \
		 --output_dir ../output/ \
		 --properties_json data/myscene0.json \
		 --shape_dir data/shapes \
		 --material_dir data/materials \
		 --blendfiles \
		 --num_images 1 \
		#  --gpu \