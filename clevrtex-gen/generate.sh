export PYTHONPATH=~/anaconda3/envs/p37/bin/python
export PYTHONHOME=~/anaconda3/envs/p37
blender --background \
     --python-use-system-env --python generate.py -- \
     --render_tile_size 8 \
     --width 256 --height 256 \
		 --start_idx 0 \
		 --filename_prefix clevrtex \
		 --variant retest \
		 --output_dir ../output/ \
		 --properties_json data/myscene_army.json \
		 --shape_dir data/shapes \
		 --material_dir data/materials \
		 --blendfiles \
		 --num_images 1 \
		 --scene_blendfile 'data/scene_uorf.blend' \
		 --N_imgs 216 \
		 --n_my_objects 3