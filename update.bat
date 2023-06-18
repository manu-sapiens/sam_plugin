@echo off
if exist ".\venv" (

	call .\venv\Scripts\activate
	python -m ensurepip --upgrade
	python -m pip install --upgrade "pip>=22.3.1,<23.1.*"
	python -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
	python -m pip install git+https://github.com/facebookresearch/segment-anything.git
	python -m pip install -r requirements.txt

	call download_file.bat "https://dl.fbaipublicfiles.com/segment_anything" sam_vit_h_4b8939.pth checkpoints
	call download_file.bat "https://dl.fbaipublicfiles.com/segment_anything" sam_vit_l_0b3195.pth checkpoints
	call download_file.bat "https://dl.fbaipublicfiles.com/segment_anything" sam_vit_b_01ec64.pth checkpoints
	echo "DONE"
)
pause