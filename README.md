# Vision Transformer Adapter for the semantic segmentation of structural defects bridge inspection
Repository for the Dacl10k challenge. Try the notebook for model inference. 

## Installation
```python
pip install requirements.txt
```
Model weights could be retrieved from https://drive.google.com/drive/folders/120MExaJwfFyd4-SA6ShC4ufsXeCVbHcl?usp=sharing. To be put in ./weights/.

## Inference
To display predictions:
```python
from model import make_prediction
images_path = "data/testdev/"
images_list = os.listdir(images_path)
i = random.randint(0, len(images_list)-1)
img_name = images_list[i]
img_path = os.path.join( images_path , img_name )
img = Image.open(img_path) #.resize((512,512))
make_prediction(segmenter, img)
```
<img src="https://github.com/mpaques269546/dacl10k_challenge/blob/main/pics/dacl_demo.jpg" width="500" height="500">


## Model performance
The proposed model obtained the following results on Dacl10k testdev dataset:
Class Name  | mIoU (%)  |
------------|-----------|
Crack	| 0.32
ACrack | 0.42
Efflorescence | 0.46
Rockpocket | 0.35
WConccor | 0.16
Hollowareas | 0.56
Cavity | 0.16
Spalling | 0.50
Restformwork | 0.37
Wetspot | 0.33
Rust | 0.49
Graffiti | 0.70
Weathering | 0.43
ExposedRebars | 0.34
Bearing | 0.62
EJoint | 0.66
Drainage | 0.70
PEquipment | 0.82
JTape | 0.45
ALL | 0.47



