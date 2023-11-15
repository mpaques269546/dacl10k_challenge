# Vision Transformer Adapter for the semantic segmentation of structural defects bridge inspection
Repository for the Dacl10k challenge. Try the notebook for model inference. 


Model weights could be retrieved from https://drive.google.com/drive/folders/120MExaJwfFyd4-SA6ShC4ufsXeCVbHcl?usp=sharing. To be put in ./weights/.

The proposed model obtained the following results on Dacl10k testdev dataset:
```python
Crack	0.32
ACrack	0.42
Efflorescence	0.46
Rockpocket	0.35
WConccor	0.16
Hollowareas	0.56
Cavity	0.16
Spalling	0.50
Restformwork	0.37
Wetspot	0.33
Rust	0.49
Graffiti	0.70
Weathering	0.43
ExposedRebars	0.34
Bearing	0.62
EJoint	0.66
Drainage	0.70
PEquipment	0.82
JTape	0.45
mIoU	0.47
```


