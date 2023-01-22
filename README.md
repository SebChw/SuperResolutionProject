# SuperResolutionProject


To install the module in editable mode run
```
pip install -e .
```

To run training you must define your own config in yaml format. Look at configs folder. Then run

```
python torchsr/train.py -c=path_to_your_config
```

To generate patches out of DIV2K dataset

```
  python torchsr/data/create_patches.py
```
