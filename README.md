# CAF-Net

a crispr off-target prediction model, that leverage pre-train, synthetic over-sample, and fine-tune, to improve accuracy and recall on train set with imbalanced labels. 


## Installation
It's suggested to set up `tensorflow-gpu, python, cudnn, cudatoolkit` for first, due to certain feature of conda's dependency resolution. 
```
conda create --prefix=env python=3.x tensorflow-gpu=2.x cudnn cudatoolkit
conda activate ./env/
```

## Usage

*Note* Not implemented to run multiple instance simultaneously.

### 1.Validation on dataset with DNA/RNA bulges(indel)

```
cd src/
CUDA_VISIBLE_DEVICES=1 DEVICE=GPU:0 python validation1.py
```

### 2.Validation on dataset with mismatch only

```
cd src/
CUDA_VISIBLE_DEVICES=2 DEVICE=GPU:0 python validation2.py
```
## Dataset

the dataset in `data` directory are acquired from
[crispr-net](https://codeocean.com/capsule/9553651/tree/v3) under [CC 1.0 license](https://creativecommons.org/licenses/by/1.0/).


###   The details of the dataset
| Name | Location in data/ | Technique |with Indel| Lierature
| ----:| :---- |----: |----: |----: |
| Dataset I-1| I-1 |CIRCLE-Seq|Yes| Tasi et al., Nat Method, 2017|
| Dataset I-2| I-2 |GUIDE-Seq|Yes| Listgarten et al., Nat BME, 2018 |
| Dataset II-1| II-1 |protein knockout detection|No| Doench et al., Nat biotech, 2016 |
| Dataset II-3| II-3 |SITE-Seq|No|Cameron et al., Nature Methods, 2017 |
| Dataset II-4| II-4 |GUIDE-Seq|No| Tasi et al., Nat biotech, 2015|
| Dataset II-5| II-5 |GUIDE-Seq|No| Kleinstiver et al., Nature, 2015|
| Dataset II-6| II-6 |GUIDE-Seq|No| Listgarten et al., Nat BME, 2018 |
