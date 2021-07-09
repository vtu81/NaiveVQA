# NaiveVQA

### File Directory

* `data/`
    * `annotations/` -- annotations data (ignored)
    * `images/` -- images data (ignored)
    * `questions/` -- questions data (ignored)
    * `results/` -- now contains only a fake results, for evaluation demo at `PythonEvaluationTools/vqaEvalDemo.py`
    * `clean.py` -- a script to clean up `train.json` in both `data/annotations/` and `data/questions/`
* `resnet/` -- resnet directory
* `config.py` -- global configure file
* `preprocess-image.py` -- preprocess the images, using ResNet152 to extract features for further usages
* `preprocess-vocab.py` -- preprocess the questions and annotations to get their vocabularies for further usages
* `data.py` -- dataset, dataloader and data processing code
* `utils.py` -- helper code
* `PythonHelperTools/` (currently not used)
    * `vqaDemo.py` -- a demo for VQA dataset APIs
    * `vqaTools/`
* `PythonEvaluationTools/` (currently not used)
    * `vqaEvalDemo.py` -- a demo for VQA evaluation
    * `vaqEvaluation/`
* `README.md`

### Prerequisite

* Disk avaiable storage at least 60GB
* A piece of Nivida GPU

### Quick Begin

Get the VQA dataset from [here](https://drive.google.com/open?id=1_VvBqqxPW_5HQxE6alZ7_-SGwbEt2_zn). Unzip the file and move the subdirectories

* `annotations/`
* `images/`
* `questions/`

into the repository directory `data/`.


Then, clean up your dataset (there are some images whose ids are referenced in the annotation & question files, while the images themselves don't exist!) with:

```bash
cd data
python clean.py # run the clean up script

mv annotations/train.json annotations/train_backup.json
mv annotations/train_cleaned.json annotations/train.json

mv questions/train.json questions/train_backup.json
mv questions/train_cleaned.json questions/train.json
```

Preprocess the images with:

```bash
python preprocess-images.py
```

* If you want to accelerate it, tune up `preprocess_batch_size` at `config.json`
* If you run out of CUDA memory, tune down `preprocess_batch_size` ata `config.json`

The output should be `./resnet-14x14.h5`.

Preprocess the questions and annotations to get their vocabularies with:

```bash
python preprocess-vocab.py
```

The output should be `./vocab.json`.