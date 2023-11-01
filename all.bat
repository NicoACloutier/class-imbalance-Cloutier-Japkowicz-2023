cd preprocessing
python preprocess.py
python finetune.py
cd ../modeling
python resample_vectorize.py
python model.py
cd ..
echo FINISHED