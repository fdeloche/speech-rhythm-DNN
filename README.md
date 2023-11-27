Code for *Acoustic characterization of speech rhythm:
going beyond metrics with recurrent neural
networks*, Deloche, Bonnasse-Gahot, Gervain 2024. Implements a RNN for a language-identification task based on speech envelopes (+F0, optional), used for identifying rhythmic features of languages.

Contains :

* Code for data processing
* Code for setting and training the DNN models
* Code for evaluating the models
* Specific code + results for the analyses in the paper
* Weights for main version of the model presented in paper (`./models/weights_paper/weights_2021-08-19_14-53-14.h5`), see also the `./selected_logdir` for additional data (can be read with tensorboard)
* Weights of the linear regressions for the analysis of correlation with rhythm metrics (`./compar/coeffs_paper_reg.json`).

The code contains some rigidities as it was created in the goal of the study and not as a ready-to-use tool. However, it should be adaptable to other data/purposes with little work. As authors of this study, we are always looking for opportunities to extend this work, so do not hesitate to contact us for assistance or sharing project ideas.

Pipeline
--------------


1. Put wav data in /Files/languages folders
    Files must be sampled at 16 kHz, default length 10*2**14 samples (approx. 10 s; for now, the number of samples per file must be the same for every file)
    Could be /TRAIN, /TEST, /VALID subfolders or folds (see `createfolders.py`)
2. Normalize data with 
    ```python normalize_data.py [language1] [language2] ...```
    if "all" in language, autodetect
3. Compute scores (features). Also create markers for processed data. Run `python clear_scores_markers.py [languages]` to delete these or run `compute_scores.py` with `--clear_markers` .

  ```bash
python compute_scores.py [language1] [language2] ... --batch_size [batch_size, default : 16] --length [length, default : 10*2^14]  --max_files [max files per language] ... 
  ```

if "all" in language, autodetect

Other example (for Ramus Data, sentences of 3 sec)

```shell
python compute_scores.py --batch_size 1 --length 49152 --RamusData all
```



4. Training: `Model Training.ipynb` , based on `model.py` (defines RNN, LSTM or GRU) and `input.py` (pipeline for tensorflow)

    * Note that features for the NN are different than features computed in the previous step (pre-processing on the fly)

5. Some tools for evaluating the model are in 'Evaluate model.ipynb'

   ​         

Requirements
---------------------

Python3 and modules:

```
numpy
matplotlib
parselmouth
praat_parselmouth
scipy
scikit_learn
soundfile
tensorflow==2.1.0
```





