# torch-chimera
Unofficial pytorch implementation of deep clustering family

### Summary

This repo is an **unofficial** implementation of deep clustering [Hershey et al. ICASSP (2016)] and its succedents [Luo et al. ICASSP (2017)], [Wang et al. ICASSP (2018)], [Roux et al. ICASSP (2019)], [Roux et al. IEEE JSTSP (2019)].

The purpose of this repo is to perform single-source signal separation with deep neural networks.
Depicting with the following figure, the mixture of vocal and instruments music (top) is given to the model, and the model predicts mixtures sources (i.e. vocal (center) and instruments music (bottom)).

![Separating into vocals and instruments](separation.png)

[Hershey et al. ICASSP (2016)]: https://arxiv.org/abs/1508.04306
[Luo et al. ICASSP (2017)]: https://arxiv.org/abs/1611.06265
[Wang et al. ICASSP (2018)]: https://ieeexplore.ieee.org/document/8462507
[Roux et al. ICASSP (2019)]: https://arxiv.org/abs/1810.01395
[Roux et al. IEEE JSTSP (2019)]: https://arxiv.org/abs/1810.01395

### Requirements

* `ffmpeg` : 4.3.1
* `museval` : 0.4.0
* `numpy` : 1.18.5
* `pysocks` : 1.7.1
* `pysoundfile` : 0.10.2
* `python` : 3.7.10
* `pytorch` : 1.9.0
* `resampy` : 0.2.2
* `scikit-learn` : 0.23.2
* `scipy` : 1.4.1
* `torchaudio` : 0.9.0

See `requirements.txt` for more information.

### Prediction with pretrained models

#### Pretrained model

Download [pretrained model](https://drive.google.com/file/d/1GBmntQqJkGIeGbOcihYKU52PPJJ-VEF7/view?usp=sharing)

- The pretrained model is based on the combook model [Roux et al. ICASSP (2019)].
- Datasets used are
  - music - `DSD100`, `MedleyDB`, `SLMD`
  - vocal - `DSD100`, `MedleyDB`, `VCTK corpus`, `JVS corpus`.
- Training procedure is following:
  1. train to fit DC (deep clustering) loss for 10 epochs
  2. train to fit DC and MI (mask inference) loss for 10 epochs
    - the scale factors of DC and MI loss are 0.95 and 0.05 respectively
  3. train to fit WA (wave approximation) loss for 5 epochs
- The result of evaluation with `museval` is below.
  - The samples of histogram are median sdr for each window per channel and track.
  - The dashed lines show the median sdr of music and vocal channel.

![evaluation](/combook-dsd100-score.png)


#### Sample script

Source separation can be done with `scripts/chimera-torch.py`.

```shell
python scripts/chimera-torch.py predict \
  --sr 44100 \
  --n-fft 1024 \
  --segment-duration 30.0
  --input-checkpoint models/model.pth \
  --input-file mixture.wav \
  --output-files instrumental.wav vocal.wav 
```

### Training a model

TBA
