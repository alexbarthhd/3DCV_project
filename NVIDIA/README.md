# Donkey Simulator Notes:

**In nvidia.py you can find Nvidia's CNN architecture and our custom improved architecture.**

- Legende for our Datasets: *#run number: ModdelSettings | ModellArchitektur*
- slow_and_precise called our recorded dataset with that we trained our simulator.
- Model 1 to 3 trained on the same dataset called slow_and_preciset
- The dataset slow_and_precise and h5 files were to big for uploaded in GitHub. So here a cloud link for it: https://www.dropbox.com/sh/poaxqblz7nh3ikg/AAB57o1XTzIzMHDo0KcTwBg7a?dl=0

## Start Donkeycar:
1. Start Conda: ```conda activate donkey```
2. Record Data: ```python manage.py drive```
3.
  - *Settings*:           Move myconfig.py to /mysim/
  - *ModellArchitektur*:  Move nvidia.py to donkeycar/donkeycar/parts/
  - *utils*:              Move utils.py to donkeycar/donkeycar/

4. Train a model: ```python /home/mg7/3DCV_project/mysim/manage.py train --model ./models/Nvidia.h5```

5. Test Model: ```python manage.py drive --model models/Nvidia.h5```


## log:
**Legende for our Datasets:** *#run number: ModdelName | ModdelSettings*

1: Nvidia | Nvidia's CNN architecture (see report)

2: NvidiaImproved | custom improved architecture (see report)

3: Nvidia_with_MaxPolling | Nvidia's CNN architecture with additional two MaxPolling layers

4: Nvidia_32000_big_dataset | Nvidia's CNN architecture an a dataset bigger than 32000 batches
