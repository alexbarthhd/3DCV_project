# Donkey Simulator Notes:

## Start Donkeycar:
1. Start Conda: ```conda activate donkey```
2. Record Data: ```python manage.py drive```
3.
  - Settings: Move myconfig.py to /mysim/
  - **ModellArchitektur**: Move rojin.py to donkeycar/donkeycar/parts/
  - utils: Move utils.py to donkeycar/donkeycar/

4. Train a model: ```python /home/mg7/3DCV_project/mysim/manage.py train --tub ./data/slow_and_precise/ --model ./models/slow_and_precise_rojin.h5```

5. Test Model: ```python manage.py drive --model models/slow_and_precise_rojin.h5```



## log (Durchläufe):
**Legende:** *#Durchlaufnummer/ModdelSettings/ModellArchitektur*

1: Dense without Dropouts; 24 --> 36 --> 48 --> 64 --> 64

2: Dense with 2 Dropouts; 24 --> 36 --> 48 --> 64 --> 64

3: Dense with Dropouts; 24 --> 36 --> 48 --> 64 --> 64

4: Dense without Dropouts; 16 --> 16 --> 32 ---> 32 --> MaxPool ---> 64

5: Dense with Dropouts; 16 --> 16 --> MaxPool --> 32 ---> 32 --> MaxPool ---> 64
- --> Gut auf der lernstrecke
- --> sehr schlecht auf andere Strecken

6: Dense without Dropouts; 16 --> 16 --> MaxPool --> 32 ---> 32 --> MaxPool --> 64
outputs1_loss: 0.0012
Epoch 00032: early stopping
Training completed in 0:12:28.
- --> Ok auf der lernstrecke
- --> gut auf andere Strecken

7: Dense without Dropouts; 8 --> 8 --> 16 --> 16 --> MaxPool --> 32
outputs1_loss: 0.0017
Epoch 00029: early stopping
Training completed in 0:09:25.
- --> sehr gut auf der lern strecke
- --> sehr schlecht auf andere Strecken


8: Dense without Dropouts; RojinLinear1; 24 --> 36 --> MaxPool --> 48 --> 64 --> MaxPool --> 64
l_outputs1_loss: 0.0023
Epoch 00025: early stopping
Training completed in 0:33:05.
- --> Ok auf der lernstrecke
- --> sehr schlecht auf andere Strecken

9: Dense with Dropouts; with angle; with throttle; 8 --> 8 --> MaxPool --> 16 --> 16 --> MaxPool --> 32
outputs1_loss: 0.0041
Epoch 00019: early stopping
Training completed in 0:04:41.
- --> sehr gut auf der lernstrecke
- --> sehr schlecht auf andere Strecken


10: Dense with Dropouts; 8 --> 8 --> MaxPool --> 16 --> 16 --> MaxPool --> 32
outputs1_loss: 0.0030
Epoch 00022: early stopping
Training completed in 0:05:34.
- --> sehr schlecht auf der lernstrecke
- --> sehr schlecht auf andere Strecken

11: Dense without Dropouts; with angle; with throttle; 16 --> 16 --> MaxPool --> 32 --> 32 --> MaxPool --> 64 --> 64
outputs1_loss: 0.0023
Epoch 00024: early stopping
Training completed in 0:12:42.
- --> gut auf der lernstrecke
- --> sehr schlecht auf andere Strecken


12: Dense without Dropouts; with angle; with throttle; 16 --> 16 --> MaxPool --> 32 ---> 32 --> MaxPool --> 64 --> 128
outputs1_loss: 0.0010
Epoch 00037: early stopping
Training completed in 0:20:03.
- --> schlecht auf der lernstrecke
- --> sehr schlecht auf andere Strecken


13: Dense without any Dropouts; 16 --> 16 --> MaxPool --> 32 ---> 32 --> MaxPool --> 64
outputs1_loss: 9.3376e-04
Epoch 00037: early stopping
Training completed in 0:12:54.
- --> schlecht auf der lernstrecke; sehr unstabil
- --> schlecht auf andere Strecken

14: Dense with Dropouts; 16 --> 16 --> MaxPool --> 32 --> 32 --> MaxPool --> 64
- outputs1_loss: 0.0028
Epoch 00026: early stopping
Training completed in 0:16:15.
- --> gut auf der lernstrecke; hält eine runde aus; sehr zentrierter
- --> schlecht auf andere Strecken

15: Dense without Dropouts; with angle; with throttle; 16 --> 16 --> MaxPool --> 32 ---> 32 --> MaxPool --> 64
outputs1_loss: 9.6716e-04
Epoch 00036: early stopping
Training completed in 0:17:47.
- --> sehr gut auf der lernstrecke
- --> sehr schlecht auf andere Strecken

16: Dense without Dropouts; 16 --> 16 --> MaxPool --> 32 --> 32 --> MaxPool --> 64
outputs1_loss: 0.0014
Epoch 00030: early stopping
Training completed in 0:15:30.
- --> gut auf der lernstrecke
- --> ok auf andere Strecken

17: Dense without Dropouts; 16 --> 16 --> 32 --> 32 --> MaxPool --> 64
outputs1_loss: 0.0011
Epoch 00025: early stopping
Training completed in 0:12:37.
- --> sehr gut auf der lernstrecke; zimmlich Flot
- --> sehr schlecht auf andere Strecken

18: Dense without Dropouts; 16 --> MaxPool 16 --> MaxPool 32 --> MaxPool 32 --> MaxPool --> 64
outputs1_loss: 0.0038
Epoch 00032: early stopping
Training completed in 0:19:16.
- --> sehr gut auf der lernstrecke; zimmlich Flot
- --> sehr schlecht auf andere Strecken

## Summary:
Sehr gut auf der lernstrecke funktioniert am besten. **Dense without Dropouts; 16 --> 16 --> MaxPool --> 32 --> 32 --> MaxPool --> 64 --> 64** (Durchlauf 7 und 11).
Was aber sowohl solide Leistung auf der lernstrecke als auch auf der andere Strecken ist: **Dense without Dropouts; 16 --> 16 --> MaxPool --> 32 ---> 32 --> MaxPool --> 64** (Durchlauf 6 und 16).
