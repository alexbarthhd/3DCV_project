## Important clarifications on the structure and usage of the custom donkeycar application

Modification of a donkeycar application created using the command

```
donkeycar create --path /path/to/mycar
```

Train and drive an autopilot in a simulator. See http://docs.donkeycar.com/guide/simulator/ for install instructions. Usage has been changed to the commands shown below.

## usage

train with
```
python manage.py train --model 'name of model'
```

drive with
```
python manage.py drive --model 'name of model'
```

Sample models 'track' and 'dagger' are ready to drive. A zipped folder containing labeled images can be extracted to train a new model.