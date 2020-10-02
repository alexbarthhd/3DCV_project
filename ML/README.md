## Important clarifications on the structure and usage of the custom donkeycar application

This is a modification of a donkeycar application created using the command

```
donkeycar create --path /path/to/mycar
```

Train (200 epochs) and drive an autopilot in a simulator. See http://docs.donkeycar.com/guide/simulator/ for install instructions. Usage has been changed to the commands shown below.

## usage

Execute the following commands in a donkey environment (see donkeycar documentation for instructions).

```
python manage.py train --model 'name of model'
```

```
python manage.py drive --model 'name of model'
```

Sample models 'track' and 'dagger' are ready to drive. A zipped folder containing labeled images can be extracted to train a new model.
