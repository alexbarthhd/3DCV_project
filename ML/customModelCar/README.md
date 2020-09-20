## Important clarifications on the structure and usage of the custom donkeycar application

Modification of a donkeycar application created using the command

```
donkeycar create --path /path/to/mycar
```

## usage

train with
```
python manage.py train --model 'name of model'
```

drive with
```
python manage.py drive --model 'name of model'
```

### custom_camera.py
- Contains the class used to replace the simulator camera provided by donkeycar.
- A similar class needs to be created for the camera (e.g. picam) used on the actual car.

### custom_model.py
- Contains model related classes and functions

### manage.py
- The file used to start driving or training the car.
