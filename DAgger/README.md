## Important clarifications on the structure and usage of the custom donkeycar application

Modification of a donkeycar application created using the command

```
donkeycar create --path /path/to/mycar
```

## usage

```
python manage.py drive
```

### custom_camera.py
- Contains everything required for dagger

### custom_model.py
- Contains model related classes and functions

### custom_controller.py and custom_web.py
- modified versions of donkeycar parts to allow throttle triggered recording during ai control
