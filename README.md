# WeaponDetectionSystem

This is an application of computer vision for the detection of the weapons using a trained net based on YOLOv5.

The system also calculates the distance of the weapons from the camera based on assumptions of their average widths.

The weapons detected by the application are: Automatic Rifle, Bazooka, Grenade Launcher, Handgun, Knife, Shotgun, SMG, Sniper, Sword.


The parameters you could pass to the program are:
-c=camera_params # the filename for intrinsic (and extrinsic) parameters of the camera
-m=net # the filename of net to use for the detection
-n=classes_names # the filename with the names of the classes detected by the net
-w=classes_width # the filename with the width of the objects detected by the net
-l # print some informations about the performance


Example command line to use the program:
./x64/Release/WeaponDetectionSystem.exe -c=camera_data.yml -m=models/net.onnx -n=weapon_classes.names -w=width.names -l

To quit the program use the key: 'ESC', 'q' or 'Q'