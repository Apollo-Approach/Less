import os
import zipfile
for root, dirs, files in os.walk(os.path.expanduser('~/.gradle/caches/modules-2/files-2.1/')):
    for file in files:
        if file.endswith('.aar') and 'mwdat-core' in file:
            with zipfile.ZipFile(os.path.join(root, file)) as z:
                z.extract('classes.jar')
            with zipfile.ZipFile('classes.jar') as z2:
                for n in z2.namelist():
                    if 'DeviceIdentifier' in n or 'DeviceId' in n:
                        print(n)
