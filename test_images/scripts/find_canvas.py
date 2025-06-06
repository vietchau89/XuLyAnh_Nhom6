import os

for root, dirs, files in os.walk(os.__file__.split("os.py")[0]):
    for file in files:
        if file == "canvas.py":
            print(os.path.join(root, file))
