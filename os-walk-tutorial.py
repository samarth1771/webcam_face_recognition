import os

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "training_data")

## How os.walk() works, Uncomment these lines to find out

# print(base_dir)
# print(image_dir)
#
# for root, dirs, files in os.walk(image_dir):
#     print("root: ", root)
#     print("Dirs: ", dirs)
#     print("Files: ", files)

y_lables = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            x_train.append(path)
            y_lables.append(label)


print(y_lables, x_train)
