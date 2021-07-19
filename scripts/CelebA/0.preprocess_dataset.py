from zipfile import ZipFile
import shutil
import os
import tqdm


# Download the CelebA dataset from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
#
# Necessary files:
#   -- img_align_celeba.zip
#   -- list_attr_celeba.zip
#
# Files should be located in the scripts/CelebA folder


print("Extracting data")
data_folder = "img_align_celeba"
with ZipFile("img_align_celeba.zip", 'r') as archive:
    archive.extractall()

train_folder = os.path.join(data_folder, "train/images")
valid_folder = os.path.join(data_folder, "valid/images")
test_folder = os.path.join(data_folder, "test/images")

os.makedirs(train_folder)
os.makedirs(valid_folder)
os.makedirs(test_folder)

# This split is in accordance with the official split suggested by the
# dataset creators.
print("Splitting data into train, valid and test splits")
for file_name in tqdm.tqdm(os.listdir(data_folder)):
    if file_name[-4:] != ".jpg":
        continue

    file_idx = int(file_name[:-4])
    if file_idx <= 162770:
        shutil.move(
            os.path.join(data_folder, file_name),
            os.path.join(train_folder, file_name)
        )
    elif file_idx > 162770 and file_idx <= 182637:
        shutil.move(
            os.path.join(data_folder, file_name),
            os.path.join(valid_folder, file_name)
        )
    else:
        shutil.move(
            os.path.join(data_folder, file_name),
            os.path.join(test_folder, file_name)
        )

assert len(os.listdir(data_folder)) == 3
