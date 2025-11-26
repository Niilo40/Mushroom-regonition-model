import gdown

url = "https://drive.google.com/file/d/1_dVVy3dHFo-ql_wxZrGFhu8_9n3sStnh/view?usp=sharing"  # replace FILE_ID with actual ID
output = "mushroom_dataset.zip"

gdown.download(url, "mushroom_dataset.zip", fuzzy=True, quiet=False)
