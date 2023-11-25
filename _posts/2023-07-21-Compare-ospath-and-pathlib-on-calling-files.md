---
title: Compare os.path and pathlib on calling files
author: fanghao
date: 2023-07-21 02:58:05 +0200
date: 2023-07-21 02:58:05 +0200
categories: [coding tips]
tags: [Python, debugging tips]
---

# Compare os.path and pathlib on calling files

It is common to call a file in a Python project. However, the files are not always in the working directory. I compared the two common methods of calling a file by os.path function and pathlib function. 

# 1. Set the scene

suppose that we have a project like this, where the 

1. test1.csv is in a subfolder
2. test2.csv is in another folder which is in the same level as the working directory
3. test3.csv in in the parent’s parent folder
4. test4.csv is in the parent’s parent’s folder’s subfolder

![01](/commons/20230721/folders.jpg)  

# 2.Example code for each situation

## 1. test1.csv is in a subfolder

```python
import os
import pandas as pd
from pathlib import Path

# Define the subfolder and file name
subfolder = 'subworking'
filename = 'test1.csv'

# Construct the path
file_path1 = os.path.join(subfolder, filename)
file_path2 = Path(subfolder) / filename

file1 = pd.read_csv(file_path1)
file2 = pd.read_csv(file_path1)
print(file1.head())
print(file2.head())
```

## 2. test2.csv is in another folder which is in the same level as the working directory

```python
import os
import pandas as pd
from pathlib import Path

# Define the folder and file name
sibling_folder = 'parallel'
filename = 'test2.csv'

# For os.path
file_path1 = os.path.join(os.path.dirname(os.getcwd()), sibling_folder, filename)

# For pathlib
file_path2 = Path(os.getcwd()) / sibling_folder / filename

file1 = pd.read_csv(file_path1)
file2 = pd.read_csv(file_path1)
print(file1.head())
print(file2.head())

```

## 3. test3.csv in in the parent’s parent folder

```python
import os
import pandas as pd
from pathlib import Path

# Define the file name
filename = 'test3.csv'

# Construct the path to the file in the parent directory
file_path1 =os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), filename)
file_path2 = Path.cwd().parent.parent / filename

file1 = pd.read_csv(file_path1)
file2 = pd.read_csv(file_path1)
print(file1.head())
print(file2.head())
```

## 4. test4.csv is in the parent’s parent’s folder’s subfolder

```python
import os
import pandas as pd
from pathlib import Path

# Define the file name
filename = 'test4.csv'

# Construct the path to the file in the parent directory
file_path1 =os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "parallel1", filename)
file_path2 = Path.cwd().parent.parent / "parallel1"/filename

file1 = pd.read_csv(file_path1)
file2 = pd.read_csv(file_path1)
print(file1.head())
print(file2.head())
```