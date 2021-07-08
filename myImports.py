from flask_ngrok import run_with_ngrok
from logging import debug
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify, make_response
from werkzeug.utils import secure_filename
from logging import debug
# from werkzeug.utils import secure_filename
# from werkzeug.datastructures import  FileStorage
# from flask_uploads import UploadSet, configure_uploads, IMAGES
import uuid
import shutil
import zipfile
import time
import copy
import pickle
from barbar import Bar
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from PIL import Image
import cv2
from torch.optim import lr_scheduler  #this was commented
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchsummary import summary
from tqdm import tqdm
from pathlib import Path
from multiprocessing import freeze_support
import gc
import json
RANDOMSTATE = 0
import os

# from Utils import Utils
# from UploadFiles import UploadFiles
# from CBIRDataset import CBIRDataset

#Find if any accelerator is presented, if yes switch device to use CUDA or else use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

############################
from collections import defaultdict
# from __future__ import division

# from __future__ import print_function, division
