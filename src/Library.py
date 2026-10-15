import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from math import radians, cos, sin, asin, sqrt
import math
from random import sample

import folium
import webbrowser

from scipy.stats import pearsonr, linregress

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, classification_report, cohen_kappa_score, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import RobustScaler

from scipy.spatial.distance import pdist, squareform

import time
import os
import operator
import pickle
from tqdm import tqdm
