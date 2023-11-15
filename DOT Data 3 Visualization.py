# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:56:00 2023

@author: sethg
"""

import numpy as np
import ezc3d
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from keras.utils import to_categorical
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import classification_report
import os
import math

# Load and preprocess sensor data
left_wrist_data = pd.read_csv('DOT Data 3/leftwrist.csv')
right_wrist_data = pd.read_csv('DOT Data 3/rightwrist.csv')
left_elbow_data = pd.read_csv('DOT Data 3/leftelbow.csv')
right_elbow_data = pd.read_csv('DOT Data 3/rightelbow.csv')
left_shoulder_data = pd.read_csv('DOT Data 3/leftshoulder.csv')
right_shoulder_data = pd.read_csv('DOT Data 3/rightshoulder.csv')

def magnitude(activity):
    x2 = activity['Quat_X'] * activity['Quat_X']
    y2 = activity['Quat_Y'] * activity['Quat_Y']
    z2 = activity['Quat_Z'] * activity['Quat_Z']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

left_wrist_data['m'] = magnitude(left_wrist_data)
left_wrist_data.head()

plt.figure(figsize=(10,3))
plt.title('Left Wrist - Quat')
plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

plt.figure(figsize=(10,3))
plt.title('Left Wrist - FreeAcc')
plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['FreeAcc_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['FreeAcc_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['FreeAcc_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Left Wrist - Quat (X-axis):')
ax[0].plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_X'], linewidth=0.5, color='r')

ax[1].set_title('Left Wrist - Quat (Y-axis):: ')
ax[1].plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_Y'], linewidth=0.5, color='b')

ax[2].set_title('Left Wrist - Quat (Z-axis): ')
ax[2].plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_Z'], linewidth=0.5, color='g')

ax[3].set_title('Left Wrist - Quat (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(left_wrist_data['SampleTimeFine'], left_wrist_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Left Wrist - FreeAcc (X-axis:)')
ax[0].plot(left_wrist_data['SampleTimeFine'], left_wrist_data['FreeAcc_X'], linewidth=0.5, color='r')

ax[1].set_title('Left Wrist - FreeAcc (Y-axis:) ')
ax[1].plot(left_wrist_data['SampleTimeFine'], left_wrist_data['FreeAcc_Y'], linewidth=0.5, color='b')

ax[2].set_title('Left Wrist - FreeAcc (Z-axis:) ')
ax[2].plot(left_wrist_data['SampleTimeFine'], left_wrist_data['FreeAcc_Z'], linewidth=0.5, color='g')

ax[3].set_title('Left Wrist - FreeAcc (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(left_wrist_data['SampleTimeFine'], left_wrist_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

def magnitude(activity):
    x2 = activity['Quat_X'] * activity['Quat_X']
    y2 = activity['Quat_Y'] * activity['Quat_Y']
    z2 = activity['Quat_Z'] * activity['Quat_Z']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

right_wrist_data['m'] = magnitude(right_wrist_data)
right_wrist_data.head()

plt.figure(figsize=(10,3))
plt.title('Right Wrist - Quat')
plt.plot(right_wrist_data['SampleTimeFine'], right_wrist_data['Quat_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(right_wrist_data['SampleTimeFine'], right_wrist_data['Quat_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(right_wrist_data['SampleTimeFine'], right_wrist_data['Quat_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

plt.figure(figsize=(10,3))
plt.title('Right Wrist - FreeAcc')
plt.plot(right_wrist_data['SampleTimeFine'], right_wrist_data['FreeAcc_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(right_wrist_data['SampleTimeFine'], right_wrist_data['FreeAcc_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(right_wrist_data['SampleTimeFine'], right_wrist_data['FreeAcc_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Right Wrist - Quat (X-axis:)')
ax[0].plot(right_wrist_data['SampleTimeFine'], right_wrist_data['Quat_X'], linewidth=0.5, color='r')

ax[1].set_title('Right Wrist - Quat (Y-axis:) ')
ax[1].plot(right_wrist_data['SampleTimeFine'], right_wrist_data['Quat_Y'], linewidth=0.5, color='b')

ax[2].set_title('Right Wrist - Quat (Z-axis:) ')
ax[2].plot(right_wrist_data['SampleTimeFine'], right_wrist_data['Quat_Z'], linewidth=0.5, color='g')

ax[3].set_title('Magnitude, m: Combined X-Y-Z')
ax[3].plot(right_wrist_data['SampleTimeFine'], right_wrist_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Right Wrist - FreeAcc (X-axis:)')
ax[0].plot(right_wrist_data['SampleTimeFine'], right_wrist_data['FreeAcc_X'], linewidth=0.5, color='r')

ax[1].set_title('Right Wrist - FreeAcc (Y-axis:) ')
ax[1].plot(right_wrist_data['SampleTimeFine'], right_wrist_data['FreeAcc_Y'], linewidth=0.5, color='b')

ax[2].set_title('Right Wrist - FreeAcc (Z-axis:) ')
ax[2].plot(right_wrist_data['SampleTimeFine'], right_wrist_data['FreeAcc_Z'], linewidth=0.5, color='g')

ax[3].set_title('Right Wrist - FreeAcc (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(right_wrist_data['SampleTimeFine'], right_wrist_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

def magnitude(activity):
    x2 = activity['Quat_X'] * activity['Quat_X']
    y2 = activity['Quat_Y'] * activity['Quat_Y']
    z2 = activity['Quat_Z'] * activity['Quat_Z']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

left_elbow_data['m'] = magnitude(left_elbow_data)
left_elbow_data.head()

plt.figure(figsize=(10,3))
plt.title('Left Elbow - Quat')
plt.plot(left_elbow_data['SampleTimeFine'], left_elbow_data['Quat_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(left_elbow_data['SampleTimeFine'], left_elbow_data['Quat_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(left_elbow_data['SampleTimeFine'], left_elbow_data['Quat_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

plt.figure(figsize=(10,3))
plt.title('Left Elbow - FreeAcc')
plt.plot(left_elbow_data['SampleTimeFine'], left_elbow_data['FreeAcc_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(left_elbow_data['SampleTimeFine'], left_elbow_data['FreeAcc_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(left_elbow_data['SampleTimeFine'], left_elbow_data['FreeAcc_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Left Elbow - Quat (X-axis):')
ax[0].plot(left_elbow_data['SampleTimeFine'], left_elbow_data['Quat_X'], linewidth=0.5, color='r')

ax[1].set_title('Left Elbow - Quat (Y-axis): ')
ax[1].plot(left_elbow_data['SampleTimeFine'], left_elbow_data['Quat_Y'], linewidth=0.5, color='b')

ax[2].set_title('Left Elbow - Quat (Z-axis): ')
ax[2].plot(left_elbow_data['SampleTimeFine'], left_elbow_data['Quat_Z'], linewidth=0.5, color='g')

ax[3].set_title('Left Elbow - Quat (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(left_elbow_data['SampleTimeFine'], left_elbow_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Left Elbow - FreeAcc (X-axis):')
ax[0].plot(left_elbow_data['SampleTimeFine'], left_elbow_data['FreeAcc_X'], linewidth=0.5, color='r')

ax[1].set_title('Left Elbow - FreeAcc (Y-axis): ')
ax[1].plot(left_elbow_data['SampleTimeFine'], left_elbow_data['FreeAcc_Y'], linewidth=0.5, color='b')

ax[2].set_title('Left Elbow - FreeAcc (Z-axis): ')
ax[2].plot(left_elbow_data['SampleTimeFine'], left_elbow_data['FreeAcc_Z'], linewidth=0.5, color='g')

ax[3].set_title('Left Elbow - FreeAcc (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(left_elbow_data['SampleTimeFine'], left_elbow_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

def magnitude(activity):
    x2 = activity['Quat_X'] * activity['Quat_X']
    y2 = activity['Quat_Y'] * activity['Quat_Y']
    z2 = activity['Quat_Z'] * activity['Quat_Z']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

right_elbow_data['m'] = magnitude(right_elbow_data)
right_elbow_data.head()

plt.figure(figsize=(10,3))
plt.title('Right Elbow - Quat')
plt.plot(right_elbow_data['SampleTimeFine'], right_elbow_data['Quat_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(right_elbow_data['SampleTimeFine'], right_elbow_data['Quat_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(right_elbow_data['SampleTimeFine'], right_elbow_data['Quat_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

plt.figure(figsize=(10,3))
plt.title('Right Elbow - FreeAcc')
plt.plot(right_elbow_data['SampleTimeFine'], right_elbow_data['FreeAcc_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(right_elbow_data['SampleTimeFine'], right_elbow_data['FreeAcc_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(right_elbow_data['SampleTimeFine'], right_elbow_data['FreeAcc_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Right Elbow - Quat (X-axis):')
ax[0].plot(right_elbow_data['SampleTimeFine'], right_elbow_data['Quat_X'], linewidth=0.5, color='r')

ax[1].set_title('Right Elbow - Quat (Y-axis): ')
ax[1].plot(right_elbow_data['SampleTimeFine'], right_elbow_data['Quat_Y'], linewidth=0.5, color='b')

ax[2].set_title('Right Elbow - Quat (Z-axis): ')
ax[2].plot(right_elbow_data['SampleTimeFine'], right_elbow_data['Quat_Z'], linewidth=0.5, color='g')

ax[3].set_title('Right Elbow - Quat (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(right_elbow_data['SampleTimeFine'], right_elbow_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Right Elbow - FreeAcc (X-axis):')
ax[0].plot(right_elbow_data['SampleTimeFine'], right_elbow_data['FreeAcc_X'], linewidth=0.5, color='r')

ax[1].set_title('Right Elbow - FreeAcc (Y-axis): ')
ax[1].plot(right_elbow_data['SampleTimeFine'], right_elbow_data['FreeAcc_Y'], linewidth=0.5, color='b')

ax[2].set_title('Right Elbow - FreeAcc (Z-axis): ')
ax[2].plot(right_elbow_data['SampleTimeFine'], right_elbow_data['FreeAcc_Z'], linewidth=0.5, color='g')

ax[3].set_title('Right Elbow - FreeAcc (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(right_elbow_data['SampleTimeFine'], right_elbow_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

def magnitude(activity):
    x2 = activity['Quat_X'] * activity['Quat_X']
    y2 = activity['Quat_Y'] * activity['Quat_Y']
    z2 = activity['Quat_Z'] * activity['Quat_Z']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

left_shoulder_data['m'] = magnitude(left_shoulder_data)
left_shoulder_data.head()

plt.figure(figsize=(10,3))
plt.title('Left Shoulder - Quat')
plt.plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['Quat_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['Quat_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['Quat_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

plt.figure(figsize=(10,3))
plt.title('Left Shoulder - FreeAcc')
plt.plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['FreeAcc_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['FreeAcc_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['FreeAcc_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Left Shoulder - Quat (X-axis):')
ax[0].plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['Quat_X'], linewidth=0.5, color='r')

ax[1].set_title('Left Shoulder - Quat (Y-axis): ')
ax[1].plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['Quat_Y'], linewidth=0.5, color='b')

ax[2].set_title('Left Shoulder - Quat (Z-axis): ')
ax[2].plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['Quat_Z'], linewidth=0.5, color='g')

ax[3].set_title('Left Shoulder - Quat (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Left Shoulder - FreeAcc (X-axis):')
ax[0].plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['FreeAcc_X'], linewidth=0.5, color='r')

ax[1].set_title('Left Shoulder - FreeAcc (Y-axis): ')
ax[1].plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['FreeAcc_Y'], linewidth=0.5, color='b')

ax[2].set_title('Left Shoulder - FreeAcc (Z-axis): ')
ax[2].plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['FreeAcc_Z'], linewidth=0.5, color='g')

ax[3].set_title('Left Shoulder - FreeAcc (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(left_shoulder_data['SampleTimeFine'], left_shoulder_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

def magnitude(activity):
    x2 = activity['Quat_X'] * activity['Quat_X']
    y2 = activity['Quat_Y'] * activity['Quat_Y']
    z2 = activity['Quat_Z'] * activity['Quat_Z']
    m2 = x2 + y2 + z2
    m = m2.apply(lambda x: math.sqrt(x))
    return m

right_shoulder_data['m'] = magnitude(right_shoulder_data)
right_shoulder_data.head()

plt.figure(figsize=(10,3))
plt.title('Right Shoulder - Quat')
plt.plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['Quat_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['Quat_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['Quat_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

plt.figure(figsize=(10,3))
plt.title('Right Shoulder - FreeAcc')
plt.plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['FreeAcc_X'], linewidth=0.5, color='r', label='x axis')
plt.plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['FreeAcc_Y'], linewidth=0.5, color='b', label='y axis')
plt.plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['FreeAcc_Z'], linewidth=0.5, color='g', label='z axis')
#plt.plot(left_wrist_data['SampleTimeFine'], left_wrist_data['Quat_W'], linewidth=0.5, color='y', label='w')
plt.xlabel('SampleTimeFine')
plt.ylabel('Quat')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5));

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Right Shoulder - Quat (X-axis):')
ax[0].plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['Quat_X'], linewidth=0.5, color='r')

ax[1].set_title('Right Shoulder - Quat (Y-axis): ')
ax[1].plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['Quat_Y'], linewidth=0.5, color='b')

ax[2].set_title('Right Shoulder - Quat (Z-axis): ')
ax[2].plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['Quat_Z'], linewidth=0.5, color='g')

ax[3].set_title('Right Shoulder - Quat (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(10,8))

ax[0].set_title('Right Shoulder - FreeAcc (X-axis):')
ax[0].plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['FreeAcc_X'], linewidth=0.5, color='r')

ax[1].set_title('Right Shoulder - FreeAcc (Y-axis): ')
ax[1].plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['FreeAcc_Y'], linewidth=0.5, color='b')

ax[2].set_title('Right Shoulder - FreeAcc (Z-axis): ')
ax[2].plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['FreeAcc_Z'], linewidth=0.5, color='g')

ax[3].set_title('Right Shoulder - FreeAcc (Magnitude, m: Combined X-Y-Z)')
ax[3].plot(right_shoulder_data['SampleTimeFine'], right_shoulder_data['m'], linewidth=0.5, color='k')

fig.subplots_adjust(hspace=.5)