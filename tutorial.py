# -*- coding: utf-8 -*-
"""
Created on Thu Feb 9 10:25:23 2023

@author: Vincent Bazinet
"""

# Import statements
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from neuromaps.images import (construct_shape_gii, load_data,
                              annot_to_gifti, relabel_gifti, dlabel_to_gifti)

# Set-up matplotlib properties
flierprops = dict(marker='+',
                  markerfacecolor='lightgray',
                  markeredgecolor='lightgray')
medianprops = dict(color='black')
plt.rcParams.update({'axes.spines.top': False})
plt.rcParams.update({'axes.spines.right': False})
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'font.family': 'Calibri'})
plt.rcParams.update({'font.weight': 'light'})

# Change current working directory (set it to whatever you want)
os.chdir((os.path.expanduser("~") + "/OneDrive - McGill University/"
          "presentations/Hiball Winter School/"))

#%% neuromaps.datasets
# API: netneurolab.github.io/neuromaps/api.html#module-neuromaps.datasets

from neuromaps import datasets

'''
Fetching atlases
'''

atlases_MNI152 = datasets.fetch_atlas('mni152', '1mm')
str(atlases_MNI152['6Asym_T1w'])
atlases_fsaverage = datasets.fetch_atlas("fsaverage", '41k')
atlases = datasets.fetch_all_atlases()

'''
Fetching annotations
annotation information on the wiki:
https://github.com/netneurolab/neuromaps/wiki
'''

datasets.available_annotations(source='all')
all_annotations = datasets.fetch_annotation(source='all')
margulies_gradients = datasets.fetch_annotation(source='margulies2016')
fsaverage_annotations = datasets.fetch_annotation(space='fsaverage')

datasets.available_tags()
receptors = datasets.fetch_annotation(tags='receptors')

'''
Uploading a new annotation
'''

# datasets.upload_annotation(files, email_address)

#%% neuromaps.transforms
# API: netneurolab.github.io/neuromaps/api.html#module-neuromaps.transforms

from neuromaps import transforms

'''
Fetch the neurosynth PC1 map (which will be used in this tutorial)
'''

neurosynth_mni152 = datasets.fetch_annotation(source='neurosynth')

'''
Transform the neurosynth map to fsaverage
'''

# Do the transformation
gii_images = transforms.mni152_to_fsaverage(
    neurosynth_mni152,
    fsavg_density='41k')

# Let's set values in the medial wall to NaN
neurosynth_fsaverage = []
for mask_gii, hemi_gii in zip(atlases_fsaverage['medial'], gii_images):
    # mask_gii = atlases_fsaverage['medial'][0]
    # hemi_gii = gii_images[0]
    mwall_mask = nib.load(mask_gii).agg_data()
    neurosynth_hemi_data = hemi_gii.agg_data()
    neurosynth_hemi_data[mwall_mask == 0] = np.nan
    neurosynth_fsaverage.append(
        construct_shape_gii(neurosynth_hemi_data)
        )

# Save the gifti image
for gii, hemi in zip(neurosynth_fsaverage, ['L', 'R']):
    nib.save(gii, f"tutorial_results/neurosynth_fsaverage_41k_{hemi}.shape.gii")

#%% neuromaps.nulls
# API: netneurolab.github.io/neuromaps/api.html#module-neuromaps.nulls

from neuromaps.nulls import alexander_bloch

'''
Generate 1000 null maps for our fsaverage neurosynth map
Takes about ~20-30 minutes to run
'''

nulls = alexander_bloch(
    neurosynth_fsaverage, atlas='fsaverage', density='41k', n_perm=1000)

# np.save('tutorial_results/neurosynth_fsaverage_41k_nulls.npy', nulls)
nulls = np.load('tutorial_results/neurosynth_fsaverage_41k_nulls.npy')

'''
Let's save the data of these nulls as gifti images, for visualization
'''

hemi_vertices, = neurosynth_fsaverage[0].agg_data().shape
nulls_gii_L = construct_shape_gii(nulls[:hemi_vertices, :])
nulls_gii_R = construct_shape_gii(nulls[hemi_vertices:, :])
nib.save(nulls_gii_L, 'tutorial_results/neurosynth_fsaverage_41k_nulls_L.shape.gii')
nib.save(nulls_gii_R, 'tutorial_results/neurosynth_fsaverage_41k_nulls_R.shape.gii')

#%% neuromaps.stats
# API: netneurolab.github.io/neuromaps/api.html#module-neuromaps.stats

from neuromaps import stats

'''
Compare our neurosynth fsaverage image with images in our repository that are
originally in fsaverage
'''

fsaverage_annotations = datasets.fetch_annotation(space='fsaverage')
annotation_labels = ['gene PC1',
                     '5HT1b',
                     '5HT2a',
                     '5HT1a',
                     '5HTT',
                     '5HT4',
                     'GABA']
'''
Let resample the fsaverage images to 41k
'''

fsaverage_41k_annotations = {}
for labels, annotation in fsaverage_annotations.items():
    fsaverage_41k_annotations[labels] = transforms.fsaverage_to_fsaverage(
        annotation, target_density='41k')

'''
Compute the correlations: let's try with random permutations
'''

random_permutations = np.zeros((nulls.shape))
neurosynth_data = load_data(neurosynth_fsaverage)
for i in trange(1000):
    random_permutations[:, i] = np.random.permutation(neurosynth_data)

# Takes ~1:30 minutes
r_all, p_all = [], []
for annotation in fsaverage_41k_annotations.values():
    r, p = stats.compare_images(neurosynth_fsaverage, annotation,
                                nulls=random_permutations,
                                nan_policy='omit', metric='pearsonr')
    r_all.append(r)
    p_all.append(p)

# Let's do it manually (to get each correlation for permutations)
'''
n_annotations = len(fsaverage_41k_annotations)
r_all = np.zeros((n_annotations))
r_all_nulls = np.zeros((n_annotations, 1000))
p_all = np.zeros((n_annotations))
for i, (properties, annotation) in enumerate(fsaverage_41k_annotations.items()):

    annotation_data = load_data(annotation)

    # Compute correlation for empirical data
    r_all[i] = stats.efficient_pearsonr(neurosynth_data, annotation_data,
                                           return_pval=False, nan_policy='omit')

    # Compute correlations for permuted data
    r_all_nulls[i,:] = stats.efficient_pearsonr(random_permutations, annotation_data,
                                                return_pval=False, nan_policy='omit')

    # Get p-value
    emp = r_all[i]
    perm = r_all_nulls[i,:]
    p_all[i] = len(np.where(abs(perm-np.mean(perm)) > abs(emp-np.mean(perm)))[0])/1000

np.save('tutorial_results/neurosynth_r.npy', r_all)
np.save('tutorial_results/neurosynth_r_perm.npy', r_all_nulls)
np.save('tutorial_results/neurosynth_p_perm.npy', p_all)
'''

r_all = np.load('tutorial_results/neurosynth_r.npy')
r_all_nulls = np.load('tutorial_results/neurosynth_r_perm.npy')
p_all = np.load('tutorial_results/neurosynth_p_perm.npy')

plt.figure()
plt.ylabel("r")
plt.boxplot(r_all_nulls.T, flierprops=flierprops, medianprops=medianprops,
            showcaps=False)
plt.xticks(np.arange(1, 8), annotation_labels)
# plt.scatter(np.arange(1,8), r_all, color='black')

'''
Compute the correlations: now let's try with our spatial
autocorrelation-preserving surrogate annotations
'''

# Takes ~1:30 minutes
r_all, p_all = [], []
for annotation in fsaverage_41k_annotations.values():
    r, p = stats.compare_images(neurosynth_fsaverage, annotation, nulls=nulls,
                                nan_policy='omit', metric='pearsonr')
    r_all.append(r)
    p_all.append(p)

# Let's do it manually (to get each correlation for permutations)
'''
n_annotations = len(fsaverage_41k_annotations)
r_all = np.zeros((n_annotations))
r_all_nulls = np.zeros((n_annotations, 1000))
p_all = np.zeros((n_annotations))
for i, (properties, annotation) in enumerate(fsaverage_41k_annotations.items()):

    annotation_data = load_data(annotation)

    # Compute correlation for empirical data
    r_all[i] = stats.efficient_pearsonr(neurosynth_data, annotation_data,
                                           return_pval=False, nan_policy='omit')

    # Compute correlations for permuted data
    r_all_nulls[i,:] = stats.efficient_pearsonr(nulls, annotation_data,
                                                return_pval=False, nan_policy='omit')

    # Get p-value
    emp = r_all[i]
    perm = r_all_nulls[i,:]
    p_all[i] = len(np.where(abs(perm-np.mean(perm)) > abs(emp-np.mean(perm)))[0])/1000

np.save('tutorial_results/neurosynth_r.npy', r_all)
np.save('tutorial_results/neurosynth_r_spin.npy', r_all_nulls)
np.save('tutorial_results/neurosynth_p_spin.npy', p_all)
'''

r_all = np.load('tutorial_results/neurosynth_r.npy')
r_all_nulls = np.load('tutorial_results/neurosynth_r_spin.npy')
p_all = np.load('tutorial_results/neurosynth_p_spin.npy')

plt.figure()
plt.ylabel("r")
plt.boxplot(r_all_nulls.T, flierprops=flierprops, medianprops=medianprops,
            showcaps=False)
plt.xticks(np.arange(1, 8), labels=annotation_labels)
# plt.scatter(np.arange(1,8), r_all, color='black')

#%% neuromaps.parcellate
# API: netneurolab.github.io/neuromaps/api.html#module-neuromaps.parcellate

from neuromaps.parcellate import Parcellater

from netneurotools.datasets import fetch_schaefer2018
from netneurotools.plotting import plot_fsaverage

def plot_parcellated_brain(data):
    '''
    Helper function to plot a surface mesh of the brain with parcellated data
    on it.
    '''

    lhannot, rhannot = fetch_schaefer2018('fsaverage')['100Parcels7Networks']
    im = plot_fsaverage(data, lhannot=lhannot, rhannot=rhannot,
                        data_kws={'representation': 'wireframe',
                                  'line_width': 4.0})
    return im

'''
For this experiments: let's compare the margulies gradient to receptor maps!
'''

# fsaverage41k
parcels_fsav_41k = fetch_schaefer2018('fsaverage6')['100Parcels7Networks']
parcels_fsav_41k = annot_to_gifti(parcels_fsav_41k)
parcels_fsav_41k = relabel_gifti(parcels_fsav_41k)

#fsLR32k
parcels_fslr_32k = fetch_schaefer2018('fslr32k')['100Parcels7Networks']
parcels_fslr_32k = dlabel_to_gifti(parcels_fslr_32k)
parcels_fslr_32k = relabel_gifti(parcels_fslr_32k)

# mni152
parcels_mni152 = 'data/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'

parc_fsLR = Parcellater(parcels_fslr_32k, 'fslr', resampling_target=None)
parc_mni152 = Parcellater(parcels_mni152, 'mni152', resampling_target='parcellation')
parc_fsav = Parcellater(parcels_fsav_41k, 'fsaverage', resampling_target='parcellation')

# Parcellate margulies gradient
FCPC1_fsLR = datasets.fetch_annotation(desc='fcgradient01')
FCPC1_100 = parc_fsLR.fit_transform(FCPC1_fsLR, 'fslr', ignore_background_data=True)

# Let's see what it looks like
plot_parcellated_brain(FCPC1_100)

# Let's fetch all the receptor, related annotations
receptor_annotations = datasets.fetch_annotation(tags='receptors')

# Let's parcellate the data for all receptors
parcellated_receptors = {}
for (source, desc, space, density), annotation in receptor_annotations.items():

    if (source == 'beliveau2017' or source == 'norgaard2021' and
        space == 'MNI152'):
        continue
    else:
        if space == 'MNI152':
            parc = parc_mni152.fit_transform(
                annotation, 'mni152', ignore_background_data=True)
        elif space == 'fsaverage':
            parc = parc_fsav.fit_transform(
                annotation, 'fsaverage', ignore_background_data=True)
        parcellated_receptors[desc] = parc

# Let's generate surrogate maps for our FCPC1 map
nulls = alexander_bloch(FCPC1_100, atlas='fsaverage', density='41k',
                        parcellation=parcels_fsav_41k)

results = {}
for desc, receptor_data in parcellated_receptors.items():
    results[desc] = stats.compare_images(FCPC1_100, receptor_data.flatten(),
                                         nulls=nulls)

'''
Among the significant results:
    methylreboxetine -> NET
    mrb -> NET
'''

plot_parcellated_brain(parcellated_receptors['methylreboxetine'].flatten())
plot_parcellated_brain(parcellated_receptors['mrb'].flatten())

'''
We could do the same thing for the BigBrain cortical layers!
'''

layer4_thickness = ('data/BigBrain/layer4_thi_L.shape.gii',
                    'data/BigBrain/layer4_thi_R.shape.gii')

parcels_BigBrain = ('data/BigBrain/lh.Schaefer2018_100Parcels_7Networks_order.label.gii',
                    'data/BigBrain/rh.Schaefer2018_100Parcels_7Networks_order.label.gii')
parcels_BigBrain = relabel_gifti(parcels_BigBrain)


parc_BigBrain = Parcellater(parcels_BigBrain, 'fsaverage')

layer4_thi_parcellated = parc_BigBrain.fit_transform(
    layer4_thickness, 'fsaverage', ignore_background_data=True,
    background_value=0)

plot_parcellated_brain(layer4_thi_parcellated)

#%% abagen

import abagen
import pandas as pd

schaefer_atlas = fetch_schaefer2018(version='fsaverage5')['100Parcels7Networks']
schaefer_atlas = annot_to_gifti(schaefer_atlas)
schaefer_info = pd.read_csv("data/Schaefer2018_100Parcels_7Networks_order_info.csv")

# Takes a couple of minutes to run
expression = abagen.get_expression_data(schaefer_atlas, schaefer_info)
# expression.to_csv("tutorial_results/expresion.csv", index=False)

expression = pd.read_csv("tutorial_results/expresion.csv")

# Let's look at SNCA
plot_parcellated_brain(expression['SNCA'].values)

# We see some missing values. To solve this issue:
expression = abagen.get_expression_data(schaefer_atlas, schaefer_info,
                                        missing='centroids')

