"""Custom implementation of XRAI algorithm"""

import tensorflow as tf
import numpy as np
from skimage.transform import resize
from skimage import segmentation
from skimage.morphology import disk, dilation


def get_segments_felzenszwalb(
        x: np.array,
        scale_range = [-1.0, 1.0],
        dilation_rad = 5):
	"""Segmentation based on Felzenswalb's graph-based method.

    Args:
        x (np.array): input image
        scale_range (list, optional): Defaults to [-1.0, 1.0].
        dilation_rad (int, optional): Defaults to 5.
	"""
	original_shape = x.shape[:2]

	x_max, x_min = np.max(x), np.min(x)
	x = (x - x_min) / (x_max - x_min)
	x = x * (scale_range[1] - scale_range[0]) + scale_range[0]
	x = resize(
		x,
		(224, 224),
		order=3,
		mode='constant',
		preserve_range=True,
		anti_aliasing=True)

	# Create segment sequences
	segs = []
	for scale in [50, 100, 150, 250, 500, 1200]:
		seg = segmentation.felzenszwalb(
			x, scale=scale, sigma=0.8, min_size=150)
		seg = resize(seg,
			original_shape,
			order=0,
			preserve_range=True,
			mode='constant',
			anti_aliasing=False).astype(int)
		segs.append(seg)

	# Unpack into masks
	masks = []
	for seg in segs:
		for l in range(seg.min(), seg.max() + 1):
			masks.append(seg == l)

	# Dilate segments and return masks
	selem = disk(dilation_rad)
	masks = [dilation(mask, selem=selem) for mask in masks]

	return masks


def xrai_full(attr, segs):
	"""
	Sequentially procure Xrai importances according to attr mask.
		1. Algo altered to prioritise negative importances
		2. We are after entropy reduction
	"""
	output_attr = np.inf * np.ones(shape=attr.shape, dtype=float)
	current_area_perc = 0.0
	current_mask = np.zeros(attr.shape, dtype=bool)

	# Track used and remaining masks
	masks_trace = []
	remaining_masks = {ind: mask for ind, mask in enumerate(segs)}

	added_masks_cnt = 1
	# While the mask area is less than 1.0 and remaining_masks is not empty
	while current_area_perc <= 1.0:
		best_gain = np.inf  # We are after negative importances -> decrease entropy
		best_key = None
		remove_key_queue = []
		for mask_key in remaining_masks:
			mask = remaining_masks[mask_key]

			# Accept for appending only if improvement
			mask_pixel_diff = np.sum(np.logical_and(mask, np.logical_not(current_mask)))
			if mask_pixel_diff < (attr.shape)[0] * (attr.shape)[1] * 0.001:
				remove_key_queue.append(mask_key)
				continue

			# Compute gain
			added_mask = np.logical_and(mask, np.logical_not(current_mask))
			gain = attr[added_mask].mean()

			if gain < best_gain:  # We are after entropy reduction
				best_gain = gain
				best_key = mask_key

		# Remove masks with little use
		for key in remove_key_queue:
			del remaining_masks[key]

		if not remaining_masks:
			break

		# Add best mask to trace
		added_mask = remaining_masks[best_key]
		mask_diff = np.logical_and(added_mask, np.logical_not(current_mask))
		masks_trace.append((mask_diff, best_gain))

		# Attach to current mask and delete its key
		current_mask = np.logical_or(current_mask, added_mask)
		current_area_perc = np.mean(current_mask)
		output_attr[mask_diff] = best_gain
		del remaining_masks[best_key]  # delete used key

		added_masks_cnt += 1

	uncomputed_mask = output_attr == np.inf
	
	# Assign the uncomputed areas a value such that sum is same as ig
	if np.any(uncomputed_mask):
		output_attr[uncomputed_mask] = attr[uncomputed_mask].mean()

	return output_attr
