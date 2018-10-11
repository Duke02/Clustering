#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2

def createData():
	# Equal to N in other comments.
	num_of_samples = 100
	# offsets are added together to make them more dramatic, but not
	# too much, such as if added 1 to each.
	offset = np.random.rand(2, num_of_samples) + np.random.rand(2, num_of_samples)
	# Complete list of points (more than we want)
	pre_theta = np.linspace(0, 2 * np.pi, num_of_samples * 2)
	# randomly select from the given pre_theta's the actual number we want.
	theta = np.random.choice(pre_theta, num_of_samples)
	# random radius for each circles.
	radius = np.random.rand(2, num_of_samples)
	# Original points
	points = np.asarray([np.cos(theta) * radius[0] + offset[0], np.sin(theta) * radius[0] + offset[1]])
	# opposite points
	opp_points = np.asarray([np.cos(theta) * radius[1] - offset[0], np.sin(theta) * radius[1] - offset[1]])
	# reshape the output arrays to a (2, N)
	# If you get an error here, that's because you're concatting
	# somewhere you should be adding. Or something similar.
	points = np.reshape(points, (2, num_of_samples))
	opp_points = np.reshape(opp_points, (2, num_of_samples))
	return np.concatenate((points, opp_points), axis=1).T

# Adding centroids allows for plotting the centers of the data
# Adding labels and colors allows for coloring of the different
# groups of data.
# len(colors) must equal the number of clusters, or the k used
# in the k-means algorithm.
def plot(data, centroids=[], labels=[], colors=[]):
	# clear the current frame
	plt.clf()
	if len(labels) == 0 or len(colors) == 0:
		_colors = ["#999999"]
		_labels = [0 for x in range(len(data))]
	else:
		_colors = colors
		_labels = labels
	for i in range(len(data)):
		# scatter the points
		plt.scatter(data[i,0], data[i,1], c=_colors[_labels[i]])
	for centroid in centroids:
		plt.scatter(centroid[0], centroid[1], marker='*', c="#000000", s=100)
	# show the plot
	plt.show()
	# redraw the plot
	plt.draw()

# The function is a modified version of this SO answer
# => https://stackoverflow.com/questions/40434352/updating-matplotlib-plot-when-clicked
def onclick(event):
	# if left mouse click
	if event.button == 1:
		# recreate new data
		data = createData()
		# Cluster data again.
		centroids, labels = cluster_the_data(data)
		# Pretty Colors
		colors = ["#0000FF", "#FF0000"]
		# plot that data
		plot(data, centroids, labels, colors)

def cluster_the_data(data):
	centroids, labels = kmeans2(data=data, k=2)
	return centroids, labels

def main():
	# generate the data
	points = createData()
	# Cluster the data and such.
	centroids, labels = cluster_the_data(points)
	# Colors per each grouping.
	colors = ["#0000FF", "#FF0000"]
	# Create the plot and the axis for it.
	figure, axis = plt.subplots()
	# add the event listener for a button press
	figure.canvas.mpl_connect('button_press_event', onclick)
	# scatter the points on the plot
	plot(points, centroids, labels, colors)

if __name__ == "__main__":
	main()
