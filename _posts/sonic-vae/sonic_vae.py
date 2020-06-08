from retro_contest.local import make
from time import sleep
from gym import spaces
from os import listdir, rename
from sys import argv
from PIL import Image
import tensorflow as tf
import csv, pickle
import numpy, math, random
import os.path
import msvcrt

SCREEN_X = 320
SCREEN_Y = 224

H_SCREEN_X = 160
H_SCREEN_Y = 112

SQRT_2_OVER_2 = 0.7071

NUM_ACTION = 8

log_name = "vae"

weight_sigma = .01
adam_alpha = 1e-5

def createWeightedCenterMask():
	ret = []

	for y in range(SCREEN_Y):
		temp = []
		for x in range(SCREEN_X):
			dx = abs(x - H_SCREEN_X)
			dy = abs(y - H_SCREEN_Y)

			dx = (float(dx) * SQRT_2_OVER_2) / H_SCREEN_X
			dy = (float(dy) * SQRT_2_OVER_2) / H_SCREEN_Y

			dx = SQRT_2_OVER_2 - dx
			dy = SQRT_2_OVER_2 - dy

			val = math.sqrt(dx**2 + dy**2)
			val = max(min(val, 1.0), .25)
			temp.append([val])

		ret.append(temp)

	return numpy.asarray([ret])

def deconv2d(i, W, shape, stride):
	return tf.nn.conv2d_transpose(i, W, shape, [1, stride, stride, 1], padding = "SAME")

def conv2d(i, W, stride):
	return tf.nn.conv2d(i, W, strides = [1, stride, stride, 1], padding = "SAME")

def createVar(shape, validate=True):
	return tf.Variable(tf.truncated_normal(shape, stddev = weight_sigma), validate_shape=validate)

def createConvImageSummary(W_conv, conv_size, name, abs, absasdasd,sa, o):
	ret = tf.reshape(W_conv, [conv_size, conv_size, 1, -1])
	ret = tf.transpose(ret, [3, 0, 1, 2])
	x_min = tf.reduce_min(ret)
	x_max = tf.reduce_max(ret)
	ret = (ret - x_min) / (x_max - x_min)
	return tf.summary.image(name, ret, 4096)

def createNetwork():
	num_conv_filter_1 = 128
	num_conv_filter_2 = 128
	num_conv_filter_3 = 128
	num_conv_filter_4 = 128
	num_conv_filter_5 = 128
	num_conv_filter_6 = 128
	num_conv_filter_7 = 128
	num_conv_filter_8 = 128
	num_conv_filter_9 = 256
	num_conv_filter_10 = 256
	num_conv_filter_11 = 256
	num_conv_filter_12 = 256

	num_conv_fc1 = 2048
	num_conv_fc2 = 2048
	num_middle = 48
	num_deconv_fc1 = 70
	num_deconv_filter_1 = 256
	num_deconv_filter_2 = 256
	num_deconv_filter_3 = 256
	num_deconv_filter_4 = 256
	num_deconv_filter_5 = 256
	num_deconv_filter_6 = 256
	num_deconv_filter_7 = 256

	conv_size1 = 3
	conv_size2 = 3
	conv_size3 = 3
	conv_size4 = 3
	conv_size5 = 3
	conv_size6 = 3
	conv_size7 = 3
	conv_size8 = 3
	conv_size9 = 3
	conv_size10 = 3
	conv_size11 = 3
	conv_size12 = 3

	deconv_size1 = 3
	deconv_size2 = 3
	deconv_size3 = 3
	deconv_size4 = 3
	deconv_size5 = 3
	deconv_size6 = 3
	deconv_size7 = 3
	deconv_size8 = 3

	W_conv1 = createVar([conv_size1, conv_size1, 1, num_conv_filter_1])
	b_conv1 = createVar([num_conv_filter_1])

	W_conv2 = createVar([conv_size2, conv_size2, num_conv_filter_1, num_conv_filter_2])
	b_conv2 = createVar([num_conv_filter_2])

	W_conv3 = createVar([conv_size3, conv_size3, num_conv_filter_2, num_conv_filter_3])
	b_conv3 = createVar([num_conv_filter_3])

	W_conv4 = createVar([conv_size4, conv_size4, num_conv_filter_3, num_conv_filter_4])
	b_conv4 = createVar([num_conv_filter_4])

	W_conv5 = createVar([conv_size5, conv_size5, num_conv_filter_4, num_conv_filter_5])
	b_conv5 = createVar([num_conv_filter_5])

	W_conv6 = createVar([conv_size6, conv_size6, num_conv_filter_5, num_conv_filter_6])
	b_conv6 = createVar([num_conv_filter_6])

	W_conv7 = createVar([conv_size7, conv_size7, num_conv_filter_6, num_conv_filter_7])
	b_conv7 = createVar([num_conv_filter_7])

	W_conv8 = createVar([conv_size8, conv_size8, num_conv_filter_7, num_conv_filter_8])
	b_conv8 = createVar([num_conv_filter_8])

	W_conv9 = createVar([conv_size9, conv_size9, num_conv_filter_8, num_conv_filter_9])
	b_conv9 = createVar([num_conv_filter_9])

	W_conv10 = createVar([conv_size10, conv_size10, num_conv_filter_9, num_conv_filter_10])
	b_conv10 = createVar([num_conv_filter_10])

	W_conv11 = createVar([conv_size11, conv_size11, num_conv_filter_10, num_conv_filter_11])
	b_conv11 = createVar([num_conv_filter_11])

	W_conv12 = createVar([conv_size12, conv_size12, num_conv_filter_11, num_conv_filter_12])
	b_conv12 = createVar([num_conv_filter_12])

	obs = tf.placeholder(tf.float32, [SCREEN_Y, SCREEN_X, 3])
	obs_norm = obs / 255.0
	obs_norm_grey = tf.image.rgb_to_grayscale(obs_norm)
	obs_4d = tf.reshape(obs_norm_grey, [1, SCREEN_Y, SCREEN_X, 1])

	h_conv1 = tf.nn.elu(conv2d(obs_4d, W_conv1, 1) + b_conv1)
	h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2, 1) + b_conv2)
	h_conv3 = tf.nn.elu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
	h_conv4 = tf.nn.elu(conv2d(h_conv3, W_conv4, 1) + b_conv4)
	h_conv5 = tf.nn.elu(conv2d(h_conv4, W_conv5, 2) + b_conv5)
	h_conv6 = tf.nn.elu(conv2d(h_conv5, W_conv6, 1) + b_conv6)
	h_conv7 = tf.nn.elu(conv2d(h_conv6, W_conv7, 2) + b_conv7)
	h_conv8 = tf.nn.elu(conv2d(h_conv7, W_conv8, 1) + b_conv8)
	h_conv9 = tf.nn.elu(conv2d(h_conv8, W_conv9, 2) + b_conv9)
	h_conv10 = tf.nn.elu(conv2d(h_conv9, W_conv10, 1) + b_conv10)
	h_conv11 = tf.nn.elu(conv2d(h_conv10, W_conv11, 2) + b_conv11)
	h_conv12 = tf.nn.elu(conv2d(h_conv11, W_conv12, 1) + b_conv12)

	# Should be 70 long
	conv_res = tf.reshape(h_conv12, [1, 7*10*num_conv_filter_12])

	W_conv_fc1 = createVar([7*10*num_conv_filter_12, num_conv_fc1])
	b_conv_fc1 = createVar([num_conv_fc1])
	W_conv_fc2 = createVar([num_conv_fc1, num_conv_fc2])
	b_conv_fc2 = createVar([num_conv_fc2])

	h_conv_fc1 = tf.nn.elu(tf.matmul(conv_res, W_conv_fc1) + b_conv_fc1)
	h_conv_fc2 = tf.nn.elu(tf.matmul(h_conv_fc1, W_conv_fc2) + b_conv_fc2)

	W_middle = createVar([num_conv_fc2, num_middle])
	b_middle = createVar([num_middle])

	h_middle = tf.nn.elu(tf.matmul(h_conv_fc2, W_middle) + b_middle)

	W_deconv_fc1 = createVar([num_middle, num_deconv_fc1])
	b_deconv_fc1 = createVar([num_deconv_fc1])
	W_deconv_fc2 = createVar([num_deconv_fc1, 70])
	b_deconv_fc2 = createVar([70])

	h_deconv_fc1 = tf.nn.elu(tf.matmul(h_middle, W_deconv_fc1) + b_deconv_fc1)
	h_deconv_fc2 = tf.nn.elu(tf.matmul(h_deconv_fc1, W_deconv_fc2) + b_deconv_fc2)

	h_deconv_fc2_4d = tf.reshape(h_deconv_fc2, [1, 7, 10, 1])

	W_deconv1 = createVar([deconv_size1, deconv_size1, \
		num_deconv_filter_1, 1])
	b_deconv1 = createVar([num_deconv_filter_1])

	W_deconv2 = createVar([deconv_size2, deconv_size2, \
		num_deconv_filter_2, num_deconv_filter_1])
	b_deconv2 = createVar([num_deconv_filter_2])

	W_deconv3 = createVar([deconv_size3, deconv_size3, \
		num_deconv_filter_3, num_deconv_filter_2])
	b_deconv3 = createVar([num_deconv_filter_3])

	W_deconv4 = createVar([deconv_size4, deconv_size4, \
		num_deconv_filter_4, num_deconv_filter_3])
	b_deconv4 = createVar([num_deconv_filter_4])

	W_deconv5 = createVar([deconv_size5, deconv_size5, \
		num_deconv_filter_5, num_deconv_filter_4])
	b_deconv5 = createVar([num_deconv_filter_5])

	W_deconv6 = createVar([deconv_size6, deconv_size6, \
		num_deconv_filter_6, num_deconv_filter_5])
	b_deconv6 = createVar([num_deconv_filter_6])

	W_deconv7 = createVar([deconv_size7, deconv_size7, \
		num_deconv_filter_7, num_deconv_filter_6])
	b_deconv7 = createVar([num_deconv_filter_7])

	W_deconv8 = createVar([deconv_size8, deconv_size8, \
		16, num_deconv_filter_7])
	b_deconv8 = createVar([16])

	sy = int(SCREEN_Y / 32)
	sx = int(SCREEN_X / 32) 
	h_deconv1 = tf.nn.elu(deconv2d(h_deconv_fc2_4d, W_deconv1, \
		[1, sy * 1, sx * 1, num_deconv_filter_1], 1) + b_deconv1)
	h_deconv2 = tf.nn.elu(deconv2d(h_deconv1, W_deconv2, \
		[1, sy * 1, sx * 1, num_deconv_filter_2], 1) + b_deconv2)
	h_deconv3 = tf.nn.elu(deconv2d(h_deconv2, W_deconv3, \
		[1, sy * 2, sx * 2, num_deconv_filter_3], 2) + b_deconv3)
	h_deconv4 = tf.nn.elu(deconv2d(h_deconv3, W_deconv4, \
		[1, sy * 2, sx * 2, num_deconv_filter_4], 1) + b_deconv4)
	h_deconv5 = tf.nn.elu(deconv2d(h_deconv4, W_deconv5, \
		[1, sy * 4, sx * 4, num_deconv_filter_5], 2) + b_deconv5)
	h_deconv6 = tf.nn.elu(deconv2d(h_deconv5, W_deconv6, \
		[1, sy * 4, sx * 4, num_deconv_filter_6], 1) + b_deconv6)
	h_deconv7 = tf.nn.elu(deconv2d(h_deconv6, W_deconv7, \
		[1, sy * 8, sx * 8, num_deconv_filter_7], 2) + b_deconv7)
	h_deconv8 = tf.nn.elu(deconv2d(h_deconv7, W_deconv8, \
		[1, sy * 8, sx * 8, 16], 1) + b_deconv8)
	output_image = tf.reshape(h_deconv8, [1, SCREEN_Y, SCREEN_X, 1])

	h_middle_1d = tf.reshape(h_middle, [-1])
	z_mean, z_variance = tf.nn.moments(h_middle_1d, [0])
	z_stddev = tf.sqrt(z_variance)
	latent_loss = 0.5 * tf.square(z_mean) + \
		tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1
	loss = tf.reduce_mean(tf.square(output_image - obs_4d)) + latent_loss

	optimizer = tf.train.AdamOptimizer(learning_rate=adam_alpha)
	train = optimizer.minimize(loss)

	sw_l1 = createConvImageSummary(W_conv1, conv_size1, "Conv Layer 1")
	sw_l2 = createConvImageSummary(W_conv2, conv_size2, "Conv Layer 2")
	sw_l3 = createConvImageSummary(W_conv3, conv_size3, "Conv Layer 3")
	sw_l4 = createConvImageSummary(W_conv4, conv_size4, "Conv Layer 4")
	sw_l5 = createConvImageSummary(W_conv5, conv_size5, "Conv Layer 5")
	sw_l6 = createConvImageSummary(W_conv6, conv_size6, "Conv Layer 6")
	sw_l7 = createConvImageSummary(W_conv7, conv_size7, "Conv Layer 7")
	sw_l8 = createConvImageSummary(W_conv8, conv_size8, "Conv Layer 8")
	sw_l9 = createConvImageSummary(W_conv9, conv_size9, "Conv Layer 9")
	sw_l10 = createConvImageSummary(W_conv10, conv_size10, "Conv Layer 10")
	sw_l11 = createConvImageSummary(W_conv11, conv_size11, "Conv Layer 11")
	sw_l12 = createConvImageSummary(W_conv12, conv_size12, "Conv Layer 12")

	sw_de_l1 = createConvImageSummary(W_deconv1, deconv_size1, "Deconv Layer 1")
	sw_de_l2 = createConvImageSummary(W_deconv2, deconv_size2, "Deconv Layer 2")
	sw_de_l3 = createConvImageSummary(W_deconv3, deconv_size3, "Deconv Layer 3")
	sw_de_l4 = createConvImageSummary(W_deconv4, deconv_size4, "Deconv Layer 4")
	sw_de_l5 = createConvImageSummary(W_deconv5, deconv_size5, "Deconv Layer 5")
	sw_de_l6 = createConvImageSummary(W_deconv6, deconv_size6, "Deconv Layer 6")
	sw_de_l7 = createConvImageSummary(W_deconv7, deconv_size7, "Deconv Layer 7")
	sw_de_l8 = createConvImageSummary(W_deconv8, deconv_size8, "Deconv Layer 8")

	h_W_conv1 = tf.summary.histogram("W_conv1", W_conv1)
	h_b_conv1 = tf.summary.histogram("b_conv1", b_conv1)
	h_W_conv2 = tf.summary.histogram("W_conv2", W_conv2)
	h_b_conv2 = tf.summary.histogram("b_conv2", b_conv2)
	h_W_conv3 = tf.summary.histogram("W_conv3", W_conv3)
	h_b_conv3 = tf.summary.histogram("b_conv3", b_conv3)
	h_W_conv4 = tf.summary.histogram("W_conv4", W_conv4)
	h_b_conv4 = tf.summary.histogram("b_conv4", b_conv4)
	h_W_conv5 = tf.summary.histogram("W_conv5", W_conv5)
	h_b_conv5 = tf.summary.histogram("b_conv5", b_conv5)
	h_W_conv6 = tf.summary.histogram("W_conv6", W_conv6)
	h_b_conv6 = tf.summary.histogram("b_conv6", b_conv6)
	h_W_conv7 = tf.summary.histogram("W_conv7", W_conv7)
	h_b_conv7 = tf.summary.histogram("b_conv7", b_conv7)
	h_W_conv8 = tf.summary.histogram("W_conv8", W_conv8)
	h_b_conv8 = tf.summary.histogram("b_conv8", b_conv8)
	h_W_conv9 = tf.summary.histogram("W_conv9", W_conv9)
	h_b_conv9 = tf.summary.histogram("b_conv9", b_conv9)
	h_W_conv10 = tf.summary.histogram("W_conv10", W_conv10)
	h_b_conv10 = tf.summary.histogram("b_conv10", b_conv10)
	h_W_conv11 = tf.summary.histogram("W_conv11", W_conv11)
	h_b_conv11 = tf.summary.histogram("b_conv11", b_conv11)
	h_W_conv12 = tf.summary.histogram("W_conv12", W_conv12)
	h_b_conv12 = tf.summary.histogram("b_conv12", b_conv12)

	h_W_conv_fc1 = tf.summary.histogram("FC1 Weight (Conv)", W_conv_fc1)
	h_b_conv_fc1 = tf.summary.histogram("FC1 Bias (Conv)", b_conv_fc1)
	h_W_conv_fc2 = tf.summary.histogram("FC2 Weight (Conv)", W_conv_fc2)
	h_b_conv_fc2 = tf.summary.histogram("FC2 Bias (Conv)", b_conv_fc2)
	h_W_middle = tf.summary.histogram("W_middle", W_middle)
	h_b_middle = tf.summary.histogram("b_middle", b_middle)
	h_W_deconv_fc1 = tf.summary.histogram("FC1 Weight (Deconv)", W_deconv_fc1)
	h_b_deconv_fc1 = tf.summary.histogram("FC1 Bias (Deconv)", b_deconv_fc1)
	h_W_deconv_fc2 = tf.summary.histogram("FC2 Weight (Deconv)", W_deconv_fc2)
	h_b_deconv_fc2 = tf.summary.histogram("FC2 Bias (Deconv)", b_deconv_fc2)

	h_W_deconv1 = tf.summary.histogram("W_deconv1", W_deconv1)
	h_b_deconv1 = tf.summary.histogram("b_deconv1", b_deconv1)
	h_W_deconv2 = tf.summary.histogram("W_deconv2", W_deconv2)
	h_b_deconv2 = tf.summary.histogram("b_deconv2", b_deconv2)
	h_W_deconv3 = tf.summary.histogram("W_deconv3", W_deconv3)
	h_b_deconv3 = tf.summary.histogram("b_deconv3", b_deconv3)
	h_W_deconv4 = tf.summary.histogram("W_deconv4", W_deconv4)
	h_b_deconv4 = tf.summary.histogram("b_deconv4", b_deconv4)
	h_W_deconv5 = tf.summary.histogram("W_deconv5", W_deconv5)
	h_b_deconv5 = tf.summary.histogram("b_deconv5", b_deconv5)
	h_W_deconv6 = tf.summary.histogram("W_deconv6", W_deconv6)
	h_b_deconv6 = tf.summary.histogram("b_deconv6", b_deconv6)
	h_W_deconv7 = tf.summary.histogram("W_deconv7", W_deconv7)
	h_b_deconv7 = tf.summary.histogram("b_deconv7", b_deconv7)
	h_W_deconv8 = tf.summary.histogram("W_deconv8", W_deconv8)
	h_b_deconv8 = tf.summary.histogram("b_deconv8", b_deconv8)

	return {
		'obs' : obs,
		'obs_norm' : obs_norm,
		'obs_norm_grey' : obs_norm_grey,
		'obs_4d' : obs_4d,
		'loss' : loss,
		'optimizer' : optimizer,
		'train' : train,

		'output_image' : output_image,

		'h_W_conv1' : h_W_conv1,
		'h_b_conv1' : h_b_conv1,
		'h_W_conv2' : h_W_conv2,
		'h_b_conv2' : h_b_conv2,
		'h_W_conv3' : h_W_conv3,
		'h_b_conv3' : h_b_conv3,
		'h_W_conv4' : h_W_conv4,
		'h_b_conv4' : h_b_conv4,
		'h_W_conv5' : h_W_conv5,
		'h_b_conv5' : h_b_conv5,
		'h_W_conv6' : h_W_conv6,
		'h_b_conv6' : h_b_conv6,
		'h_W_conv7' : h_W_conv7,
		'h_b_conv7' : h_b_conv7,
		'h_W_conv8' : h_W_conv8,
		'h_b_conv8' : h_b_conv8,
		'h_W_conv9' : h_W_conv9,
		'h_b_conv9' : h_b_conv9,
		'h_W_conv10' : h_W_conv10,
		'h_b_conv10' : h_b_conv10,
		'h_W_conv11' : h_W_conv11,
		'h_b_conv11' : h_b_conv11,
		'h_W_conv12' : h_W_conv12,
		'h_b_conv12' : h_b_conv12,

		'h_W_conv_fc1' : h_W_conv_fc1,
		'h_b_conv_fc1' : h_b_conv_fc1,
		'h_W_conv_fc2' : h_W_conv_fc2,
		'h_b_conv_fc2' : h_b_conv_fc2,

		'h_W_middle' : h_W_middle,
		'h_b_middle' : h_b_middle,

		'h_W_deconv_fc1' : h_W_deconv_fc1,
		'h_b_deconv_fc1' : h_b_deconv_fc1,
		'h_W_deconv_fc2' : h_W_deconv_fc2,
		'h_b_deconv_fc2' : h_b_deconv_fc2,

		'h_W_deconv1' : h_W_deconv1,
		'h_b_deconv1' : h_b_deconv1,
		'h_W_deconv2' : h_W_deconv2,
		'h_b_deconv2' : h_b_deconv2,
		'h_W_deconv3' : h_W_deconv3,
		'h_b_deconv3' : h_b_deconv3,
		'h_W_deconv4' : h_W_deconv4,
		'h_b_deconv4' : h_b_deconv4,
		'h_W_deconv5' : h_W_deconv5,
		'h_b_deconv5' : h_b_deconv5,
		'h_W_deconv6' : h_W_deconv6,
		'h_b_deconv6' : h_b_deconv6,
		'h_W_deconv7' : h_W_deconv7,
		'h_b_deconv7' : h_b_deconv7,
		'h_W_deconv8' : h_W_deconv8,
		'h_b_deconv8' : h_b_deconv8,

		'sw_l1' : sw_l1,
		'sw_l2' : sw_l2,
		'sw_l3' : sw_l3,
		'sw_l4' : sw_l4,
		'sw_l5' : sw_l5,
		'sw_l6' : sw_l6,
		'sw_l7' : sw_l7,
		'sw_l8' : sw_l8,
		'sw_l9' : sw_l9,
		'sw_l10' : sw_l10,
		'sw_l11' : sw_l11,
		'sw_l12' : sw_l12,

		'sw_de_l1' : sw_de_l1,
		'sw_de_l2' : sw_de_l2,
		'sw_de_l3' : sw_de_l3,
		'sw_de_l4' : sw_de_l4,
		'sw_de_l5' : sw_de_l5,
		'sw_de_l6' : sw_de_l6,
		'sw_de_l7' : sw_de_l7,
		'sw_de_l8' : sw_de_l8,

		}

def getLevel():
	with open("sonic-train.csv") as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			games = []
			acts = []
			for row in readCSV:
				games.append(row[0])
				acts.append(row[1])
	randIndex = random.randint(1, len(acts)-1)
	return (games[randIndex], acts[randIndex])  

def getInput(n):
	'''
	0  1   2      3     4    5    6      7    8  9  10 11
	B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z
	{{}, {LEFT}, {RIGHT}, {LEFT, DOWN}, {RIGHT, DOWN}, {DOWN}, {DOWN, B}, {B}}
	'''
	ret = numpy.zeros((12,), numpy.int8)
	
	if n == 1:
		ret[6] = 1
		return ret
	
	elif n == 2:
		ret[7] = 1
		return ret

	elif n == 3:
		ret[5] = 1
		ret[6] = 1
		return ret

	elif n == 4:
		ret[5] = 1
		ret[7] = 1
		return ret

	elif n == 5:
		ret[5] = 1
		return ret  

	elif n == 6:
		ret[0] = 1
		ret[5] = 1
		return ret  

	elif n == 7:
		ret[0] = 1
		return ret  

	return ret

def resetEnv(env):
	obs = env.reset()
	obs, rew, done, info = env.step(getInput(0))

	return obs

def stepEnv(env, render=False):
	input_out = getInput(random.randint(0, NUM_ACTION - 1))
	obs, rew, done, info = env.step(input_out)
	if render:
		env.render()
	return obs, done

def createSession():
	sessconfig = tf.ConfigProto()
	sessconfig.gpu_options.allow_growth = True

	out = createNetwork()

	sess = tf.Session(config = sessconfig)
	sess.run(tf.global_variables_initializer())

	return sess

def getSession():
	sessconfig = tf.ConfigProto()
	sessconfig.gpu_options.allow_growth = True
	sess = tf.Session(config = sessconfig)
	out = createNetwork()

	files = getFilesInDir("data")
	if len(files) < 2:
		print("VAE not found... creating one!")
		sess.run(tf.global_variables_initializer())
		saveSession(sess)
	else:
		saver = tf.train.Saver()
		saver.restore(sess, "data/sonic_" + log_name + "_nn")

	return sess, out

def saveSession(sess):
	saver = tf.train.Saver()
	saver.save(sess, "data/sonic_" + log_name + "_nn")

def getStepSaveNumber():
	filename = log_name + "_logs/stepnum.txt"
	num = 1
	if os.path.isfile(filename):
		with open(filename, "r") as f:
			l = f.readlines()
			val = int(l[0])
			if val > num:
				num = val
			
	with open(filename, "w") as f:
		f.write(str(num+1))

	return num

def saveStuff(sess, net):
	gs = getStepSaveNumber()

	print("Saving TensorBoard...")
	sw = tf.summary.FileWriter(log_name + "_logs", sess.graph)

	sw.add_summary(net['sw_l1'].eval(session=sess), gs)
	sw.add_summary(net['sw_l2'].eval(session=sess), gs)
	sw.add_summary(net['sw_l3'].eval(session=sess), gs)
	sw.add_summary(net['sw_l4'].eval(session=sess), gs)
	sw.add_summary(net['sw_l5'].eval(session=sess), gs)
	sw.add_summary(net['sw_l6'].eval(session=sess), gs)
	sw.add_summary(net['sw_l7'].eval(session=sess), gs)
	sw.add_summary(net['sw_l8'].eval(session=sess), gs)
	sw.add_summary(net['sw_l9'].eval(session=sess), gs)
	sw.add_summary(net['sw_l10'].eval(session=sess), gs)
	sw.add_summary(net['sw_l11'].eval(session=sess), gs)
	sw.add_summary(net['sw_l12'].eval(session=sess), gs)

	sw.add_summary(net['sw_de_l1'].eval(session=sess), gs)
	sw.add_summary(net['sw_de_l2'].eval(session=sess), gs)
	sw.add_summary(net['sw_de_l3'].eval(session=sess), gs)
	sw.add_summary(net['sw_de_l4'].eval(session=sess), gs)
	sw.add_summary(net['sw_de_l5'].eval(session=sess), gs)
	sw.add_summary(net['sw_de_l6'].eval(session=sess), gs)
	sw.add_summary(net['sw_de_l7'].eval(session=sess), gs)
	sw.add_summary(net['sw_de_l8'].eval(session=sess), gs)

	sw.add_summary(net['h_W_conv1'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv1'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv2'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv2'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv3'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv3'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv4'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv4'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv5'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv5'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv6'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv6'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv7'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv7'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv8'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv8'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv9'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv9'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv10'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv10'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv11'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv11'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv12'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv12'].eval(session=sess), gs)

	sw.add_summary(net['h_W_conv_fc1'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv_fc1'].eval(session=sess), gs)
	sw.add_summary(net['h_W_conv_fc2'].eval(session=sess), gs)
	sw.add_summary(net['h_b_conv_fc2'].eval(session=sess), gs)

	sw.add_summary(net['h_W_middle'].eval(session=sess), gs)
	sw.add_summary(net['h_b_middle'].eval(session=sess), gs)

	sw.add_summary(net['h_W_deconv_fc1'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv_fc1'].eval(session=sess), gs)
	sw.add_summary(net['h_W_deconv_fc2'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv_fc2'].eval(session=sess), gs)

	sw.add_summary(net['h_W_deconv1'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv1'].eval(session=sess), gs)
	sw.add_summary(net['h_W_deconv2'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv2'].eval(session=sess), gs)
	sw.add_summary(net['h_W_deconv3'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv3'].eval(session=sess), gs)
	sw.add_summary(net['h_W_deconv4'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv4'].eval(session=sess), gs)
	sw.add_summary(net['h_W_deconv5'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv5'].eval(session=sess), gs)
	sw.add_summary(net['h_W_deconv6'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv6'].eval(session=sess), gs)
	sw.add_summary(net['h_W_deconv7'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv7'].eval(session=sess), gs)
	sw.add_summary(net['h_W_deconv8'].eval(session=sess), gs)
	sw.add_summary(net['h_b_deconv8'].eval(session=sess), gs)

	sw.flush()
	sw.close()
	print("Done")

def saveInterpretedImage(sess, net, obs):

	im, orig = sess.run([ net['output_image'], net['obs_norm_grey'] ], \
		feed_dict = { net['obs'] : obs })
	im = im[0]

	newim = []
	for y in range(len(im)):
		temp = []
		for x in range(len(im[y])):
			val = min(max(int(im[y][x][0] * 255.0), 0), 255)
			temp.append(val)
			# for i in range(len(im[y][x])):
			#   im[y][x][i] = min(max(int(im[y][x][i] * 255.0), 0), 255)
		newim.append(temp)
	im = numpy.asarray(newim)
	im = Image.fromarray(im.astype(numpy.uint8), "L")

	num = 0
	ext = ".jpeg"
	dirname = "vae_screenshots/"
	filename = log_name + "_" + str(num) + ext

	while os.path.isfile(dirname + filename):
		num = num + 1
		filename = log_name + "_" + str(num) + ext

	im.save(dirname + filename)

	newim = []
	for y in range(len(orig)):
		temp = []
		for x in range(len(orig[y])):
			val = min(max(int(orig[y][x][0] * 255.0), 0), 255)
			temp.append(val)
		newim.append(temp)
	im = numpy.asarray(newim)
	im = Image.fromarray(im.astype(numpy.uint8), "L")
	im.save(dirname + log_name + "_orig_" + str(num) + ext)

def main():

	tf.reset_default_graph()
	sess, net = getSession()

	print("\nBeginning VAE...\n")

	num_loops = 0
	while True:
		game, act = getLevel()
		env = make(game=game,state=act)
		obs = resetEnv(env)

		visionMem = []
		time_step = 0
		space = False
		done = False

		print("Watching...")
		while True:
			while (msvcrt.kbhit()):
				res = msvcrt.getch()
				if res == b' ':
					space = True
			
			for i in range(4):
				obs, done = stepEnv(env, render=render)
				if done:
					break

			visionMem.append(obs)
			time_step += 1

			if space:
				space = False
				saveInterpretedImage(sess, net, obs)
				saveStuff(sess, net)

			if done:
				print("Finished Level...")
				env.close()
				if time_step > threshold:
					break
				
				game, act = getLevel()
				env = make(game=game,state=act)
				obs = resetEnv(env)

		print("Done Playing (%d)" % num_loops)

		replayLen = len(visionMem)
		replay = random.sample(visionMem, replayLen)

		count = 0
		total_loss = 0.0
		print("\n")
		for obs in replay:
			t, curloss = sess.run([net['train'], net['loss']], \
				feed_dict = { net['obs'] : obs })

			count += 1
			print ("\r%04d out of %04d, loss: %06f" % \
				(count, replayLen, curloss), end = '\r')
			total_loss += curloss

		print("\nSaving...")
		saveSession(sess)
		with open(log_name + "_loss_history.txt", "a") as f:
			f.write( "%lf\n" % (total_loss / float(replayLen)) )
		print("Finished training")

		num_loops += 1
		if num_loops == save_interval:
			saveInterpretedImage(sess, net, obs)
			saveStuff(sess, net)
			num_loops = 0

def getFilesInDir(dir):
	return [f for f in listdir(dir) if os.path.isfile(os.path.join(dir, f))]
		
if __name__ == '__main__':
	main()
