from skimage.measure import block_reduce
import gym
import math
import numpy as np
import tensorflow as tf
from skimage.measure import block_reduce
from tensorflow.python import debug as tf_debug

class deep_q_network:

	sess = tf.Session() #needed to prevent core dump with GPU tensorflow, probably a bug
	sess = tf_debug.LocalCLIDebugWrapperSession(sess)

	def __init__(self,
              replayCache,
              batchNum,
              actionRange,
              observationSpace,
              model_directory,
              env,
              learning_rate):

     		 #Hyperparameters
		self.replayCache = replayCache #number of events we will hold in memory
		self.batchNum = batchNum #number of experiences we will replay at each time step
		self.actionRange = actionRange  #number of possible actions at each time-step
		self.env = env #openAI gym environment
		self.learning_rate = learning_rate #gradient descent step size
		 #Set up logging for predictions
		tensors_to_log = {"Q-Values": 'logits/kernel'}
		self.logging_hook = tf.train.LoggingTensorHook(
     				tensors=tensors_to_log, every_n_iter=50)
		 #Variables
		self.D = [] # list that will hold the replay cache
		self.actNum = [1 for x in range(self.actionRange)] #list meant to
			# hold the number of times a given action has been taken
		 #Create Q-network object
		self.net = tf.estimator.Estimator(
      			model_fn=Q_Net,
			model_dir=model_directory, #where the models are
							# stored and loaded from
			params = {'actionRange':self.actionRange,
				'observationSpace':observationSpace})

	def run_and_train(self, episodes, scores_file, initialize_net=False):

		 #Start episode loop with step by step loop inside
		for i_episode in range(episodes):
			F = open(scores_file, 'a') #file to keep track of scores
			preObs = self.env.reset() #observation at state s_t
			 #Convert to grayscale with average converson and downsample
			preObs = self.Preprocess(preObs, [], True)
			t = 1
			score = 0

        	 	 #initialize the observations so we have a block of 4 frames
			for i in range(3):
#				self.env.render()
				obs, reward, done, info = self.env.step(self.env.action_space.sample())
				score += reward
				preObs = self.Preprocess(obs, preObs, True)

			while 1:
#				self.env.render()

				 #Predict Q-Values of actions
				action = 0
				qVals = []

				if initialize_net and t == 2:
					action, qVals = self.Predict(t, preObs, qVals, initialize_net)
					initialize_net = False
				else:
					action, qVals = self.Predict(t, preObs, qVals)

				 #Take action, observe environment and other info
				done, info, score, postObs = self.action_step(t,
								preObs,
        							action,
        							qVals,
        							score)

				 #Train net
				self.train_net()

				 #Transition into the next state
				preObs = postObs
				t += 1

				if done: #end episode if done flag is set to true
					print("Episode ", i_episode," finished")
					print('Score ', score)
					line = 'Score for episode ' + str(i_episode) + ": " + str(score)
					F.write(line)
					F.close()
					break

	def Predict(self, t, X, qVals, initialize_net=False):

		if initialize_net == False and t > 2:
			input_fn = tf.estimator.inputs.numpy_input_fn(
		                x = {'X':np.asarray([X])},
		                batch_size=1,
		                shuffle = False)
			qVals = self.net.predict(
		                input_fn=input_fn)
			qVals = list(qVals)[0]['Q-Values']
			print(qVals) #!!!for debugging!!!
		elif initialize_net == True or t == 1:
			qVals = np.random.rand(self.actionRange)
			print(qVals)

    		 #Upper confidence bound adjustment of the Q-Values
		for a in range(self.actionRange):
        		qVals[a] = qVals[a] + ((2 * math.log(t,10)) / self.actNum[a])**.5
		action = np.argmax(qVals)

		return action, qVals

	def action_step(self, t, preObs, action, qVals, score):

		postObs, reward, done, info = self.env.step(action)
		self.actNum[action] += 1
		postObs = self.Preprocess(postObs, preObs)
		score += reward

		 #Storeing observations action-value and reward into
		 # the replay cache and replacing old values
		if len(self.D) < self.replayCache:
			self.D.append({"preObs":np.asarray([preObs]),
					'reward':np.asarray([reward]),
					'post_max_q':[]})
			if t != 1: self.D[len(self.D) - 2]['post_max_q'] = np.asarray([qVals[action]])
		elif len(self.D) == self.replayCache:
			span = t // self.replayCache #The number of times we've completely replaced
                                       # the replay cache with new values
			i = (t-1) - (span * self.replayCache) # the current oldest spot on the replay cache
			self.D[i] = {"preObs":np.asarray([preObs]),
                    		'reward':np.asarray([reward]),
                    		'post_max_q':[]}
			if i == 0: i = self.replayCache
			self.D[i-1]['post_max_q'] = np.asarray([qVals[action]])

		return done, info, score, postObs

	def train_net(self, initialize_net=False):

		 #Update neural net using random values from the cache
		rand = []
		if len(self.D) - 1 < self.batchNum:
			rand = np.arange(len(self.D) - 1)
			np.random.shuffle(rand)
		elif len(self.D) - 1 >= self.batchNum:
			rand = np.arange(len(self.D) - 1)
			np.random.shuffle(rand)
			rand = rand[:self.batchNum]

		for i in rand:
			if len(self.D) == 2:
				labels = np.asarray([-1])
			else:
				labels = self.D[i]['post_max_q']
			#Use predicted values to calculate loss and perform
			# a gradient descent step
			train_input_fn = tf.estimator.inputs.numpy_input_fn(
						x = {'X':self.D[i]['preObs'],
						    't':np.asarray([len(self.D)]),
          					    'learning_rate':np.asarray([self.learning_rate]),
						    'reward':self.D[i]['reward']},
						y = labels,
						batch_size = 1,
						shuffle = False)
			self.net.train(
				input_fn = train_input_fn,
				steps = 1,
				hooks = [self.logging_hook])

	 #Function to downsample and grayscale our images
	def Preprocess(self, new_frame, old_frames, initialize=False):

		order = [3,0,1,2] #Order to change frames after appending new frame

		if initialize == False:
			new_frame = np.average(new_frame, axis=2) #turn to grayscale
			new_frame = block_reduce(new_frame, block_size=(2,2), func=np.average) #downsample
			old_frames[3] = new_frame   #replace oldest frame
			new_frames = [old_frames[x] for x in order]
			return new_frames

		if initialize == True:
			new_frame = np.average(new_frame, axis=2) #turn to grayscale
			new_frame = block_reduce(new_frame, block_size=(2,2), func=np.average) #downsample
			old_frames.append(new_frame)

			if len(old_frames) == 4:
				new_frames = [old_frames[x] for x in range(3,-1,-1)]
				return new_frames
			else: return old_frames

	#Now create the Neural net function with Tensorflow
def Q_Net(features, labels, mode, params):

	 #Define all of the layers.
	 #Two convolutional layers with relu activation,
	 # a dense layer, dropout, and another dense layer

	 #Sizes of observation spaces after downsampling
	length = (params['observationSpace'][0]) // 2 + (params['observationSpace'][0] % 2)
	width = (params['observationSpace'][1]) // 2 + (params['observationSpace'][0] % 2)

	 #Layers
	input_layer = tf.reshape(features['X'], [-1,4,length,width,1])
	input_layer = tf.cast(input_layer, tf.float32) #Must match the float32 size of filters in conv3d
	conv1 = tf.layers.conv3d(
	inputs = input_layer,
	filters = 16,
	kernel_size = [1,8,8],
	strides = (1,4,4),
	activation = tf.nn.relu)
	conv2 = tf.layers.conv3d(
	inputs = conv1,
	filters = 32,
	kernel_size = [1,4,4],
	strides = (1,2,2),
	activation = tf.nn.relu)
	length = (length - 12) // 8	#length and width
	width = (width - 12) // 8	# of filters
	conv2_flat = tf.reshape(conv2, [1, length*width*4*32])
	dense = tf.layers.dense(
	inputs = conv2_flat,
	units = 256,
	name = 'dense',
	activation = tf.nn.relu)
#  	dropout = tf.layers.dropout(
#	        inputs = dense,
#		rate = .4,
#		training = mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(
	inputs = dense,
	units = params['actionRange'],
	name = 'logits')

	 #Store predicitons and exit if mode is predict
	prediction = {'Q-Values':logits}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(
			mode = mode,
			predictions = prediction)

	 #If we made it to here, proceed to calculate loss and make
	 # gradient descent step
	if labels[0] == -1: #set loss to 0 because we havent calculated any Q-Vals yet
     		loss = tf.losses.absolute_difference(tf.reduce_max([logits]),
                                            tf.reduce_max([logits]))
	else:
		loss = tf.losses.absolute_difference(
			tf.add(tf.cast(features['reward'][0],
				tf.float32), tf.cast(labels[0], tf.float32)),
			tf.reduce_max([logits]))
		loss = tf.square(loss)

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(
				learning_rate = features['learning_rate'][0])
		train_op = optimizer.minimize(
				loss = loss,
				global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(
				mode = mode,
				loss = loss,
				train_op = train_op)

	#If we get here, that means we were not in either of the
	# two modes and we should throw an error
	raise NameError('The Neural Net was not called in either predict or train modes')



def main():
     #hyperparameters
    replayCache = 100 #number of events we will hold in memory
    batchNum = 32 #number of experiences we will replay at each time step
    actionRange = 18 #number of possible actions at each time-step
                    # for the given environment
    observationSpace = (250, 160, 3) #observation space for the given environment
    model_dir = '/home/porrster/Documents/atari_ai/model_dir'
    env = gym.make("Centipede-v0")
    learning_rate = .001

    network = deep_q_network(replayCache,
                             batchNum,
                             actionRange,
                             observationSpace,
                             model_dir,
                             env,
                             learning_rate)
    network.run_and_train(episodes=50, scores_file='Scores.txt', initialize_net=True)

main()


