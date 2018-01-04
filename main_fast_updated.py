from skimage.measure import block_reduce
import gym
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

class deep_q_network(replayCache,
        batchNum,
        actionRange,
        observationSpace,
        model_dir):
    sess = tf.Session() #needed to prevent core dump with GPU tensorflow, probably a bug
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #hyperparameters
    replayCache = 100 #number of events we will hold in memory
    batchNum = 32 #number of experiences we will replay at each time step
    actionRange = 18 #number of possible actions at each time-step
                    # for the given environment
    observationSpace = (250, 160, 3) #observation space for the given environment
    # Set up logging for predictions
    tensors_to_log = {"Q-Values": 'logits/kernel'} #the /kernel part is added by api
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    #File to keep track of scores
    F = open('Scores.txt', 'a')
    #variables
    D = [] # list that will hold the replay cache
    actNum = [1 for x in range(actionRange)] #list meant to hold the number of times a given action has been taken
                 # must start with one so that we dont have division by zero
    #create convolutional network object and make environment
    global net
    net = tf.estimator.Estimator(
            model_fn=Q_Net,
            model_dir='/home/w/wp/wporr/Atari_AI/model_dir',
            params = {'actionRange':actionRange,
                'observationSpace':observationSpace})
    env = gym.make("Centipede-v0")

    #Start episode loop with step by step loop inside
    for i_episode in range(50):
        #File to keep track of scores
        F = open('Scores.txt', 'a') 
        preObs = env.reset() #observation at state S_t
        #Convert to grayscale with average converson and downsample
        preObs = Preprocess(preObs, [], True)
        t = 1
        score = 0

        #Initialize the observations so we have a block of 4 frames
        for i in range(3):
            #env.render()
            obs, reward, done, info = env.step(0)
            score += reward
            preObs = Preprocess(obs, preObs, True)

        while 1:
            #Render environment to see visual progress
            #env.render()
            #Predict Q-Values of actions
            action = 0
            qVals = []
            if len(D) >= 1:
                input_fn = tf.estimator.inputs.numpy_input_fn(
                        x = {'X':np.asarray([preObs])},
                        batch_size=1,
                        shuffle = False)
                qVals = net.predict(
                        input_fn=input_fn)
                        #checkpoint_path='~/Douments/atari_ai/model_dir/model.ckpt')
                action, qVals = find_max(qVals, actNum, t) 
                
            elif len(D) == 0:
                action = np.argmax([np.random.rand(actionRange)])
            #Take action, observe environment and other info
            postObs, reward, done, info = env.step(action)
            actNum[action] += 1
            #Downsample postObs and adjust the
            # posObs to have the last 4 frames and add to score
            postObs = Preprocess(postObs, preObs)
            score += reward
            #Storeing observations action-value and reward into the replay cache and replacing old values
            if len(D) < replayCache:
                D.append({"preObs":np.asarray([preObs]),
                    'reward':np.asarray([reward]),
                    'post_max_q':[]})
                if t != 1: D[len(D) - 2]['post_max_q'] = np.asarray([qVals[action]])
            elif len(D) == replayCache:
                span = t // replayCache #The number of times we've completely replaced
                                       # the replay cache with new values
                i = (t-1) - (span * replayCache) # the current oldest spot on the replay cache
                D[i] = {"preObs":np.asarray([preObs]),
                    'reward':np.asarray([reward]),
                    'post_max_q':[]}
                if i == 0: i = replayCache
                D[i-1]['post_max_q'] = np.asarray([qVals[action]])

            #Train net
            if t == 1: 
                qVals = train_net(D, batchNum, logging_hook, actionRange)
                ignore_this, D[0]['post_max_q'] = find_max(qVals, actNum, t)
            else: train_net(D, batchNum, logging_hook, actionRange)
            #Transition into the next state
            preObs = postObs
            t += 1

            if t % 10 == 0:
                print(t)

            #End episode if done flag is set to true
            if done:
                print("Episode ", i_episode," finished")
                print('Score ', score)
                line = 'Score for episode ' + str(i_episode) + ": " + str(score)
                F.write(line)
                F.close()
                break

def find_max(qVals, actNum, t):

    qVals = list(qVals)[0]['Q-Values']
    print(qVals)
    # Upper confidence bound adjustment of the Q-Values
    for a in range(actionRange):
        qVals[a] = qVals[a] + ((2 * math.log(t,10)) / actNum[a])**.5
        #Select action based on highest adjusted Q-Value
    action = np.argmax(qVals)
    return action, qVals

def train_net(D, batchNum, logging_hook, actionRange):
    #Update neural net using random values from the cache
    rand = []
    if len(D) - 1 < batchNum:
        rand = np.arange(len(D) - 1)
        np.random.shuffle(rand)
    elif len(D) - 1 >= batchNum:
        rand = np.arange(len(D) - 1)
        np.random.shuffle(rand)
        rand = rand[:batchNum]

    for i in rand:
        labels = D[i]['post_max_q']
        #Use predicted values to calculate loss and perform
        # a gradient descent step
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x = {'X':D[i]['preObs'],
                    't':np.asarray([len(D)]),
                    'reward':D[i]['reward']},
                y = labels,
                batch_size = 1,
                shuffle = False)
        qVals = net.train(
                input_fn = train_input_fn,
                steps = 1,
                hooks = [logging_hook])
        if len(D) == 1: return qVals


#Function to downsample and grayscale our images
def Preprocess(new_frame, old_frames, initialize=False):
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
        else:
            return old_frames



#Now create the Neural net function with Tensorflow
def Q_Net(features, labels, mode, params):
    #Define all of the layers.
    #Two convolutional layers with relu activation,
    # a dense layer, dropout, and another dense layer

    #Sizes of observation spaces after downsampling
    length = (params['observationSpace'][0]) // 2 + (params['observationSpace'][0] % 2)
    width = (params['observationSpace'][1]) // 2 + (params['observationSpace'][0] % 2)
    #Weight initializer
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
    #in reshape, calculate length and width of filters
    length = (length - 12) // 8
    width = (width - 12) // 8
    conv2_flat = tf.reshape(conv2, [1, length*width*4*32])
    dense = tf.layers.dense(
            inputs = conv2_flat,
            units = 256,
            name = 'dense',
            activation = tf.nn.relu)
       # dropout = tf.layers.dropout(
   #         inputs = dense,
   #         rate = .4,
   #         training = mode == tf.estimator.ModeKeys.TRAIN)
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
    loss = tf.losses.absolute_difference(
            tf.add(tf.cast(features['reward'][0], tf.float32), tf.cast(labels[0], tf.float32)),
            tf.reduce_max([logits]))
    loss = tf.square(loss)
    if mode == tf.estimator.ModeKeys.TRAIN:
        if features['t'][0] == 1:
            return tf.estimator.EstimatorSpec(
                mode = mode,
                predictions = prediction)
        else:
            optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate = .001)
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
    model_dir = '/home/w/wp/wporr/Atari_AI/model_dir'
