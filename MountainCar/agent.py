from keras.layers import Input , Dense
from keras.models import Model
from keras.optimizers import Adam

class DQNAGENT:
	def __init__(self , action_dim , observation_dim):
		self.action_dim = action_dim
		self.observation_dim = observation_dim
		self.model = self.create_model()

	def create_model(self):
		state_input = Input(shape = (self.observation_dim))
		state_h1    = Dense(400 , activation = "relu")(state_input)
		state_h2    = Dense(300 , activation = "relu")(state_h1)
		output      = Dense(self.action_dim , activation = "linear")(state_h2)
		model       = Model(inputs = state_input , outputs = output)
		model.compile(loss = "mse" , optimizer = Adam(0.005))
		return model