#!/usr/bin/env python
# _*_ coding: utf-8 _*_

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

class WaveSimulator():

    def __init__(self,Features,Targets):
        """Initialize the class"""
        self.Features = Features
        self.Targets = Targets

        return

    def fit_model(self):
        '''Train the model'''
        # Split features and targets into train and test data sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.Features, self.Targets,
                                                                                test_size=0.2, random_state=42)

        # build the Sequential model by stacking layers
        self.model = Sequential()
        self.model.add(Dense(500, input_shape=(4049,), activation="relu"))
        #self.model.add(Dropout(0.4))
        self.model.add(Dense(500, activation="relu"))
        #self.model.add(Dropout(0.4))
        self.model.add(Dense(3893, activation="relu"))

        self.model.summary()

        self.model.compile(optimizer="adam",
                      loss="mse",
                      metrics=[tf.keras.metrics.MeanSquaredError()])

        # Train the model
        self.model.fit(self.X_train, self.y_train, epochs=10)

        return

    def model_predict(self):
        '''Validate the model'''

        # Validation and evaluation
        #self.model.evaluate(self.X_test, self.y_test, verbose=2)

        self.y_pred = self.model.predict(self.X_test)

        # model performance
        self.score = self.performance_metric(self.y_test, self.y_pred)

        print("{0} model has an R2 score: {1:.2f}".format(self.model, self.score))

        return

    def wave_forecast(self, X_fore):
        '''wave forecasting'''

        y_fore = self.model.predict(X_fore)

        return y_fore

    def performance_metric(self, y_true, y_pred):
        '''Calculates and returns the performance score between
        true and predicted values based on the metric chosen.'''

        # Calculate the performance score
        score = r2_score(y_true, y_pred)

        return score