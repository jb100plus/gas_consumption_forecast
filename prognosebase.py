import numpy as np
import keras
from keras.models import model_from_json
from keras import metrics
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import sys
import json
import pandas as pd


class kerascallback(keras.callbacks.Callback):
    def __init__(self, seconds=None, safety_factor=1, verbose=0):
        super(keras.callbacks.Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if epoch > 0:
            if epoch % 10 == 0:
                print('.', end='')
                sys.stdout.flush()
            if epoch % 1000 == 0:
                print("{:6,.0f}".format(epoch))

    def on_train_end(self, logs=None):
        print('')


class PrognoseBase:
    def __init__(self, modelname, y_col_name):
        self.limits_jsonfile = modelname + '_limits.json'
        self.features_jsonfile = modelname + '_features.json'
        limits = dict()
        try:
            with open(self.limits_jsonfile, 'r') as rp:
                limits = json.load(rp)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            pass
        try:
            self.minvalue = limits['minvalue']
            self.maxvalue = limits['maxvalue']
            self.meanvalue = limits['meanvalue']
            self.stdvalue = limits['stdvalue']
        except KeyError:
            pass

        self.features = dict()
        try:
            with open(self.features_jsonfile, 'r') as rp:
                self.features = json.load(rp)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            pass

        self.modelname = modelname
        self.y_col_name = y_col_name
        self.model = None
        self.loss = [metrics.mse]
        self.optimizer = keras.optimizers.Adam(lr=0.001)
        self.metrics = [metrics.mse, metrics.mae]
        ocb = kerascallback()
        self.callbacks = [
            # keras.callbacks.TensorBoard(log_dir='gaspronose_log', histogram_freq=1),
            # Interrupt training if `val_loss` stops improving for over 100 epochs
            keras.callbacks.EarlyStopping(patience=500, monitor='val_loss'),
            # keras.callbacks.EarlyStopping(patience=100, monitor='loss'),
            # keras.callbacks.CSVLogger(filename=self.modelname + 'log.csv'),
            keras.callbacks.ModelCheckpoint(self.modelname + '.check', verbose=0, save_best_only=True,
                                            save_weights_only=True), ocb
        ]

    # normalized labels must be between 0..1, otherwise the values ​​will overflow
    def normalize_labels(self, data, min, max, lowerlimit=0, upperlimit=1):
        between_0_max = [(v - min) for v in data]
        between_0_1 = [v / (max - min) for v in between_0_max]
        between_0_2 = [v * (upperlimit - lowerlimit) for v in between_0_1]
        between_m1_1 = [v + lowerlimit for v in between_0_2]
        return between_m1_1

    # normalized labels must be between 0..1, otherwise the values ​​will overflow
    def denormalize_labels(self, data, min, max, lowerlimit=0, upperlimit=1):
        between_m1_1 = data
        between_0_2 = [v - lowerlimit for v in between_m1_1]
        between_0_1 = [v / (upperlimit - lowerlimit) for v in between_0_2]
        between_0_max = [v * (max - min) for v in between_0_1]
        between_min_max = [v + min for v in between_0_max]
        return between_min_max

    # proposed method to normalize features
    def normalize_features(self, data, mean, std):
        return [(d - mean) / std for d in data]

    def plot_xy(self, x, y, n1, n2):
        xo = np.arange(0, len(x), 1)
        plt.plot(xo, x, 'b-')
        plt.plot(xo, y, 'g-')
        plt.legend(['x', 'y'])
        plt.legend([n1, n2])
        plt.grid()
        plt.show()

    def plot_comparision(self, x, y, comp1, comp2):
        xo = np.arange(0, len(x), 1)
        plt.plot(xo, y, 'b-')
        plt.plot(xo, comp1, 'g-')
        plt.plot(xo, comp2, 'y--')
        plt.legend([self.y_col_name, 'Prog KI', 'KI - Allok'])
        # Das Achsen-Objekt des Diagramms in einer Variablen ablegen:
        ax = plt.gca()
        # Die obere und rechte Achse unsichtbar machen:
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # Die untere Diagrammachse auf den Bezugspunkt '0' der y-Achse legen:
        ax.spines['bottom'].set_position(('data', 0))
        # Ausrichtung der Achsen-Beschriftung festlegen:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.title('Validation Prediction vs. Allocation')
        plt.ylabel('kW(h)')
        ax = plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.grid(True)
        # And a corresponding grid
        # ax.grid(which='both')
        plt.show()

    def save_model(self, model):
        model_json = model.to_json()
        with open(self.modelname + '.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(self.modelname + '.h5')
        print("saved model")

    def load_model(self):
        # load json and create model
        loaded_model = None
        if os.path.isfile(self.modelname + '.json'):
            json_file = open(self.modelname + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            if os.path.isfile(self.modelname + '.h5'):
                loaded_model.load_weights(self.modelname + '.h5')
            else:
                loaded_model = None
        if None == loaded_model:
            print("No model found")
        self.model = loaded_model
        return loaded_model

    def remove_model(self):
        if os.path.isfile(self.modelname + '.json'):
            os.remove(self.modelname + '.json')
        if os.path.isfile(self.modelname + '.h5'):
            os.remove(self.modelname + '.h5')
        self.model = None

    def get_training_data(self, rawdata):
        x = self.get_prediction_data(rawdata)
        y = list(rawdata[self.y_col_name])
        y = self.normalize_labels(y, self.minvalue, self.maxvalue)
        return x, y

    def get_prediction_data(self, rawdata):
        feature_values = list()
        for key, feature in self.features.items():
            raw = list(rawdata[feature['column']])
            feature_values.append(self.normalize_features(raw, feature['mean'], feature['std']))
        x = np.column_stack(feature_values)
        return x

    def compare(self, dataset, showplot=False):
        x, y = self.get_training_data(dataset)
        model = self.load_model()
        result = None
        if None != model:
            model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=self.metrics)
            # model.summary()
            preds = model.predict(x)
            ykwh = self.denormalize_labels(y, self.minvalue, self.maxvalue)
            pkwh = self.denormalize_labels(preds, self.minvalue, self.maxvalue)
            mse = mean_squared_error(ykwh, pkwh)
            print('mse allok kWh' + "{:20,.0f}".format(mse), "{:20,.0f}".format(mse ** 0.5))
            result = pd.DataFrame()
            result['I'] = dataset.index.values.tolist()
            result['Y'] = [round(y, 0) for y in ykwh]
            result['P'] = [round(p[0], 0) for p in pkwh]
            result['D'] = result['P'] - result['Y']
            result['A'] = ((result['D'] / result['Y']) * 100).round(2)
            if showplot:
                self.plot_comparision(x, result['Y'], result['P'], result['D'])
            sy = sum(result['Y'])
            sd = sum(result['D'])
            sab = sum(result['D'].abs())
            print(f'sum {sy:,.0f} error {sd:,.0f}, {sd / sy * 100:,.2f}%  min: {min(result["A"]):,.2f}% '
                  f'max: {max(result["A"]):,.2f}%; absolute error {sab:,.0f}, {sab / sy * 100:,.2f}%')
        return result

    def predict(self, prog_x):
        preds = None
        model = self.load_model()
        if None != model:
            model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics )
            preds = model.predict(prog_x)
            preds = self.denormalize_labels(preds, self.minvalue, self.maxvalue)
        return preds

    def train(self, train_x, train_y, dimension=64):
        history = None
        model = self.load_model()
        if model == None:
            np.random.seed(7)
            inputshape = train_x.shape[1]
            model = Sequential()
            model.add(Dense(dimension, input_dim=inputshape, activation="sigmoid"))
            #model.add(Dense(dimension, activation="relu"))
            model.add(Dense(dimension, activation="relu"))
            model.add(Dense(dimension, activation="relu"))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='mean_squared_error', optimizer=self.optimizer, )
            history = model.fit(train_x, train_y, validation_split=0.2, epochs=5000, verbose=0,
                                callbacks=self.callbacks)
            self.model = model
        return history

    def trainmodel(self, rawdata, features, showhistory=False):
        self.remove_model()
        rawdata_stats = rawdata.describe()
        train_stats = rawdata_stats.transpose()  # rotate 90°
        self.minvalue = train_stats.loc[self.y_col_name]['min']
        self.maxvalue = train_stats.loc[self.y_col_name]['max']
        self.meanvalue = train_stats.loc[self.y_col_name]['mean']
        self.stdvalue = train_stats.loc[self.y_col_name]['std']
        limits = dict()
        limits['minvalue'] = self.minvalue
        limits['maxvalue'] = self.maxvalue
        limits['meanvalue'] = self.meanvalue
        limits['stdvalue'] = self.stdvalue
        with open(self.limits_jsonfile, 'w') as fp:
            json.dump(limits, fp)
        for key, feature in features.items():
            train_stats = rawdata.describe()
            # rotate 90°
            train_stats = train_stats.transpose()
            col = feature['column']
            feature['min'] = train_stats.loc[col]['min']
            feature['max'] = train_stats.loc[col]['max']
            feature['mean'] = train_stats.loc[col]['mean']
            feature['std'] = train_stats.loc[col]['std']
        with open(self.features_jsonfile, 'w') as fp:
            json.dump(features, fp)
        self.features = features
        train_dataset = rawdata.sample(frac=0.8, random_state=0)
        test_dataset = rawdata.drop(train_dataset.index)
        train_x, train_y = self.get_training_data(train_dataset)
        history = self.train(train_x, train_y)
        if None != history:
            self.model.load_weights(self.modelname + '.check')  # revert to the best model
            self.save_model(self.model)
            if showhistory:
                print(min(history.history['loss']), np.argmin(history.history['loss']),
                      min(history.history['val_loss']), np.argmin(history.history['val_loss']))
                self.plot_xy(history.history['loss'][:], history.history['val_loss'][:], 'training', 'validation')
        return test_dataset

