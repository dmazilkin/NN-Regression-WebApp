# Neural network for solving regression problem with deployment

This repo contents two following sections: 
## 1. Visualization of training process and dataset analysis
Dataset analysis, preprocessing, metrics and other analyses are shown in Jupyter notebook **regression.ipynb**.
 ## 2. Source files for neural network deployment
Three Python files can be found in *src/* folder:
1. **server.py** - contains logic running local server, loading model and predicting target variables,
2. **client.py** - contains logic for sending GET and POST requests to local server,
3. **neural_network** - this folder contains **train.py** script for training neural network and **inference.py** implementing *NeuralNetwork* class for loading and predicting target variables,
4. **neural_network/helpers** - this folder contains some useful functions and variables for generating training data, preprocessing functions and so on,
5. **data** - this folder contains trained neural network configs.

NOTE: **data/** folder contains three files:
1. neural_network.keras (*TRAINED_MODEL* variable in helpers.py) - contains trained neural network,
2. pipeline.json (*PIPELINE* variable in config.py) - contains configs for MinMaxScaler and PolynomialFeatures for features,
3. scaler.json (*SCALER* variable in config.py) - contains MinMaxScaler configs for target variables.

# Usage
All requirements need for running scripts can be found in *requirements.txt*.
## Train the neural network
To train the neural network run:
```console
cd src/
python neural_network/train.py
```
## Use trained neural network configs for predicting targets
### Running local server:
Firs run local server:
```console
python server.py
```
this command start Uvicorn *local server* at *port 5000*. Also, server.py contains business logic for handling:
1. GET '/' request - this endpoint returns HTML with this README.md,
2. GET '/func' - this endpoint returns the neural network prediction based on the passed URL parameters,
3. POST '/func' - this endpoint returns the neural network prediction based on the data from JSON request body.
### Sending requests
Using python script **client.py**:
```console
python client.py
```
next you can choose from three options:
1. *home* - for receiving home page '/' with GET request,
2. *vector* - for sending one feature vector with GET request on endpoint 'func/' and receiving prediction as text/plain response, feature vector can be specified as follows:
```console
x1=0.1, x3=-1.1, x2=0.02, x4=0
```
after what the GET request *http://127.0.0.1:5000/func?x1=0.1&x3=-1.1&x2=0.02&x4=0* will be sent,
3. *matrix* - for sending multiple feature vectors with POST request on endpoint 'func/' and receiving predictions as application/json response,
for more details see realization in the **client.py**, feature matrix can be specified as follows:
```console
[[2.647, 2.144, 0.112, 6.978], [0.4, -2.1, 0.07, -0.9]]
```
after what the POST request *http://127.0.0.1:5000/func* will be sent.


