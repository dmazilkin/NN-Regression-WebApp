import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from typing import List

from neural_network.inference import NeuralNetwork
from server_config import LOCALHOST, PORT

# Create FastAPI application
app = FastAPI()
templates = Jinja2Templates('templates')
# Load trained neural network
nn = NeuralNetwork()

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})

@app.get('/func', response_class=PlainTextResponse)
def get_func(x1: float, x2: float, x3: float, x4: float):
    predict = nn.predict([[x1, x2, x3, x4]])[0][0]
    return PlainTextResponse(str(predict))

@app.post('/func')
def post_func(data: List[List[float]]):
    predict = nn.predict(data)[0]
    return JSONResponse(predict)

def main():
    # Configure and start Uvicorn server
    config = uvicorn.Config(app, host=LOCALHOST, port=PORT, log_level='info')
    server = uvicorn.Server(config)
    server.run()

if __name__ == '__main__':
    main()