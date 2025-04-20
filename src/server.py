import uvicorn
import asyncio
from time import sleep
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
async def home(request: Request):
    loop = asyncio.get_event_loop()
    template = await loop.run_in_executor(None, templates.TemplateResponse, 'home.html', {'request': request})
    return template

@app.get('/func', response_class=PlainTextResponse)
async def get_func(x1: float, x2: float, x3: float, x4: float):
    loop = asyncio.get_event_loop()
    predict = await loop.run_in_executor(None, nn.predict, [[x1, x2, x3, x4]])
    return PlainTextResponse(str(predict[0][0]))

@app.post('/func')
async def post_func(data: List[List[float]]):
    loop = asyncio.get_event_loop()
    predict = await loop.run_in_executor(None, nn.predict, data)
    return JSONResponse(predict[0])

def main():
    # Configure and start Uvicorn server
    config = uvicorn.Config(app, host=LOCALHOST, port=PORT, log_level='info')
    server = uvicorn.Server(config)
    server.run()

if __name__ == '__main__':
    main()