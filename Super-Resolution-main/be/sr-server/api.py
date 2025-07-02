import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from celery import Celery
from celery.result import AsyncResult
from dotenv import load_dotenv
from utility import setup_logging
import requests

load_dotenv()

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins in production for security
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

logger = logging.getLogger('sr.api')
logger = setup_logging(logger)
logger.info('Starting SR Client API Server.')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
JWT_SECRET = os.getenv('SUPABASE_JWT_SECRET')

celery = Celery(
    'tasks',
    broker='amqp://guest:guest@192.168.1.7:5672//',
    backend='rpc://'
)

celery.conf.update(
    task_track_started=True,
    task_time_limit=300,
    worker_concurrency=4,
    broker_transport_options={'heartbeat': 60}
)

# Pydantic models
class TaskResponse(BaseModel):
    task_id: str

class ProcessedImage(BaseModel):
    filename: str
    mimetype: str
    data: str

def get_queue_length(queue_name, vhost='/', host='192.168.1.7', port=15672, username='guest', password='guest'):
    # Encode the vhost properly if it's not '/'
    vhost_encoded = vhost if vhost != '/' else '%2F'
    url = f"http://{host}:{port}/api/queues/{vhost_encoded}/{queue_name}"
    response = requests.get(url, auth=(username, password))
    if response.status_code == 200:
        data = response.json()
        return data.get("messages", 0)
    else:
        # Handle errors appropriately in production
        return None

# Endpoints
@app.post("/api/task/create", response_model=TaskResponse)
async def enhance_image(
    request: Request,
    image: UploadFile = File(...),
):
    if not image.content_type.startswith('image/'):
        raise HTTPException(400, "Invalid file type")

    client_host = request.client.host
    if "x-forwarded-for" in request.headers:
        # Handle proxy chains
        forwarded = request.headers["x-forwarded-for"]
        client_host = forwarded.split(",")[0].strip()
    elif "x-real-ip" in request.headers:
        client_host = request.headers["x-real-ip"]

    extension = 'jpg' if 'jpeg' in image.content_type else 'png'
    file_data = await image.read()
    
    task = celery.send_task(
        'sr_tasks.process_image_task',
        args=[
            file_data, 
            extension,
            {
                "ip": client_host
            }
        ]
    )
    
    return {"task_id": task.id}

@app.post("/api/task/test", response_model=TaskResponse)
async def test(
    # user: dict = Depends(get_current_user)
):
    task = celery.send_task(
        'sr_tasks.test_task',
        args=[
            "placeholder",
            "placeholder",
        ]
    )
    
    return {"task_id": task.id}

@app.get("/api/task/{task_id}")
async def get_task_result(task_id: str):
    result = AsyncResult(task_id)
    queue_length = get_queue_length("celery")
    if not result.ready():
        return {"status": result.state, 'queue_length': queue_length}
    
    if result.failed():
        return {"status": "failed", "error": str(result.result)}
    
    return {
        "status": "complete",
        "result": result.result
    }

logger.info('''

                                            8888888b.                            888          888    d8b                                             d888        .d8888b.  
                                            888   Y88b                           888          888    Y8P                                            d8888       d88P  Y88b 
                                            888    888                           888          888                                                     888       888    888 
.d8888b  888  888 88888b.   .d88b.  888d888 888   d88P .d88b.  .d8888b   .d88b.  888 888  888 888888 888  .d88b.  88888b.                   888  888  888       888    888 
88K      888  888 888 "88b d8P  Y8b 888P"   8888888P" d8P  Y8b 88K      d88""88b 888 888  888 888    888 d88""88b 888 "88b                  888  888  888       888    888 
"Y8888b. 888  888 888  888 88888888 888     888 T88b  88888888 "Y8888b. 888  888 888 888  888 888    888 888  888 888  888      888888      Y88  88P  888       888    888 
     X88 Y88b 888 888 d88P Y8b.     888     888  T88b Y8b.          X88 Y88..88P 888 Y88b 888 Y88b.  888 Y88..88P 888  888                   Y8bd8P   888   d8b Y88b  d88P 
 88888P'  "Y88888 88888P"   "Y8888  888     888   T88b "Y8888   88888P'  "Y88P"  888  "Y88888  "Y888 888  "Y88P"  888  888                    Y88P  8888888 Y8P  "Y8888P"  
                  888                                                                                                                                                      
                  888                                                                                                                                                      
                  888

''')
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3073)
