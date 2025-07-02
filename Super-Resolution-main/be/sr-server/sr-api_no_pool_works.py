# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
#
import faulthandler
from sys import maxsize
import time
import os
import queue
import logging
import threading
from logging.handlers import RotatingFileHandler
import json
import base64
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS

from mimetypes import guess_type
from uuid_extensions import uuid7

import tensorflow_hub as hub

from dotenv import load_dotenv
from supabase import create_client, Client

from celery import Celery



# consts and env
os.environ['KMP_DUPLICATE_LIB_OK']='True'

load_dotenv()

LOGS_DIR = 'logs'
UPLOAD_DIR = 'uploads'
PROCESSED_DIR = 'processed'
REPORT_DIR = 'reports'
REQUEST_INFO_DIR = 'requests'
POOL_SIZE = 2
ALLOWED_IMAGE_MIME_TYPES = {'image/jpeg', 'image/png'}    
SAVED_MODEL_PATH = "./esrgan-tf2/1"
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
JWT_SECRET = os.getenv('SUPABASE_JWT_SECRET')
utc_plus_4 = timezone(timedelta(hours=4))




# faulthandler
timestamp = int(time.time())
filename = f'{os.path.join(os.path.normpath(LOGS_DIR), str(timestamp))}.log'
with open(filename, "w") as f:
    faulthandler.enable(file=f, all_threads=True)





# app setup
app = Flask(__name__)
CORS(app)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def setup_logging():
    # Create a rotating log handler (logs to a file with rotation)
    handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    
    # Define the log format
    log_format = ('%(asctime)s [%(levelname)s] '
                  '%(filename)s:%(lineno)d - %(message)s')
    
    # Set formatter
    handler.setFormatter(logging.Formatter(log_format))
    
    # Add the handler to the Flask app logger
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

setup_logging()
app.logger.info('Starting SR API Server.')
app.logger.info('''

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



# Celery setup inside Internal API
celery = Celery(
    'sr_tasks',
    broker='amqp://guest:guest@localhost:5672//',  # Connect to RabbitMQ
    backend='rpc://'  # Using RabbitMQ for result backend
)

celery.conf.update(
    task_track_started=True,
    task_time_limit=300,  # Timeout of 5 minutes
    worker_concurrency=2,  # Adjust based on server resources
    # include=['sr-api']  # Important: Include the module in Celery's task discovery
)


# Model pool setup
def InitModel():
    import super_resol
    enh_model = hub.load(SAVED_MODEL_PATH)
    return enh_model, super_resol.enhance_image_api_method

# app.logger.info(f'Start pool generation for model instances of size: {POOL_SIZE}')
# model_pool = queue.Queue(maxsize=POOL_SIZE)
# for _ in tqdm(range(POOL_SIZE), desc='Generating model instance pool...'):
#     model_pool.put(InitModel())
# app.logger.info('Pool generation successful')




def get_new_uuid7(ns: int):
    return uuid7(as_type="uuid", ns=ns)

def placeholder(path):
    return [path, path, path]

def is_image(file):
    return file.content_type in ALLOWED_IMAGE_MIME_TYPES

def get_extension(file):
    extension = ''
    if file.content_type == "image/jpeg":
        extension = "jpg"
    elif file.content_type == "image/png":
        extension = "png"
    return extension



class ProcessingResult():
    def __init__(self):
        self.result = None
        self.error = None
        # self.event = threading.Event()

def process_thread(file_path, pool, result):
    app.logger.info(f'Starting thread for {file_path}')
    # Get model from pool
    if pool.empty():
        app.logger.warning(f'All model instances are busy. Closing thread for {file_path}')
        result.error = 'Server is busy. Please try again later.'
        result.event.set()

    try:
        app.logger.info(f'Pool size {pool.qsize()}')
        model_instance, method = pool.get()
        app.logger.info('Model instance retrieved from pool')
        app.logger.info(f'Pool size {pool.qsize()}')
        # Get processed images
        processed_images = method(file_path, model_instance)

        if processed_images is None:
            result.error = 'Error occured in image processing (No retina found). Please retry. If issue persists, contact system administrator.'

        app.logger.info('Image processed successfully')
        result.result = processed_images

    except Exception as e:
        result.result = []
        result.error = f'Error in processing. Please try again. {e}'

    finally:
        # Return model instance to pool
        pool.put(model_instance, method)
        app.logger.info('Model instance returned to pool')
        app.logger.info(f'Pool size {pool.qsize()}')

    app.logger.info(f'Closing thread for {file_path}')
    result.event.set()


# @app.route('/api/pool_size', methods=['GET'])
# def get_pool_size():
#     return {'pool_size': model_pool.qsize()}


# Celery Task
@celery.task(bind=True, autoretry_for=(Exception,), max_retries=3)
def test_task(self, *args):
    app.logger.debug(f"Received args: {args}")
    email, user_id = args[0], args[1]
    fields = []
    # fields.append({'pool_size': model_pool.qsize()})
    fields.append({'email': email})
    fields.append({'user_id': user_id})
    return {'data': fields}


@celery.task(bind=True, autoretry_for=(Exception,), max_retries=3, 
             retry_backoff=True, retry_backoff_max=60, retry_jitter=False)
def process_image_task(self, file_data: bytes, extension: str):
    """Process image data in the internal API."""
    # if model_pool.empty():
    #     app.logger.warning('Rejecting task due to empty model pool. Capacity full for instance.')
    #     raise Exception("Model pool is empty. Task rejected.")  # Reject task immediately
    
    try:
        # Generate request ID
        unixt = time.time()
        requestId = get_new_uuid7(int(unixt * 1000))

        # Save request image
        filename = f'{requestId}.{extension}'
        file_path = os.path.join(os.path.normpath(UPLOAD_DIR), filename)
        with open(file_path, 'wb') as f:
            f.write(file_data)
        # file.save(file_path)

        result = ProcessingResult()
        # thread = threading.Thread(target=process_thread, args=(file_path, model_pool, result))
        # thread.start()
        # result.event.wait()

        modelinit = time.time()
        import super_resol
        app.logger.info('Initializing model inside task.')
        model_instance = hub.load(SAVED_MODEL_PATH)
        method = super_resol.enhance_image_api_method
        app.logger.info(f'Initializing model inside task took {round(time.time() - modelinit, 3)} s.')

        app.logger.info(f'Starting thread for {file_path}')
        # Get model from pool
        if model_instance is None:
            app.logger.warning(f'All model instances are busy. Closing thread for {file_path}')
            result.error = 'Server is busy. Please try again later.'
            # result.event.set()
        else:
            # model_instance = None
            # method = None

            try:
                # app.logger.info(f'Pool size {model_pool.qsize()}')
                # model_instance, method = model_pool.get()
                # app.logger.info('Model instance retrieved from pool')
                # app.logger.info(f'Pool size {model_pool.qsize()}')
                # Get processed images
                processed_images = method(file_path, model_instance)

                if processed_images is None:
                    result.error = 'Error occured in image processing (No retina found). Please retry. If issue persists, contact system administrator.'

                app.logger.info('Image processed successfully')
                result.result = processed_images

            except Exception as e:
                result.result = []
                result.error = f'Error in processing. Please try again. {e}'

            # finally:
                # Return model instance to pool
                # model_pool.put(model_instance, method)
                # app.logger.info('Model instance returned to pool')
                # app.logger.info(f'Pool size {model_pool.qsize()}')

            app.logger.info(f'Closing thread for {file_path}')



        if result.error:
            return {'error': result.error}

        processed_images = result.result

        # Save processed images
        # TODO: Move the save logic to execute async so response is sent back even faster. Will have to convert the processed_images to base64
        processed_images_paths = []
        proccessed_images_dir = os.path.join(os.path.normpath(PROCESSED_DIR), str(requestId))
        os.makedirs(proccessed_images_dir, exist_ok=True)
        for i, img in enumerate(processed_images):
            proccessed_filename = f'{requestId}_{i}.{extension}'
            pathforprocessed = os.path.join(proccessed_images_dir, proccessed_filename)
            processed_images_paths.append(pathforprocessed)
            img.save(pathforprocessed) 

        # Save request info
        requestInfo = {
            "timestamp": str(datetime.fromtimestamp(unixt)),
            "requestId": str(requestId),
            # "userId": user_id,
            # "userEmail": user_email,
            # "userLastSignIn": str(user_last_sign_in_at.astimezone(utc_plus_4)),
            "requestFilePath": file_path,
            "requestProcessedImagesPaths": processed_images_paths,
            # "ip_address": request.remote_addr,
            # "user_agent": request.headers.get('User-Agent'),
        } 
        request_info_filepath = os.path.join(os.path.normpath(REQUEST_INFO_DIR), str(requestId))
        with open(f'{request_info_filepath}.json', 'w') as json_file:
            json.dump(requestInfo, json_file, indent=2)
        app.logger.info(f'Request info written to {request_info_filepath}')

        # Save pdf
        # pdf_data = reate_pdf(file_path, processed_images_paths, requestInfo)

        # Generate response 
        # TODO: rewrite to use raw image data not saved images - check how much time can be saved. only do it if time is more than 100ms
        fields = []
        for i, path in enumerate(processed_images_paths):
            if os.path.exists(path):
                with open(path, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    mimetype = guess_type(path)[0] or 'application/octet-stream'
                    img_data = {
                        'filename': os.path.basename(path),
                        'mimetype': mimetype,
                        'data': img_base64
                    }
                    fields.append(img_data)
            else:
                print(f'Warning: {path} does not exist')
                return {'error': 'Error occured in image processing (No processed images saved on server). Please retry. If issue persists, contact system administrator.'}

        # # Save PDF for test purposes
        # pdf_output_path = os.path.join(os.path.normpath(REPORT_DIR), f'{str(requestId)}.pdf')
        # # with open(pdf_output_path, 'wb') as f:
        # #     f.write(pdf_data)
        # # print(f"PDF saved to: {pdf_output_path}")
        #
        # # Add PDF to the response
        # pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        # pdf_obj = {
        #     'filename': os.path.basename(pdf_output_path),
        #     'mimetype': 'application/pdf',
        #     'data': pdf_base64
        # }
        # fields.append(pdf_obj)

        return {'images': fields}
    except Exception as e:
        app.logger.warning(f"Task failed: {str(e)}")
        return {'error': f'Task failed: {str(e)}'}


if __name__ == '__main__':
    # Start the Celery worker
    celery_worker_thread = threading.Thread(target=lambda: celery.worker_main(argv=['worker', '--loglevel=debug', '--concurrency=3']))
    celery_worker_thread.daemon = True  # Daemonize thread so it exits when the app exits
    celery_worker_thread.start()

    app.run('0.0.0.0', 4073, debug=False, threaded=True)
    app.logger.info('Shutting down SR API Server.')
