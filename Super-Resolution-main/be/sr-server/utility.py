import logging
from logging.handlers import RotatingFileHandler
import colorlog
from uuid_extensions import uuid7
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

ALLOWED_IMAGE_MIME_TYPES = {'image/jpeg', 'image/png'}    

def get_new_uuid7(ns: int):
    return uuid7(as_type="uuid", ns=ns)

# def is_image(file):
#     return file.content_type in ALLOWED_IMAGE_MIME_TYPES
#
# def get_extension(file):
#     extension = ''
#     if file.content_type == "image/jpeg":
#         extension = "jpg"
#     elif file.content_type == "image/png":
#         extension = "png"
#     return extension

def create_pdf(request_image_path, processed_images_paths, request_info):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Add header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "SR App Report")

    # Add label for original image
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, height - 80, "Original Image")

    # Add original image
    p.drawImage(request_image_path, 50, height - 300, width=200, height=200, preserveAspectRatio=True)

    # Add label for request info
    p.setFont("Helvetica-Bold", 12)
    p.drawString(300, height - 80, "Request Information")

    # Add request info
    p.setFont("Helvetica", 10)
    p.drawString(300, height - 100, f"Timestamp: {request_info['timestamp']}")
    p.drawString(300, height - 120, f"Task ID: {request_info['taskId']}")
    p.drawString(300, height - 140, f"User Email: {request_info['userEmail']}")

    # Add label for processed images
    p.setFont("Helvetica-Bold", 12)
    p.drawString(50, height - 340, "Processed Images")

    y_offset = height - 500  # Starting position for processed images
    images_per_row = 3
    image_width = 150
    image_height = 150
    horizontal_gap = 10  # Reduced horizontal gap between images
    vertical_gap = 20     # Vertical gap between rows

    for i, path in enumerate(processed_images_paths):
        if i % images_per_row == 0 and i != 0:
            y_offset -= (image_height + vertical_gap)  # Move down for the next row

        x_offset = 50 + (i % images_per_row) * (image_width + horizontal_gap)  # Calculate x position
        p.drawImage(path, x_offset, y_offset, width=image_width, height=image_height, preserveAspectRatio=True)

    p.showPage()
    p.save()

    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data



def setup_logging(logger):
    # Create a rotating file handler
    file_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
    
    # Define log format for file handler
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Create a colored console handler
    console_handler = logging.StreamHandler()
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    console_handler.setFormatter(console_formatter)
    
    # Add the handler to the Flask logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger
