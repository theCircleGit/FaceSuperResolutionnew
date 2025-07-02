FROM sr_api_base:1.2

RUN mkdir /app

COPY api.py /app/api.py
COPY .env /app/.env
COPY utility.py /app/utility.py

WORKDIR /app

ENV BASE_DIR=/app

SHELL ["/bin/bash", "-c"]

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "3073"]
