FROM dpbiomatrix/sr_base:1.3

WORKDIR /app

COPY . .

ENV BASE_DIR=/app

CMD ["celery", "-A", "sr_api", "worker", "--loglevel=debug", "--concurrency=2"]
