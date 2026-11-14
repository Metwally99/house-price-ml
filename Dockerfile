FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask==3.1.2 waitress==3.0.2
COPY app.py .
COPY docs/ docs/
EXPOSE 8000
CMD ["waitress-serve", "--listen=0.0.0.0:8000", "app:app"]
