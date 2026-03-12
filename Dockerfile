FROM python:3.13
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", f"{$PORT}"]