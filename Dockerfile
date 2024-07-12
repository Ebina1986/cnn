FROM python:3.9.11

COPY .. .

WORKDIR /var/www


RUN pip install -r requirements.txt


CMD ["uvicorn" , "main:app" , "--host" , "0.0.0.0" , "--port", "8001"]