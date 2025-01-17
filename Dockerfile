FROM python:3.9.11



WORKDIR /var/www

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["uvicorn" , "main:app" , "--host" , "0.0.0.0" , "--port", "8001" , "--reload"]
