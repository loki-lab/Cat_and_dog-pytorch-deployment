FROM python:3

WORKDIR /flaskProject1

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "./app.py"]
