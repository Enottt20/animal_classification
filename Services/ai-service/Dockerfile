FROM python:3.11-slim

WORKDIR /src

COPY requirements.txt /src/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /src/app
COPY ./templates /src/templates
COPY ./animalsv2.keras /src/animalsv2.keras

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "5001"]
