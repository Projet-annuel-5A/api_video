FROM python:3.11.3

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]

EXPOSE 8002
