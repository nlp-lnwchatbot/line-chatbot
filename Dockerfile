FROM python:3.10.13
COPY . /app/
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "main.py"]