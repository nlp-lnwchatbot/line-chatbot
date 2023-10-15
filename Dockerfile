FROM ubuntu:22.04
RUN apt-get update -y && apt-get install -y python3-pip python3-dev
WORKDIR /app
COPY . /app/
RUN pip install -r requirements.txt