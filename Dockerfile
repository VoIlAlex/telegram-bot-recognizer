FROM python:3.11-alpine3.17

RUN apk update && apk add --update gcc python3-dev musl-dev linux-headers g++

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY install.sh .
RUN sh install.sh
