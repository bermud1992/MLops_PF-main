FROM python:3.10-slim-buster
# Install python packages
COPY requirements.txt /tmp
RUN pip install psycopg2-binary 
RUN pip install -r /tmp/requirements.txt
