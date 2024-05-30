FROM python:3.11
#
WORKDIR /code
COPY requirements.txt /code/requirements.txt
COPY inference_api.py /code/app.py
# 
# COPY . .
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# 
#WORKDIR /code
# 
# COPY models /code/models
# COPY encoders /code/encoders
#COPY inference/inference_api.py /code/app.py
#
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
