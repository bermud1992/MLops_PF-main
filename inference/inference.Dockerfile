FROM python:3.11
#
COPY inference/requirements.txt /code/requirements.txt
# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# 
WORKDIR /code
# 
# COPY models /code/models
# COPY encoders /code/encoders
COPY inference/inference_api.py /code/app.py
#
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]