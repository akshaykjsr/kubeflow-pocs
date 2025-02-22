FROM python:3.8
WORKDIR /app
COPY train_model.py /app/train_model.py
RUN pip install numpy scikit-learn
ENTRYPOINT ["python3", "/app/train_model.py"]

