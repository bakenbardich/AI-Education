FROM python:3.10-slim-buster

RUN pip install explainerdashboard
RUN pip install catboost
RUN pip install optuna

COPY model.py ./
COPY app.py ./

RUN python model.py

EXPOSE 9050
CMD ["python", "./app.py"]