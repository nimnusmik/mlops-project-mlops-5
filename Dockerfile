FROM python:3.11-bookworm 

WORKDIR /opt



COPY . .



COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install pandas psycopg2-binary dotenv 


CMD ["python", "main.py"]
#CMD ["bash", "-c", "python main.py && python save_to_db.py"]
