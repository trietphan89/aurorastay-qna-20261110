FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY deployment/frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY deployment/frontend/app.py /app/app.py
COPY .streamlit/config.toml /app/.streamlit/config.toml
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
