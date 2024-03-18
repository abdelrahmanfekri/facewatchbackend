# Base image
FROM python:3.8-slim-buster

# Set working directory
WORKDIR /code

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=backend.settings

# Collect static files
RUN python manage.py collectstatic --noinput

# Start the Django development server when the container starts
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
 