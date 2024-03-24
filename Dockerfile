# Use an official Python runtime as a parent image
FROM python:3.8

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY . /code/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# make migrations
RUN python manage.py makemigrations

# Apply the migrations
RUN python manage.py migrate

# Set default values for creating superuser
ENV DJANGO_SUPERUSER_USERNAME=root
ENV DJANGO_SUPERUSER_EMAIL=root@gmail.com
ENV DJANGO_SUPERUSER_PASSWORD=root

# Create superuser non-interactively
RUN python manage.py shell -c "from django.contrib.auth.models import User; User.objects.create_superuser('$DJANGO_SUPERUSER_USERNAME', '$DJANGO_SUPERUSER_EMAIL', '$DJANGO_SUPERUSER_PASSWORD') if not User.objects.filter(username='$DJANGO_SUPERUSER_USERNAME').exists() else print('Superuser already exists')"

# Expose the port the app runs on
EXPOSE 8000

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
