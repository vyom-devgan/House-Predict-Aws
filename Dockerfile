# Step 1: Use an official Python runtime as a parent image
FROM python:3.9-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . /app

# Step 4: Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Make port 8501 available to the world outside this container
EXPOSE 8501

# Step 6: Define environment variable to prevent buffering of output
ENV PYTHONUNBUFFERED=1

# Step 7: Run Streamlit when the container starts
CMD ["streamlit", "run", "app.py"]