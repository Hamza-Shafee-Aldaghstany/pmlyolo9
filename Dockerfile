# Use the CUDA 12.2 image as the parent image
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Set the working directory inside the container
WORKDIR /app

# Set the environment to avoid interactive timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Install Python 3.9, pip, and other necessary libraries, including timezone data
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    libgl1-mesa-glx \
    tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Copy the current directory contents into the container at /app
COPY ./ /app

# Install the dependencies from requirements.txt
COPY requirements.txt ./ 
RUN python -m pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI app using uvicorn
CMD ["sh", "-c", "cd /app/deployment/api && uvicorn api:app --host 0.0.0.0 --port 8000 --reload"]
