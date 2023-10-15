# LINE Chatbot - iPhone 15 Sentiment Analysis

## Environment setup

Copy `.env.example` to `.env` and fill in the environment variables.

```bash
cp .env.example .env
```

## Development setup

### Prerequisites

- [Python](https://www.python.org/downloads/)

### Installation

Install the required packages.

```bash
pip install -r requirements.txt
```

### Usage

Run the following command to start the server.

```bash
python main.py
```

## Deployment

### Docker compose

Run the following command to start the containers.

```bash
docker-compose up
```

### Google Cloud Run

1. Build the image.

   ```bash
   docker build -t gcr.io/<PROJECT_ID>/<IMAGE_NAME> .
   ```

2. Tag the image.

   ```bash
   docker tag <IMAGE_ID> gcr.io/<PROJECT_ID>/<IMAGE_NAME>:<TAG>
   ```

3. Push the image to Google Container Registry.

   ```bash
   docker push gcr.io/<PROJECT_ID>/<IMAGE_NAME>
   ```

4. Deploy the image to Google Cloud Run
