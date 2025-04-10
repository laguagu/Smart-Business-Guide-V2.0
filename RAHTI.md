# Step-by-step Instructions for Deploying Smart Business Guide to Rahti

## 1. Build the Docker image

```bash
docker build -t smart-business-guide:latest .
```

## 2. (Optional) Test the Docker image locally

```bash
docker run -p 8501:8501 --env-file .env smart-business-guide:latest
```

Check the application at http://localhost:8501

## 3. Upload the Docker image to Rahti registry

First, obtain the Rahti login command:

1. Go to the Rahti web console
2. Click on your username in the upper right corner
3. Select "Copy login command"
4. Authenticate if prompted
5. Copy the provided login command and run it in your terminal

```bash
# Log in to the CSC Rahti 2 Docker registry
docker login -u unused -p $(oc whoami -t) image-registry.apps.2.rahti.csc.fi

# Tag the Docker image
docker tag smart-business-guide image-registry.apps.2.rahti.csc.fi/upbeat-apps/smart-business-guide:latest

# Push the Docker image to the Rahti registry
docker push image-registry.apps.2.rahti.csc.fi/upbeat-apps/smart-business-guide:latest
```

## 4. Create the application in Rahti management panel

1. Log in to the Rahti management panel
2. Select your project
3. Select "Add to Project" > "Deploy Image"
4. In the Image Name field, enter `docker-registry.rahti.csc.fi/upbeat-apps/smart-business-guide:latest`
5. Give your application a name, e.g., "smart-business-guide"
6. **Set environment variables** in the "Environment Variables" section:
   - `TAVILY_API_KEY`: Your Tavily API key
   - `GROQ_API_KEY`: Your Groq API key
   - `OPENAI_API_KEY`: Your OpenAI API key
7. Create the application by clicking "Create"

## 5. Check that the application works

1. Wait for the application to start in Rahti
2. Open the application URL from the Rahti management panel
3. Test the RAG functionality by making queries that require database lookups
