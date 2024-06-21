# api

## Deployement :

### Send docker image

```bash
gcloud auth configure-docker
docker compose build <service_name>
docker tag <service_name> gcr.io/annual-project-427112/<service_name>
docker push gcr.io/annual-project-427112/<service_name>
```

### Deploy cloud run

gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --set-env-vars HUGGINGFACE_TOKEN=<>,API_TEXT_IP=<> ....
