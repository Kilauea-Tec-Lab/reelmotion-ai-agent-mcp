#!/bin/bash

#Comand Upload Project to server
#gcloud compute ssh reelmotion-vm --zone=us-central1-a --command="cd reelmotion-ai-agent-mcp ; git fetch origin ; git reset --hard origin/main ; bash deploy.sh"

set -euo pipefail

echo "🚀 Starting ReelMotion deployment..."

echo "📥 Pulling latest code..."
git pull origin main

echo "🛑 Stopping containers..."
docker-compose down

echo "🔨 Building images..."
docker-compose build --no-cache

echo "▶️  Starting services..."
docker-compose up -d

echo "📊 Containers:"
docker-compose ps

if [ "${SKIP_HEALTHCHECK:-0}" = "1" ]; then
    echo "⚠️  Skipping health check (SKIP_HEALTHCHECK=1)"
else
    echo "🏥 Readiness check (OPTIONS /api/chat via nginx:80)..."
    max_attempts="${HEALTHCHECK_ATTEMPTS:-30}"
    attempt=1

    until curl -fsS -X OPTIONS --connect-timeout 2 --max-time 5 http://localhost/api/chat > /dev/null 2>&1; do
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "❌ Health check failed after $max_attempts attempts!"
            echo "🧾 API logs (last 200 lines):"
            docker-compose logs --tail=200 api || true
            echo "🧾 Nginx logs (last 200 lines):"
            docker-compose logs --tail=200 nginx || true
            echo "🧾 Redis logs (last 200 lines):"
            docker-compose logs --tail=200 redis || true
            exit 1
        fi
        echo "⏳ Not ready yet... ($attempt/$max_attempts)"
        attempt=$((attempt+1))
        sleep 2
    done

    echo "✅ Deployment successful!"
fi

echo "🧹 Cleanup..."
docker image prune -f

echo "🎉 Deployment complete!"
docker-compose ps
