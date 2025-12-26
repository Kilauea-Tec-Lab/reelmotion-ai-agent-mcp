#!/bin/bash

set -e

echo "🚀 Starting ReelMotion deployment..."

echo "📥 Pulling latest code..."
git pull origin main

echo "🛑 Stopping containers..."
docker-compose down

echo "🔨 Building images..."
docker-compose build --no-cache

echo "▶️  Starting services..."
docker-compose up -d

echo "⏳ Waiting for services..."
sleep 15

echo "🏥 Health check..."
if curl -sS http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ Deployment successful!"
else
    echo "❌ Health check failed!"
    docker-compose logs --tail=50 api
    exit 1
fi

echo "🧹 Cleanup..."
docker image prune -f

echo "🎉 Deployment complete!"
docker-compose ps
