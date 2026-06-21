# reelmotion AI Agent MCP

AI-powered agent for image and video generation using Google Gemini and multiple AI models.

## Features

- 🤖 **Gemini-powered chatbot** with multi-language support (English/Spanish auto-detection)
- 🖼️ **Image generation** (Seedream, GPT, Nano Banana 2, Midjourney)
- 🎬 **Video generation** (Runway Aleph, Veo 3.1, Seedance 2.0, and more)
- 💾 **Redis-based session management** for concurrent conversations
- 🔄 **Reference file persistence** (images/videos via URL)
- 📊 **Token cost calculation** before generation
- 🌐 **Multi-session support** with UUID-based conversation tracking
- 🚀 **URL-based file handling** (no base64, optimized performance)

## Tech Stack

- **Python 3.11+**
- **Google Gemini API** (gemini-2.5-flash)
- **Redis** (for session storage)
- **FastMCP** (Model Context Protocol server)
- **Docker & Docker Compose**

## 🆕 New: URL-Based File Handling

The MCP now accepts **file URLs directly** instead of base64 encoding:

```php
// Laravel example
Http::asForm()->post('http://localhost/api/chat', [
    'message' => 'Generate an image with this reference',
    'token' => $token,
    'conversation_uuid' => $uuid,
    'files[0]' => 'https://storage.googleapis.com/bucket/image.jpg',
    'file_types[0]' => 'image',
]);
```

**Benefits:**

- ✅ No base64 conversion overhead
- ✅ Reduced memory usage in Redis
- ✅ Faster processing
- ✅ Direct integration with Cloud Storage

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for detailed examples.

## Quick Start (Production)

### Prerequisites

- Ubuntu 22.04+ VM (Google Cloud, AWS, etc.)
- Docker & Docker Compose installed
- Domain name (optional, for SSL)

### Deployment

```bash
# Clone repository
git clone https://github.com/VictorEspinosa98/reelmotion-ai-agent-mcp.git
cd reelmotion-ai-agent-mcp

# Configure environment
nano .env.production
# Add your GOOGLE_API_KEY

# Give execution permissions
chmod +x deploy.sh

# Deploy
./deploy.sh
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Local Development (Windows)

### Prerequisites

- Python 3.11+
- Redis running locally or Docker
- Google Gemini API Key

### Setup

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your API keys

# Start Redis (use redis-portable or Docker)
cd redis-portable
.\redis-server.exe redis.windows.conf

# Start server (in another terminal)
python reelmotion_mcp/server.py http
```

## API Endpoints

### Chat

```bash
POST /api/chat
Content-Type: multipart/form-data

{
  "message": "Generate an image of a christmas pug",
  "conversation_uuid": "unique-session-id",
  "token": "optional-bearer-token"
}
```

### Health Check

```bash
GET /health
```

## Available Models

### Image Generation

- **Seedream** - Realism / photographic fidelity, cinematic, reference images (4 tokens/image) — recommended default
- **GPT** - Readable text in image, strict instruction following (6 tokens/image)
- **Nano Banana 2** - Quick edits / multi-reference composition (7 tokens/image)
- **Midjourney** - Artistic style, illustration, creative concepts (9 tokens/image)

### Video Generation

- **Runway Aleph** (19 tokens/sec) - 5-10 seconds, video-to-video
- **Veo 3.1** (48 tokens/sec) - 8 seconds, high quality
- **Veo 3.1 Flash** (21 tokens/sec) - 8 seconds, fast & economical
- **Veo 3.1 Ultra** (60 tokens/sec) - 8 seconds, maximum quality

## Architecture

```
┌─────────────────┐
│   Frontend      │
│   (React)       │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐     ┌─────────────────┐
│   FastMCP       │────▶│   Google        │
│   Server        │     │   Gemini API    │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Redis         │     │   reelmotion    │
│   (Sessions)    │     │   Backend API   │
└─────────────────┘     └─────────────────┘
```

## Environment Variables

```env
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
REDIS_URL=redis://localhost:6379
BACKEND_URL=https://backend.reelmotion.ai/
IMAGE_CREATION_ENDPOINT=api/ai/mcp-image-generation
VIDEO_CREATION_ENDPOINT=api/ai/mcp-video-generation
```

## Contributing

This is a private project for reelmotion Media Ltd.

## License

Proprietary - All rights reserved reelmotion Media Ltd.

## Support

For issues, contact: reelmeinmedialtd@gmail.com
