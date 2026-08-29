# reelmotion AI Agent MCP

AI-powered agent for image and video generation using Google Gemini and multiple AI models.

## Features

- 🤖 **Gemini-powered chatbot** with multi-language support (English/Spanish auto-detection)
- 🖼️ **Image generation** (Seedream, Seedream Pro, GPT, Nano Banana 2, Midjourney)
- 🎬 **Video generation** (Runway Aleph 2, Veo 3.1, Seedance 2.5, Kling, and more)
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

- **Seedream** (UI: Seedream 5.0 Lite) - Realism / photographic fidelity, cinematic, reference images (3 tokens/image) — cheapest, recommended default
- **Seedream Pro** (UI: Seedream 5.0 Pro) - Same realism with higher fidelity, for maximum image quality (4 tokens/image)
- **GPT** (UI: GPT Image 2) - Readable text in image, strict instruction following (6 tokens/image)
- **Nano Banana 2** - Quick edits / multi-reference composition (8 tokens/image)
- **Midjourney** (UI: Midjourney V8.1) - Artistic style, illustration, creative concepts (9 tokens/image)

### Video Generation

- **Runway Aleph 2** (30 tokens/sec) - 5-10 seconds, video-to-video
- **Runway 4.5** (13 tokens/sec) - 5, 8 or 10 seconds, high quality
- **Veo 3.1** (42 tokens/sec) - 8 seconds, high quality
- **Veo 3.1 Lite** (6 tokens/sec) - 8 seconds, cheapest video with native audio
- **Veo 3.1 Flash** (11 tokens/sec) - 8 seconds, fast & economical
- **Veo 3.1 Ultra** (63 tokens/sec) - 8 seconds, maximum quality
- **Seedance 2.5** (480p=15, 720p=32, 1080p=78 tokens/sec) - 4-30 seconds, audio included free
- **Seedance 2.0 Mini** (480p=5, 720p=11 tokens/sec) - 4-15 seconds, cheapest video option at 480p
- **Kling V3 / V3 Turbo / O3** (resolution + route based, from 9 tokens/sec) - 3-15 seconds, 4K and native audio on V3/O3
- **Kling O1** (flat 12 tokens/sec) - 5 or 10 seconds, unified generate+edit (image-to-video or video editing, no text-to-video)

### Speech Generation

- **eleven_v3** (default) and **eleven_multilingual_v2** - 11 tokens per 1000 characters
- **eleven_flash_v2_5** - 6 tokens per 1000 characters

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
