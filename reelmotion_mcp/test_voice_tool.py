import os
import asyncio
import httpx
import base64

async def test_elevenlabs():
    text = "Hola, esto es una prueba de generación de voz con ElevenLabs desde reelmotion."
    voice_id = "21m00Tcm4TlvDq8ikWAM" # Rachel
    # Using the key provided by the user
    api_key = os.getenv("ELEVENLABS_API_KEY", "sk_2255a4e8aaeaf2c8211f2ffc968686b602250cd260314f16")
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    print(f"Testing ElevenLabs API with Voice ID: {voice_id}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                print("SUCCESS! Audio generated.")
                print(f"Content length: {len(response.content)} bytes")
                
                # Save to file to verify
                with open("test_output.mp3", "wb") as f:
                    f.write(response.content)
                print("Saved locally to test_output.mp3")
                
                # Verify base64 conversion works (same as in implementation)
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                print(f"Base64 prefix: {audio_base64[:50]}...")
            else:
                print(f"FAILED. Status: {response.status_code}")
                print(f"Response: {response.text}")
    except Exception as e:
        print(f"Exception happened: {e}")

if __name__ == "__main__":
    asyncio.run(test_elevenlabs())
