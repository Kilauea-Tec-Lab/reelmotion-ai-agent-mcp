import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock

# 1. Mock dependencies BEFORE importing tools
# Mock request_context
mock_req_ctx = MagicMock()
mock_req_ctx.get_conversation_uuid.return_value = "test-uuid"
mock_req_ctx.get_api_token.return_value = "test-token"
sys.modules["request_context"] = mock_req_ctx

# Mock chatbot
mock_chatbot_module = MagicMock()
mock_chatbot_instance = AsyncMock()
mock_chatbot_module.get_chatbot.return_value = mock_chatbot_instance
sys.modules["chatbot"] = mock_chatbot_module

# 2. Import the function to test
# (This will import the mocked modules above)
from tools import generate_speech

async def test_integration():
    print("=== Testing generate_speech with Backend Callback Logic ===")
    
    # Set Environment for Backend
    # Try to target a likely local backend API (Laravel default port)
    # Adjust this URL if your local backend is running elsewhere (e.g., http://reelmotion.test)
    os.environ["BACKEND_URL"] = "http://127.0.0.1:8000/api" 
    os.environ["API_TOKEN"] = "test-token"
    
    # Ensure ElevenLabs Key is set (using the one from user if not in env)
    if not os.getenv("ELEVENLABS_API_KEY"):
         os.environ["ELEVENLABS_API_KEY"] = "sk_2255a4e8aaeaf2c8211f2ffc968686b602250cd260314f16"

    print("Calling generate_speech...")
    
    # Call the tool
    # The ElevenLabs call will be REAL.
    # The Backend call will fail (connection refused), which we expect to be caught.
    result = await generate_speech("Hello, verifying backend integration.")
    
    print(f"\nResult: {result}")
    
    # Verify Chatbot Interaction
    if mock_chatbot_instance.add_generated_file.call_count > 0:
        print("\n✅ Chatbot.add_generated_file was called successfully.")
        args = mock_chatbot_instance.add_generated_file.call_args[0]
        # args[0] is the url (data uri), args[1] is type
        print(f"   Saved File Type: {args[1]}")
        print(f"   Data URI Length: {len(args[0])} chars")
    else:
        print("\n❌ Chatbot.add_generated_file was NOT called.")

    print("\nCheck the console logs above for 'WARNING: Exception calling backend usage endpoint' to confirm the callback was attempted.")

if __name__ == "__main__":
    asyncio.run(test_integration())
