"""
Script de prueba para el chatbot de Gemini.
Puedes ejecutarlo directamente desde la consola para probar el chatbot.
"""
import asyncio
from chatbot import GeminiChatbot

async def test_chatbot():
    """Test the Gemini chatbot with a simple conversation."""
    print("=== Iniciando prueba del Chatbot Gemini ===\n")
    
    # Create chatbot instance
    chatbot = GeminiChatbot()
    chatbot.start_chat()
    
    # Test messages
    test_messages = [
        "Hola, ¿cómo estás?",
        "¿Puedes ayudarme a generar una imagen de un gato espacial?",
        "¿Y qué tal un video de una ciudad futurista?",
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"Usuario: {message}")
        response = await chatbot.send_message(message)
        print(f"Gemini: {response}\n")
        print("-" * 80 + "\n")
    
    print("=== Prueba completada ===")

async def interactive_chat():
    """Interactive chat session with Gemini."""
    print("=== Chat Interactivo con Gemini ===")
    print("Escribe 'salir' para terminar la conversación\n")
    
    chatbot = GeminiChatbot()
    chatbot.start_chat()
    
    while True:
        try:
            user_input = input("Tú: ")
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break
            
            if not user_input.strip():
                continue
            
            response = await chatbot.send_message(user_input)
            print(f"Gemini: {response}\n")
            
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    # Check if interactive mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_chat())
    else:
        asyncio.run(test_chatbot())
