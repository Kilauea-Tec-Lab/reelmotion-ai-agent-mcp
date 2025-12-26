# Pruebas del MCP Server

## 1. Obtener tu API Key de Gemini

Primero, necesitas obtener tu API key de Google Gemini:

- Ve a https://makersuite.google.com/app/apikey
- Crea una nueva API key
- Copia la API key y agrégala al archivo `.env`:
  ```
  GOOGLE_API_KEY=tu_api_key_aqui
  ```

## 2. Instalar dependencias actualizadas

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Probar el Chatbot desde la consola

### Opción A: Prueba automática (mensajes predefinidos)

```bash
python test_chatbot.py
```

### Opción B: Chat interactivo

```bash
python test_chatbot.py --interactive
```

Escribe tus mensajes y presiona Enter. Escribe 'salir' para terminar.

## 4. Probar el servidor MCP completo

### Ejecutar el servidor

```bash
python server.py
```

O usando el CLI de FastMCP:

```bash
fastmcp run server.py:mcp
```

### Probar con el cliente de FastMCP

Crea un archivo `test_mcp_client.py`:

```python
import asyncio
from fastmcp import Client

async def test_chat():
    # Si el servidor corre en STDIO, necesitas usar subprocess
    # Si corre en HTTP (puerto 8000), usa:
    async with Client("http://localhost:8000/mcp") as client:
        # Llamar al tool 'chat'
        result = await client.call_tool(
            name="chat",
            arguments={"message": "Hola, quiero crear una imagen"}
        )
        print("Respuesta:", result)

asyncio.run(test_chat())
```

## 5. Ver herramientas disponibles

```bash
fastmcp dev server.py:mcp
```

Esto abrirá una interfaz de desarrollo donde puedes:

- Ver todas las herramientas disponibles
- Probar cada herramienta interactivamente
- Ver los esquemas de parámetros

## Notas importantes

1. **STDIO vs HTTP**: Por defecto, el servidor corre en modo STDIO (ideal para integrarse con Laravel como subprocess). Para HTTP, modifica `server.py`:

   ```python
   if __name__ == "__main__":
       mcp.run(transport="http", port=8000)
   ```

2. **Variables de entorno**: Asegúrate de que el archivo `.env` tenga todas las variables necesarias.

3. **Errores comunes**:
   - "API key not found": Verifica que `GOOGLE_API_KEY` esté en `.env`
   - "Module not found": Ejecuta `pip install -r requirements.txt`
