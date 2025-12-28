# 📖 Guía de Uso: Envío de Archivos por URL al MCP

## Ejemplo desde Laravel (PHP)

### Opción 1: Form Data (Recomendado)

```php
use Illuminate\Support\Facades\Http;

// Enviar mensaje con archivo adjunto
$response = Http::asForm()->post('http://localhost/api/chat', [
    'message' => 'Usa esta imagen y quitale al vato el refresco, vas a generar la imagen con GPT',
    'token' => '1240|qcYl1I4TLSryHhYt64HV4pHqgEzT8n5YcNx8sX0fb097a455',
    'conversation_uuid' => 'a0b3d279-1484-42a5-87f8-19bced13b97c',
    'chat_id' => 'a0b3d279-1484-42a5-87f8-19bced13b97c',
    'files[0]' => 'https://storage.googleapis.com/reelmotion-ai-images/chat_attachments/a0b3d279-1484-42a5-87f8-19bced13b97c/69516a7b7d18e_1766943355.jpg',
    'file_types[0]' => 'image',
]);

$data = $response->json();
echo $data['response']; // Respuesta del chatbot
print_r($data['files']); // Archivos generados
```

### Opción 2: JSON

```php
use Illuminate\Support\Facades\Http;

$response = Http::withHeaders([
    'Content-Type' => 'application/json',
])->post('http://localhost/api/chat', [
    'message' => 'Genera un video con esta imagen',
    'token' => '1240|qcYl1I4TLSryHhYt64HV4pHqgEzT8n5YcNx8sX0fb097a455',
    'conversation_uuid' => 'a0b3d279-1484-42a5-87f8-19bced13b97c',
    'files' => [
        'https://storage.googleapis.com/reelmotion-ai-images/chat_attachments/uuid/imagen.jpg'
    ],
    'file_types' => ['image']
]);
```

### Múltiples Archivos

```php
$response = Http::asForm()->post('http://localhost/api/chat', [
    'message' => 'Edita este video usando esta imagen de referencia',
    'token' => $token,
    'conversation_uuid' => $uuid,

    // Primera referencia: imagen
    'files[0]' => 'https://storage.googleapis.com/bucket/reference.jpg',
    'file_types[0]' => 'image',

    // Segunda referencia: video
    'files[1]' => 'https://storage.googleapis.com/bucket/source_video.mp4',
    'file_types[1]' => 'video',
]);
```

## Ejemplo desde cURL

### Un solo archivo

```bash
curl -X POST http://localhost/api/chat \
  -F "message=Genera una imagen con este estilo" \
  -F "token=1240|..." \
  -F "conversation_uuid=a0b3d279-1484-42a5-87f8-19bced13b97c" \
  -F "files[0]=https://storage.googleapis.com/reelmotion-ai-images/reference.jpg" \
  -F "file_types[0]=image"
```

### Múltiples archivos

```bash
curl -X POST http://localhost/api/chat \
  -F "message=Combina estas imágenes" \
  -F "token=1240|..." \
  -F "conversation_uuid=uuid-aqui" \
  -F "files[0]=https://storage.googleapis.com/bucket/img1.jpg" \
  -F "file_types[0]=image" \
  -F "files[1]=https://storage.googleapis.com/bucket/img2.jpg" \
  -F "file_types[1]=image"
```

### JSON

```bash
curl -X POST http://localhost/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Genera un video",
    "token": "1240|...",
    "conversation_uuid": "uuid-aqui",
    "files": [
      "https://storage.googleapis.com/bucket/reference.jpg"
    ],
    "file_types": ["image"]
  }'
```

## Respuesta del Servidor

```json
{
  "response": "¡Por supuesto! Voy a generar la imagen con GPT quitándole el refresco al vato. Esto costará 10 tokens (10 tokens/imagen × 1). ¿Confirmas?",
  "files": []
}
```

Después de confirmar:

```json
{
  "response": "¡Listo! La imagen ha sido generada exitosamente.",
  "files": [
    {
      "url": "https://storage.googleapis.com/reelmotion-generated/output_123.jpg",
      "type": "image"
    }
  ]
}
```

## Tipos de Archivos Soportados

| `file_types[n]` | Descripción             | Uso                                           |
| --------------- | ----------------------- | --------------------------------------------- |
| `image`         | Imagen (JPG, PNG, WebP) | Referencia para generación de imágenes/videos |
| `video`         | Video (MP4, WebM)       | Referencia para video-to-video (Runway Aleph) |
| `audio`         | Audio (futuro)          | No implementado aún                           |
| `document`      | Documento (futuro)      | No implementado aún                           |

## Flujo Completo

1. **Laravel sube el archivo a Google Cloud Storage**

   ```php
   $path = Storage::disk('gcs')->put(
       "chat_attachments/{$uuid}/",
       $request->file('attachment')
   );
   $url = Storage::disk('gcs')->url($path);
   ```

2. **Laravel envía la URL al MCP**

   ```php
   $response = Http::asForm()->post('http://mcp-server/api/chat', [
       'message' => $request->input('message'),
       'token' => $request->user()->currentAccessToken()->plainTextToken,
       'conversation_uuid' => $uuid,
       'files[0]' => $url,
       'file_types[0]' => 'image',
   ]);
   ```

3. **MCP procesa con Gemini y llama a los tools**

   - Gemini decide si usar `generate_image` o `generate_video`
   - El tool recibe la URL y la envía a Laravel
   - Laravel procesa y retorna el resultado

4. **MCP devuelve la respuesta a Laravel**

   ```json
   {
     "response": "Texto de Gemini",
     "files": [{ "url": "...", "type": "image" }]
   }
   ```

5. **Laravel envía la respuesta al frontend**
   - Puede guardar las URLs en la BD
   - Puede enviarlas directamente al usuario

## Ventajas de este Enfoque

✅ **No hay conversión a base64** (ahorro de CPU)  
✅ **Menor uso de memoria** (solo URLs en Redis)  
✅ **Más rápido** (sin descarga/subida innecesaria)  
✅ **Archivos ya en Cloud Storage** (persistentes)  
✅ **Cacheable** (las URLs son estables)

## Migración desde Base64

Si actualmente envías base64, necesitas:

1. **Guardar el archivo en storage primero:**

   ```php
   // Antes (base64):
   $base64 = base64_encode(file_get_contents($file));

   // Después (URL):
   $path = Storage::disk('gcs')->put('chat_attachments/', $file);
   $url = Storage::disk('gcs')->url($path);
   ```

2. **Enviar la URL:**

   ```php
   // Antes:
   'image_base64' => $base64,

   // Después:
   'files[0]' => $url,
   'file_types[0]' => 'image',
   ```

## Debug / Troubleshooting

Para ver los logs del MCP:

```bash
docker-compose logs -f api
```

Buscar líneas como:

```
DEBUG: Retrieved 1 files from chatbot session
DEBUG: Sending request with 1 image URLs
DEBUG: Using reference image URL: https://storage.googleapis.com/...
```
