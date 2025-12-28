# Changelog: Soporte para URLs de Archivos

## Fecha: 28 de Diciembre de 2025

## 🎯 Objetivo
Refactorizar el servidor MCP para que acepte **URLs de archivos directamente** en lugar de archivos en base64, optimizando el flujo de datos y reduciendo el procesamiento innecesario.

## 📋 Cambios Realizados

### 1. **server.py** - Endpoint de Chat
**Antes:**
- Recibía archivos binarios vía `multipart/form-data`
- Convertía archivos a base64 para almacenarlos en Redis
- Enviaba base64 a los tools

**Después:**
- Recibe URLs de archivos directamente
- Formato esperado:
  ```
  files[0] = "https://storage.googleapis.com/bucket/file.jpg"
  file_types[0] = "image"
  files[1] = "https://storage.googleapis.com/bucket/video.mp4"
  file_types[1] = "video"
  ```
- Almacena URLs en Redis (mucho más ligero)
- Pasa URLs directamente a los tools

### 2. **chatbot.py** - Gestión de Referencias
**Nuevos métodos:**
- `set_reference_files(file_urls, file_types)` - Almacena URLs de archivos
- `get_reference_files()` - Retorna lista de `{url, type}`
- `clear_reference_files()` - Limpia referencias después de usarlas

**Compatibilidad:**
- Los métodos antiguos (`set_reference_images`, etc.) siguen funcionando
- Se adaptaron para manejar URLs internamente

### 3. **tools.py** - Funciones de Generación
**Cambios en `generate_image`:**
- Ya NO descarga imágenes
- Ya NO convierte a base64
- Envía URLs directamente a Laravel:
  ```json
  {
    "prompt": "texto del usuario",
    "model": "GPT",
    "reference_image": "https://storage.googleapis.com/...",
    "reference_images": ["url1", "url2"]
  }
  ```

**Cambios en `generate_video`:**
- Maneja URLs de imágenes y videos de referencia
- Diferencia entre `reference_image` (URL) y `reference_video` (URL)
- Soporte específico para Runway Aleph (video-to-video)

### 4. **session_manager.py** - Almacenamiento
**Nuevos métodos:**
- `save_reference_files(files_data)` - Guarda lista de `{url, type}`
- `get_reference_files()` - Retorna archivos de referencia

**Formato de almacenamiento en Redis:**
```json
[
  {"url": "https://...", "type": "image"},
  {"url": "https://...", "type": "video"}
]
```

## 🔄 Flujo de Datos (Antes vs Después)

### Antes (Base64):
```
Laravel → Binary File → MCP Server → Base64 → Redis → Base64 → Tools → Base64 → Laravel
```
**Problemas:**
- 33% más tamaño en base64
- Doble conversión innecesaria
- Alto uso de memoria en Redis

### Después (URLs):
```
Laravel → URL → MCP Server → URL → Redis → URL → Tools → URL → Laravel
```
**Beneficios:**
- ✅ Sin conversiones
- ✅ Mínimo uso de memoria
- ✅ Más rápido
- ✅ URLs ya están en Google Cloud Storage

## 📡 Ejemplo de Request desde Laravel

### Formato Form Data:
```php
$data = [
    'message' => 'Genera un video con esta imagen',
    'token' => '1240|...',
    'conversation_uuid' => 'uuid-aqui',
    'files[0]' => 'https://storage.googleapis.com/reelmotion-ai-images/chat_attachments/uuid/imagen.jpg',
    'file_types[0]' => 'image',
];
```

### Formato JSON:
```json
{
  "message": "Genera un video con esta imagen",
  "token": "1240|...",
  "conversation_uuid": "uuid-aqui",
  "files": [
    "https://storage.googleapis.com/reelmotion-ai-images/chat_attachments/uuid/imagen.jpg"
  ],
  "file_types": ["image"]
}
```

## 🎯 Payload a Laravel desde MCP

### Generate Image:
```json
{
  "prompt": "texto exacto del usuario",
  "model": "GPT",
  "type": 1,
  "quantity": 1,
  "reference_image": "https://storage.googleapis.com/..."
}
```

### Generate Video:
```json
{
  "prompt": "texto exacto del usuario",
  "ai_model": "veo-3.1",
  "video_duration": 8,
  "aspect_ratio": "16:9",
  "reference_image": "https://storage.googleapis.com/..."
}
```

## ⚠️ Notas Importantes

1. **URLs válidas:** Deben ser URLs completas (http:// o https://)
2. **Sin base64:** El sistema ya NO acepta base64 en los nuevos flujos
3. **Compatibilidad:** Los métodos legacy siguen funcionando para transición
4. **Limpieza:** Las referencias se limpian automáticamente después de usarse
5. **Redis:** Ahora almacena solo URLs (mucho más eficiente)

## 🧪 Testing

Para probar los cambios:

1. Enviar request con URLs:
   ```bash
   curl -X POST http://localhost/api/chat \
     -F "message=Genera una imagen" \
     -F "token=..." \
     -F "conversation_uuid=uuid" \
     -F "files[0]=https://storage.googleapis.com/..." \
     -F "file_types[0]=image"
   ```

2. Verificar que el MCP envía las URLs directamente a Laravel
3. Confirmar que las imágenes/videos generados se retornan correctamente

## 🚀 Próximos Pasos

- [ ] Actualizar Laravel para aceptar URLs en lugar de base64 (si aún no lo hace)
- [ ] Eliminar código legacy después de confirmar que todo funciona
- [ ] Optimizar el tamaño de Redis con TTL más cortos si es necesario
