# 🚀 Guía de Despliegue en Google Cloud VM

## Requisitos Previos

- VM de Google Cloud (Ubuntu 20.04+ recomendado)
- Python 3.8 o superior
- Redis instalado
- Acceso SSH a la VM

## 1. Preparar la VM

### Actualizar el sistema

```bash
sudo apt update && sudo apt upgrade -y
```

### Instalar Python y dependencias

```bash
sudo apt install python3 python3-pip python3-venv -y
```

### Instalar Redis

```bash
sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server

# Verificar instalación
redis-cli ping  # Debe responder "PONG"
```

### Configurar Redis (Opcional - para producción)

```bash
sudo nano /etc/redis/redis.conf
```

Configuraciones recomendadas:

```conf
# Persistencia
save 900 1
save 300 10
save 60 10000

# Memoria máxima (ejemplo: 256MB)
maxmemory 256mb
maxmemory-policy allkeys-lru

# Bind a localhost (seguridad)
bind 127.0.0.1
```

Reiniciar Redis:

```bash
sudo systemctl restart redis-server
```

## 2. Clonar y Configurar el Proyecto

### Clonar el repositorio

```bash
cd /home/$USER
git clone <tu-repositorio-url> reelmotion-ai-agent-mcp
cd reelmotion-ai-agent-mcp
```

### Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Configurar variables de entorno

```bash
cp .env.example .env
nano .env
```

Agregar:

```env
GOOGLE_API_KEY=tu_api_key_real
GEMINI_MODEL=gemini-2.5-flash
REDIS_URL=redis://localhost:6379
```

## 3. Configurar como Servicio Systemd

### Crear archivo de servicio

```bash
sudo nano /etc/systemd/system/reelmotion-mcp.service
```

Contenido:

```ini
[Unit]
Description=ReelMotion MCP Server
After=network.target redis-server.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/reelmotion-ai-agent-mcp
Environment="PATH=/home/your_username/reelmotion-ai-agent-mcp/venv/bin"
ExecStart=/home/your_username/reelmotion-ai-agent-mcp/venv/bin/python reelmotion_mcp/server.py http
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**⚠️ Reemplaza `your_username` con tu usuario real.**

### Habilitar e iniciar el servicio

```bash
sudo systemctl daemon-reload
sudo systemctl enable reelmotion-mcp
sudo systemctl start reelmotion-mcp
```

### Verificar estado

```bash
sudo systemctl status reelmotion-mcp
```

### Ver logs

```bash
sudo journalctl -u reelmotion-mcp -f
```

## 4. Configurar Nginx como Reverse Proxy (Opcional)

### Instalar Nginx

```bash
sudo apt install nginx -y
```

### Configurar sitio

```bash
sudo nano /etc/nginx/sites-available/reelmotion
```

Contenido:

```nginx
server {
    listen 80;
    server_name tu-dominio.com;  # O la IP de tu VM

    location /api/chat {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' '*' always;
    }
}
```

### Habilitar sitio

```bash
sudo ln -s /etc/nginx/sites-available/reelmotion /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## 5. Configurar Firewall

```bash
# Permitir SSH, HTTP, HTTPS
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

## 6. Configurar Limpieza Automática de Archivos

### Crear script de limpieza

```bash
nano /home/$USER/reelmotion-ai-agent-mcp/cleanup_files.py
```

Contenido:

```python
import asyncio
import sys
import os
from pathlib import Path

# Agregar el directorio del proyecto al path
sys.path.insert(0, str(Path(__file__).parent / 'reelmotion_mcp'))

from session_manager import get_session_manager

async def cleanup():
    """Limpia archivos expirados."""
    manager = get_session_manager()

    # Limpiar archivos del directorio temp_files más viejos de 2 horas
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(hours=2)

    temp_dir = Path("temp_files")
    if temp_dir.exists():
        for file in temp_dir.iterdir():
            if file.is_file():
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff:
                    file.unlink()
                    print(f"Deleted: {file}")

    print("Cleanup completed")

if __name__ == "__main__":
    asyncio.run(cleanup())
```

### Configurar crontab

```bash
crontab -e
```

Agregar (ejecutar cada hora):

```cron
0 * * * * cd /home/your_username/reelmotion-ai-agent-mcp && /home/your_username/reelmotion-ai-agent-mcp/venv/bin/python cleanup_files.py >> /var/log/reelmotion-cleanup.log 2>&1
```

## 7. Monitoreo y Mantenimiento

### Ver logs del servidor

```bash
sudo journalctl -u reelmotion-mcp -f
```

### Ver logs de limpieza

```bash
tail -f /var/log/reelmotion-cleanup.log
```

### Reiniciar servicio

```bash
sudo systemctl restart reelmotion-mcp
```

### Actualizar código

```bash
cd /home/$USER/reelmotion-ai-agent-mcp
git pull
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart reelmotion-mcp
```

### Monitorear Redis

```bash
# Conectar a Redis
redis-cli

# Ver todas las keys
KEYS *

# Ver info de memoria
INFO memory

# Ver estadísticas
INFO stats
```

## 8. Pruebas

### Probar endpoint local

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hola",
    "conversation_uuid": "test-uuid-123",
    "context": ""
  }'
```

### Probar desde el exterior (si tienes Nginx)

```bash
curl -X POST http://tu-dominio.com/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hola",
    "conversation_uuid": "test-uuid-456",
    "context": ""
  }'
```

## 9. Configuración de SSL/HTTPS (Recomendado)

### Instalar Certbot

```bash
sudo apt install certbot python3-certbot-nginx -y
```

### Obtener certificado

```bash
sudo certbot --nginx -d tu-dominio.com
```

Certbot configurará automáticamente Nginx para HTTPS.

### Renovación automática

```bash
# Probar renovación
sudo certbot renew --dry-run
```

## 📊 Estructura Final en la VM

```
/home/your_username/reelmotion-ai-agent-mcp/
├── reelmotion_mcp/
│   ├── server.py
│   ├── chatbot.py
│   ├── session_manager.py
│   ├── tools.py
│   ├── prompts.py
│   └── request_context.py
├── temp_files/          # Archivos temporales (auto-limpieza)
├── venv/                # Entorno virtual
├── .env                 # Variables de entorno
├── requirements.txt
└── cleanup_files.py     # Script de limpieza
```

## 🔥 Tips de Producción

1. **Logs**: Rotar logs con logrotate
2. **Monitoreo**: Usar herramientas como Prometheus + Grafana
3. **Backups**: Redis puede hacer snapshots automáticos
4. **Escalabilidad**: Considera usar Redis Cluster para múltiples VMs
5. **Seguridad**:
   - Cambia el puerto SSH por defecto
   - Usa autenticación por llave SSH
   - Configura fail2ban
   - Limita acceso a Redis solo localhost

## 🆘 Troubleshooting

### Servicio no inicia

```bash
# Ver logs detallados
sudo journalctl -u reelmotion-mcp -n 50 --no-pager

# Verificar permisos
ls -la /home/$USER/reelmotion-ai-agent-mcp
```

### Redis no conecta

```bash
# Verificar que Redis está corriendo
sudo systemctl status redis-server

# Probar conexión
redis-cli ping
```

### Errores de permisos

```bash
# Asegurar permisos correctos
sudo chown -R $USER:$USER /home/$USER/reelmotion-ai-agent-mcp
```
