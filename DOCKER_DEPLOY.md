# Docker 部署指南

本文档介绍如何使用 Docker 和 Docker Compose 部署 Amazon Q to Claude API Proxy 服务。

## 前置要求

- Docker Engine 20.10+
- Docker Compose v2+
- 有效的 Amazon Q 凭证

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的凭证：

```bash
AMAZONQ_REFRESH_TOKEN=your_refresh_token_here
AMAZONQ_CLIENT_ID=your_client_id_here
AMAZONQ_CLIENT_SECRET=your_client_secret_here
AMAZONQ_PROFILE_ARN=  # 可选，组织账号需要
PORT=8080
```

### 2. 启动服务

```bash
docker compose up -d
```

### 3. 验证服务

```bash
curl http://localhost:8080/health
```

## 常用命令

```bash
# 启动服务
docker compose up -d

# 停止服务
docker compose down

# 查看日志
docker compose logs -f

# 重启服务
docker compose restart

# 重新构建
docker compose up -d --build
```

## 配置说明

### 端口配置

默认端口为 8080，可通过 `.env` 文件修改：

```bash
PORT=3000
```

### 环境变量

| 变量名 | 必需 | 说明 |
|--------|------|------|
| `AMAZONQ_REFRESH_TOKEN` | ✅ | Amazon Q 刷新令牌 |
| `AMAZONQ_CLIENT_ID` | ✅ | 客户端 ID |
| `AMAZONQ_CLIENT_SECRET` | ✅ | 客户端密钥 |
| `AMAZONQ_PROFILE_ARN` | ❌ | Profile ARN（组织账号） |
| `PORT` | ❌ | 服务端口（默认 8080） |

### 数据持久化

Token 缓存保存在 Docker volume `token_cache` 中，位于容器内 `/home/appuser` 目录。

## 故障排查

### 查看日志

```bash
docker compose logs -f amq2api
```

### 进入容器

```bash
docker compose exec amq2api /bin/bash
```

### 清理并重启

```bash
# 停止并删除容器和 volume
docker compose down -v

# 重新启动
docker compose up -d
```

### 端口冲突

如果端口被占用，修改 `.env` 中的 `PORT` 值。

## 生产环境建议

### 1. 使用反向代理

推荐使用 Nginx 或 Traefik：

```nginx
location / {
    proxy_pass http://localhost:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_read_timeout 300s;
}
```

### 2. 配置日志轮转

在 `docker-compose.yml` 中添加：

```yaml
services:
  amq2api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 3. 资源限制

```yaml
services:
  amq2api:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
```

## 更新服务

```bash
git pull
docker compose up -d --build
```

## 卸载

```bash
# 停止并删除所有资源
docker compose down -v

# 删除镜像
docker rmi amq2api-amq2api
```
