# Osiris Compute — signaling router + static client in one small Node process.
# Model shards are NOT baked into the image (they're large and optional); fetch
# them at deploy time with ./fetch-demo-model.sh or serve them from object storage.
FROM node:22-alpine

WORKDIR /app

# Install production deps only (just `ws`) against the committed lockfile.
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# App code + browser client.
COPY server.js ./
COPY public ./public

# Bind on all interfaces inside the container; map the port at `docker run`.
ENV HOST=0.0.0.0 PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD wget -qO- http://127.0.0.1:8080/healthz >/dev/null 2>&1 || exit 1

CMD ["node", "server.js"]
