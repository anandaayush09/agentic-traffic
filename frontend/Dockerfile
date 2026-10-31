# ATLAS dashboard: a static React build served by nginx. Two stages -- node builds, nginx serves;
# node, npm and node_modules never reach the final image (a few MB of static files + nginx).
#
# The API address is a RUN-time setting, not baked in: nginx generates /config.js from the
# ATLAS_API_BASE_URL environment variable when the container starts, so the SAME image is promoted
# from staging to production with a different address in each:
#   docker run -p 8080:8080 -e ATLAS_API_BASE_URL=https://api.example.com atlas-web
FROM node:22-alpine AS build
WORKDIR /app
# Dependencies first so this layer only rebuilds when the lockfile changes. `npm ci` installs
# exactly package-lock.json (and fails if it disagrees with package.json) -- unlike `npm install`.
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# nginx-unprivileged runs as a non-root user and listens on 8080 (non-root can't bind <1024).
FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime
# Files in /etc/nginx/templates/*.template are run through envsubst into /etc/nginx/conf.d at start.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
ENV ATLAS_API_BASE_URL=http://localhost:8000
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
