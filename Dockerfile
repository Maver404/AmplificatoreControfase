# syntax=docker/dockerfile:1
# Requires BuildKit (default in Docker Desktop and Docker Engine >= 23) for COPY --chmod and
# the npm cache mount.

# ---- Build stage: install, run the unit tests, build the static site ------------------------
FROM node:24-alpine AS build
WORKDIR /app
# Browsers are not needed here: the Playwright e2e tests run outside Docker.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --no-audit --no-fund

COPY . .
# A failing unit test (DSP, safety, presets, validation) stops the image build.
RUN npm test
RUN npm run build

# ---- Runtime stage: nginx serving /dist on 80 (HTTP) and 443 (HTTPS) --------------------------
# Pinned stable release; keep the official ENTRYPOINT/CMD, which run /docker-entrypoint.d/*.sh.
FROM nginx:1.30.6-alpine

# openssl generates the self-signed certificate at start-up (it is not part of the image);
# curl, used by the compose healthcheck, already is.
RUN apk add --no-cache openssl \
    && rm -f /etc/nginx/conf.d/default.conf \
    && mkdir -p /etc/nginx/snippets /etc/nginx/certs

COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/security-headers.conf /etc/nginx/snippets/security-headers.conf
COPY --chmod=0755 docker/05-selfsigned-cert.sh /docker-entrypoint.d/05-selfsigned-cert.sh
COPY --from=build /app/dist /usr/share/nginx/html

# Strip CR in case the repository was checked out with Windows line endings (a "#!/bin/sh\r"
# shebang would stop the container from starting). Then validate the whole setup at build time:
# run the certificate script once and `nginx -t`, and delete that throwaway certificate in the
# same layer so no private key is ever baked into the image.
RUN cr="$(printf '\r')" \
    && sed -i "s/${cr}\$//" /docker-entrypoint.d/05-selfsigned-cert.sh /etc/nginx/nginx.conf \
        /etc/nginx/snippets/security-headers.conf \
    && chmod 0755 /docker-entrypoint.d/05-selfsigned-cert.sh \
    && /docker-entrypoint.d/05-selfsigned-cert.sh \
    && nginx -t \
    && rm -rf /etc/nginx/certs-generated /etc/nginx/certs-active

EXPOSE 80 443
