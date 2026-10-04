# ---------- Stage 1: build the static React bundle ----------
FROM node:20-alpine AS build
WORKDIR /app
# Install dependencies first (better layer caching)
COPY package.json package-lock.json ./
RUN npm ci
# Copy the sources and build into /app/dist
COPY . .
RUN npm run build

# ---------- Stage 2: serve it with nginx ----------
FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
