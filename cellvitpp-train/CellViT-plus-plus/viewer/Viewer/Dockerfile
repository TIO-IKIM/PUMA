FROM node:20-alpine

WORKDIR /app

# Copy the entire project to the container
COPY . .

# Install dependencies
RUN npm install

# Expose port 3000 to the outside world
EXPOSE 3000

# Command to run the application
CMD ["npx", "parcel", "index.html", "--host", "0.0.0.0", "--port", "3000"]
