#!/bin/bash

# Exit script when any command fails
set -e

# Change to the script's directory
cd "$(dirname "$0")"

echo "==================================="
echo "  Starting Brain2Text Services..."
echo "==================================="

# Start the Flask backend
echo "-> Starting Backend (Flask)..."
cd server
python3 app.py &
BACKEND_PID=$!
cd ..

# Wait a second to ensure backend starts
sleep 2

# Start the Vite frontend
echo "-> Starting Frontend (Vite)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "==================================="
echo "  Services are running!"
echo "  Backend PID:  $BACKEND_PID"
echo "  Frontend PID: $FRONTEND_PID"
echo "  Press Ctrl+C to stop both."
echo "==================================="

# Function to handle shutdown
cleanup() {
    echo ""
    echo "==================================="
    echo "  Shutting down services..."
    echo "==================================="
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "Done."
    exit 0
}

# Trap termination signals
trap cleanup SIGINT SIGTERM

# Wait indefinitely for the background processes
wait
