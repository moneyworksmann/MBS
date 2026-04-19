#!/bin/bash
PORT=${PORT:-8000}
python3 -m http.server $PORT
