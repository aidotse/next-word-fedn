#!/bin/bash

cd input
npm install &
npm run dev &

cd ..

python3 modelserver.py &