#!/bin/bash

cd input
npm install &
npm run dev &

cd ..

python modelserver.py &