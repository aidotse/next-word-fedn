@echo off
cd input

start /b npm install
start /b npm run dev

cd ..
start /b python modelserver.py