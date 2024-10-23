@echo off

start /b python modelserver.py

cd client/svelte
start /b npm install
start /b npm run dev