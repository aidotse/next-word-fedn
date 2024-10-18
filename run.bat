@echo off

start /b python modelserver.py

cd input
start /b npm install
start /b npm run dev