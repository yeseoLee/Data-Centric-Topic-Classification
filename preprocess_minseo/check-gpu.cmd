@echo off
:loop
cls

echo ===== GPU Information =====
nvidia-smi --query-gpu=name,index --format=csv,noheader

echo.
echo ===== Real-time GPU Usage =====
nvidia-smi | findstr "Default"



echo.
echo ===== Real-time Memory Usage (GB) =====
for /f "tokens=2 delims==" %%A in ('wmic OS get FreePhysicalMemory /Value') do set FreeMem=%%A
for /f "tokens=2 delims==" %%A in ('wmic OS get TotalVisibleMemorySize /Value') do set TotalMem=%%A

set /a FreeMemGB=%FreeMem% / 1024 / 1024
set /a TotalMemGB=%TotalMem% / 1024 / 1024
set /a UsedMemGB=%TotalMemGB% - %FreeMemGB%

echo Total Memory: %TotalMemGB% GB
echo Used Memory : %UsedMemGB% GB
echo Free Memory : %FreeMemGB% GB

timeout /t 1 >nul
goto loop
