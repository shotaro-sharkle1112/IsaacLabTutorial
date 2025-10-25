@echo off
setlocal

rem バッチファイルの置き場所を作業ディレクトリに（任意）
cd /d "%~dp0"



echo [1/3] Running Limo-Pendulum-NoNoise-R2-64...
python scripts\rsl_rl\train.py --task Limo-Pendulum-NoNoise-R2-64 --max_iterations 5000 --headless --video

echo [2/3] Running Limo-Pendulum-NoNoise-R2-32...
python scripts\rsl_rl\train.py --task Limo-Pendulum-NoNoise-R2-32 --max_iterations 5000 --headless --video

echo [3/3] Running Limo-Pendulum-NoNoise-R2-128...
python scripts\rsl_rl\train.py --task Limo-Pendulum-NoNoise-R2-128 --max_iterations 5000 --headless --video

echo.
echo Done.
pause
