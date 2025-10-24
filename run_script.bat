@echo off
setlocal

rem バッチファイルの置き場所を作業ディレクトリに（任意）
cd /d "%~dp0"



python scripts\rsl_rl\train.py --task Limo-Pendulum-NoNoise --max_iterations 3 --load_run 2025-10-24_22-32-19 --checkpoint model_4750.pt --resume --headless --video

echo [1/3] Running Limo-Pendulum-NoNoise-R2-64...
python scripts/rsl_rl/train.py --task Limo-Pendulum-NoNoise-R2-64 --max_iterations 3 --headless --video

echo [2/3] Running Limo-Pendulum-NoNoise-R2-32...
python scripts/rsl_rl/train.py --task Limo-Pendulum-NoNoise-R2-32 --max_iterations 3 --headless --video

echo [3/3] Running Limo-Pendulum-NoNoise-R2-128...
python scripts/rsl_rl/train.py --task Limo-Pendulum-NoNoise-R2-128 --max_iterations 3 --headless --video

echo.
echo Done.
pause
