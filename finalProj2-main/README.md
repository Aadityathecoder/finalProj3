# finalProj2

Mac Terminal 1: start the server

cd /Users/pl256041/Downloads/finalProj2-main/pwpr/turnAuto
python3 -m uvicorn api:app --host 0.0.0.0 --port 5000
Mac Terminal 2: copy latest motor driver to Pi

scp /Users/pl256041/Downloads/finalProj2-main/pwpr/turnAuto/motorDriver.py pi@192.168.240.29:~/Robot/MotorDriver.py
Mac Terminal 2: SSH into Pi

ssh pi@192.168.240.29
Pi Terminal: run robot client

cd ~/Robot
ROBOT_API_HOST=192.168.240.20 ROBOT_MOTOR_SPEED=10 ROBOT_TURN_INTENSITY=5 python3 MotorDriver.py
Browser on Mac

http://localhost:5000
Press manual arrows first. Then use Start Auto.

