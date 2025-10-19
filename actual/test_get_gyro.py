from mpu6050 import mpu6050
sensor = mpu6050(0x68, bus=0)
while True:
   print(sensor.get_accel_data())