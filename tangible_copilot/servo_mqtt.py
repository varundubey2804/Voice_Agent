import time
import json
import queue
import threading
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
import paho.mqtt.client as mqtt

# --- Configuration ---
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC = "tangible/lamp/motion"

# --- PCA9685 & Servo Setup ---
print("Initializing I2C and PCA9685...")
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

servo_lr = servo.Servo(pca.channels[0], min_pulse=500, max_pulse=2500)
servo_ud = servo.Servo(pca.channels[1], min_pulse=500, max_pulse=2500)

ANGLE_LR_MID = 90
ANGLE_LR_LEFT = 60
ANGLE_LR_RIGHT = 120

ANGLE_UD_AWAKE = 65
ANGLE_UD_UP = 90
ANGLE_UD_SLEEP = 60

# Single lock so only one motion routine ever drives the servos at a time.
_servo_lock = threading.Lock()


def move_servo(servo_obj, target_angle, speed='normal'):
    """Smoothly interpolate a servo to a target angle."""
    if servo_obj.angle is None:
        servo_obj.angle = target_angle
        time.sleep(0.5)
        return

    current_angle = int(servo_obj.angle)
    diff = target_angle - current_angle
    if diff == 0:
        return

    if speed == 'slow':
        steps = max(30, int(abs(diff) * 3))
        delay = 0.04
    else:
        steps = max(10, int(abs(diff)))
        delay = 0.02

    step_size = diff / steps
    for i in range(1, steps + 1):
        try:
            next_angle = current_angle + (step_size * i)
            next_angle = max(0, min(180, next_angle))
            servo_obj.angle = next_angle
        except ValueError:
            pass
        time.sleep(delay)


# --- Motion Presets (run under the lock, sequentially, never overlapping) ---

def action_ack_nod():
    move_servo(servo_lr, ANGLE_LR_MID, speed='slow')
    move_servo(servo_ud, 75, speed='normal')
    time.sleep(0.2)
    move_servo(servo_ud, ANGLE_UD_UP, speed='normal')


def action_look_left_right():
    move_servo(servo_ud, ANGLE_UD_AWAKE, speed='normal')
    time.sleep(0.2)
    move_servo(servo_lr, ANGLE_LR_LEFT, speed='slow')
    time.sleep(0.3)
    move_servo(servo_lr, ANGLE_LR_RIGHT, speed='slow')
    time.sleep(0.3)
    move_servo(servo_lr, ANGLE_LR_MID, speed='slow')


def action_wake_and_nod():
    move_servo(servo_lr, ANGLE_LR_MID, speed='slow')
    move_servo(servo_ud, 90, speed='normal')
    time.sleep(0.3)
    move_servo(servo_ud, 75, speed='normal')
    time.sleep(0.2)
    move_servo(servo_ud, ANGLE_UD_UP, speed='normal')


def action_sleep():
    move_servo(servo_lr, ANGLE_LR_MID, speed='slow')
    time.sleep(0.5)
    move_servo(servo_ud, ANGLE_UD_SLEEP, speed='normal')


MOTION_HANDLERS = {
    "ack_nod": action_ack_nod,
    "look_left_right": action_look_left_right,
    "wake_and_nod": action_wake_and_nod,
    "sleep": action_sleep,
}

# --- Single worker thread + queue: motions run one at a time, in order,
# so a fast burst of MQTT messages can't fight over the same servo. ---
_motion_queue: "queue.Queue[str]" = queue.Queue(maxsize=4)


def _motion_worker():
    while True:
        motion = _motion_queue.get()
        handler = MOTION_HANDLERS.get(motion)
        if handler is None:
            print(f"Unknown motion '{motion}', skipping")
            continue
        with _servo_lock:
            try:
                handler()
            except Exception as e:
                print(f"Motion '{motion}' failed: {e}")


threading.Thread(target=_motion_worker, daemon=True).start()


def enqueue_motion(motion: str):
    try:
        _motion_queue.put_nowait(motion)
    except queue.Full:
        # Drop the oldest queued motion rather than blocking the MQTT thread
        try:
            _motion_queue.get_nowait()
        except queue.Empty:
            pass
        _motion_queue.put_nowait(motion)


# --- MQTT Network Listener ---

def _make_client():
    # Pin to VERSION1 explicitly. paho-mqtt 2.x's VERSION2 API changes the
    # on_connect/on_disconnect callback signatures (ConnectFlags/ReasonCode
    # objects instead of plain ints) — VERSION1 keeps the old signatures our
    # callbacks below actually expect, on both paho-mqtt 1.x and 2.x installs.
    try:
        return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        # paho-mqtt 1.x has no callback_api_version param at all
        return mqtt.Client()


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker. Status: {rc}")
    client.subscribe(MQTT_TOPIC, qos=1)


def on_disconnect(client, userdata, rc):
    print(f"Disconnected from MQTT broker (rc={rc}), will auto-reconnect")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        motion = payload.get("motion")
        if motion:
            enqueue_motion(motion)
    except Exception as e:
        print(f"Error processing message: {e}")


if __name__ == "__main__":
    print("Calibrating servos to rest position...")
    try:
        servo_lr.angle = ANGLE_LR_MID
        servo_ud.angle = ANGLE_UD_UP
    except Exception as e:
        print(f"Servo init error: {e}")
    time.sleep(1)

    client = _make_client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    print(f"Connecting to local MQTT broker at {MQTT_BROKER}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print("Connected! Waiting for commands from Snapdragon PC...")
        client.loop_forever(retry_first_connection=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        enqueue_motion("sleep")
        time.sleep(2)
        pca.deinit()
    except Exception as e:
        print(f"Connection failed: {e}. Is Mosquitto running on the Pi?")
