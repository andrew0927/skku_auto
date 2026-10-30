"""The tested driving.ino accepts s<steer>l<left>r<right> followed by newline."""


def convert_serial_message(steering, left_speed, right_speed):
    steering = max(-7, min(7, int(steering)))
    left_speed = max(-255, min(255, int(left_speed)))
    right_speed = max(-255, min(255, int(right_speed)))
    message = f's{steering}l{left_speed}r{right_speed}\n'
    # driving.ino has a 15-byte buffer including the terminating null.
    if len(message.rstrip('\n')) > 14:
        raise ValueError('Command exceeds the Arduino input buffer')
    return message
