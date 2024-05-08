import numpy as np
import time
# # Shutter parameters (0 = closed, 90 = open; s = signal, C = control, T = transfer)

def find_angles(settings):
    l = len(settings)
    s = np.zeros(l);c = np.zeros(l);t = np.zeros(l);

    exp=['signal','memory','rephasing','dark','control only','transfer only','control and transfer']

    angles = [[90,0,0],[90,90,0],[90,90,90],[0,0,0],[0,90,0],[0,0,90],[0,90,90]]


    for i in range(l):
        setting = settings[i]
        a = exp.index(setting)
        s[i]=int(angles[a][0])
        c[i]=int(angles[a][1])
        t[i]=int(angles[a][2])

    return s, c, t
    
def initialise_shutters(shutterFlag):
    if(shutterFlag):
        from pyfirmata import ArduinoMega, util
        ArduinoPort = 'COM9'
        board = ArduinoMega(ArduinoPort)
        it = util.Iterator(board) # to do with inputs
        it.start()
        time.sleep(0.5)
        servo_Signal_pin = '5'
        servo_C1_pin = '7'
        servo_transfer_pin = '6'
        # servo_Dressing_pin = '8'
        servo_Sig = board.get_pin('d:'+ servo_Signal_pin +':s')
        servo_Control = board.get_pin('d:'+ servo_C1_pin +':s')
        servo_Transfer = board.get_pin('d:'+ servo_transfer_pin +':s')
        servos=[servo_Sig,servo_Control,servo_Transfer]
        return board, servos
    
def go_to_angle(servos,angles):
    for i in range(len(servos)):
        servo = servos[i]
        servo.write(angles[i]);time.sleep(0.2)
       
    
#%% To test:
    
board, servos = initialise_shutters(1)
s,c,t = find_angles(['memory'])
go_to_angle(servos, [s[0],c[0],t[0]])
board.exit()
# %%
