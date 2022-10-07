import cv2
import numpy as np
import random
import pyaudio
import wave
import audioop
from glob import glob
from deepface import DeepFace

'''
How to use:

- Run python file
- Look into camera, make sure it only detects 1 face for a few seconds
- Window will freeze, display analysis and elmo should speak
'''

# opencv stuff
faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
# video_capture = cv2.VideoCapture('http://172.16.0.25:4747/video')
video_capture = cv2.VideoCapture(0)
width, height = 320, 240

thresh_count = 0
thresh_requirements = False
pause_frame = []

#audio stuff
chunk = 1024
p = pyaudio.PyAudio()
speech_copy = ''

dominant_race_to_voiceline = {
    'asian':'Sound/Asian/*.wav',
    'indian':'Sound/Asian/*.wav',
    'black':'Sound/Black/*.wav',
    'latino hispanic':'Sound/Hispanic/*.wav',
    'white':'Sound/White/*.wav',
    'middle eastern':'Sound/White/*.wav',
    '?':'Sound/Other/*.wav'
}

voiceline_subtitle = {
    'Sound/Asian/Assembly_Line.wav': 'Oh, an Asian! Elmo remembers you from the assembly line!',
    'Sound/Asian/Iphones.wav':'Take a break from playing with Elmo, Iphones aren\'t going to make themselves!',
    'Sound/Asian/Puppet.wav':'Elmo\'s a puppet, not a dog! Please don\'t try to eat me!',
    'Sound/Black/black.wav':'Elmo respects and admires your rich culture, and will now narrate a documentary, An African American\'s trials and tribulations. Chapter 1.',
    'Sound/Hispanic/hispanic.wav':'Elmo respects all individuals of Hispanic lineage, and wish you a wonderful day.',
    'Sound/Other/No_idea.wav':"Elmo can't make fun of you, because he has no idea what race you are",
    'Sound/Other/What.wav':"White? Asian? Elmo can't tell what you are.",
    'Sound/White/white_meth.wav':'White people like you remind Elmo of cookie monster. But with meth. yeah.',
    'Sound/White/white_south.wav':"You look like you're from the south. Sesame street might be a little too hot for you."
}

# directory of saved images
directory = 'Saved_images'

def generate_random_string():
    res = ''.join(random.choice('1234567890') for _ in range(10))
    return f'{directory}/{res}.png'

elmo1 = cv2.imread('elmo_face/elmo1.png')
elmo2 = cv2.imread('elmo_face/elmo2.png')
elmo_x = 440
elmo_y = 170
elmo_width, elmo_height = 400, 400

while True:
    ret, frame = video_capture.read()
    
    original_frame = frame.copy()
    frame = cv2.resize(frame, (width,height), interpolation = cv2.INTER_AREA)
    tmp = np.zeros((720,1280,3),np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if len(faces) == 1:
        thresh_count += 1
    else:
        thresh_count = 0


    if thresh_requirements:
        tmp = pause_frame[:]
    else:
        # paste frame onto black canvas
        tmp[0:height, 0:width] = frame[:]

    if thresh_count >= 50 and not(thresh_requirements):
        pause_frame = tmp[:] #save pause frame

        # saving image file for memories
        img_name = generate_random_string()
        while img_name in glob(f'{directory}/*.png'):
            img_name = generate_random_string()
        
        # cv2.imwrite(img_name, original_frame)

        # analysing race from face
        try:
            analysis = DeepFace.analyze(original_frame, actions=['race'], prog_bar=False)

        except Exception as e:
            print(e)
            analysis = {
                'race': {
                    'asian': 0, 
                    'indian': 0, 
                    'black': 0, 
                    'white': 0, 
                    'middle eastern': 0, 
                    'latino hispanic': 0
                }, 
                'dominant_race': '?'
            }
            voice_line = random.choice(glob('Sound/Other/*.wav'))
        
        dominant_race = analysis['dominant_race']
        race_analysis = analysis['race']

        voice_line = random.choice(glob(dominant_race_to_voiceline[dominant_race]))

        while voice_line == speech_copy:
            voice_line = random.choice(glob(dominant_race_to_voiceline[dominant_race]))
        
        speech_copy = voice_line # make a copy to avoid repetition

        subtitle = voiceline_subtitle[voice_line.replace('\\','/')]

        textsize = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]

        # get coords based on boundary
        textX = int((pause_frame.shape[1] - textsize[0]) / 2)

        # play voice line
        wf = wave.open(voice_line,'rb')
        stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)
        data = wf.readframes(chunk)

        while data != b'':
            stream.write(data)
            data = wf.readframes(chunk)
            rms = audioop.rms(data,2) # get volume to simulate talking

            if rms >= 500: #open mouth
                tmp[elmo_y:elmo_y+elmo_height, elmo_x:elmo_x+elmo_width] = elmo2
            else: #close mouth
                tmp[elmo_y:elmo_y+elmo_height, elmo_x:elmo_x+elmo_width] = elmo1

            # append results to video
            x_val = 700
            y_val = 25
            increment_val = 27

            for race, confidence in race_analysis.items():
                cv2.putText(tmp, f'{race} : {confidence}', (x_val, y_val), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, 3)
                y_val += increment_val
            cv2.putText(tmp, f'dominant race : {dominant_race}', (x_val, y_val), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, 3)
            cv2.putText(pause_frame, subtitle, (textX, 690), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),1)
            cv2.waitKey(1)
            cv2.imshow('Video', tmp)
        thresh_requirements = False
        thresh_count = 0

    # Paste Elmo's face
    tmp[elmo_y:elmo_y+elmo_height, elmo_x:elmo_x+elmo_width] = elmo1

    # Display the resulting frame
    cv2.imshow('Video', tmp)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
