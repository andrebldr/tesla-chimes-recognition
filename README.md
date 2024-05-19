# tesla-chimes-recognition
This is a project to recognize the chimes of a Tesla car. The chimes are used to alert the driver of various events, such as autopilot engaged, autopilot disengaged, etc. The goal of this project is to recognize the chimes with a Raspberry Pi and a microphone. So that the Pi can control other devices.


## Setup
1. Install Python 3.10
2. Install the required packages
```bash
pip install -r requirements.txt
```

## Usage
1. Run training
```bash
python train.py
```
2. Run the test
```bash
python test.py
```


# Resources
- [Tesla Chimes](https://www.reddit.com/r/teslamotors/comments/ehepm1/the_autopilot_chime_sounds_are_awesome_i/)
- [Audio classification](https://medium.com/@oluyaled/audio-classification-using-deep-learning-and-tensorflow-a-step-by-step-guide-5327467ee9ab)