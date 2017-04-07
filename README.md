# Handwriting Recognition
### Arda Mavi - [ardamavi.com](http://www.ardamavi.com/)

Handwriting recognition with neural network

### Using Commend: <br/>
`python3 main.py <ImageFileName>`

### Adding new characters and train dataset:
If you want to adding new character to datasets, you create a directory and rename what you want to add character (like 'a' or 'i').

If you want to add a new training images to previously character datasets or null dataset of character, you add a image(suggested image size: 25x25) to about character directory.

Note: We work on grayscale(2D matrix) image also if you use color images(like RGB), this app automatically return to grayscale from your color image.

### Used Modules:
- sys
- numpy
- PIL (Image)
- os (listdir)
- sklearn (joblib and MLPClassifier)

### Important Notes:
- Suggested image size: 25x25
- Install above modules
